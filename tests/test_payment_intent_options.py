"""Focused contracts for request-scoped Stripe PaymentIntent input."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import stripe

from mpp import Challenge, Credential, Receipt
from mpp.errors import BadRequestError, VerificationFailedError
from mpp.methods.stripe import ChargeIntent, PaymentIntentContext
from mpp.methods.stripe.payment_intent_options import validate_options
from mpp.server import Mpp, Validation
from mpp.server.compose import ComposedChallenges
from tests.test_stripe_machine_payments import TEMPO_ADDRESS, make_payments

OPTIONS = {
    "customer": "cus_test",
    "receipt_email": "buyer@example.com",
    "metadata": {"order_id": "order_test"},
    "hooks": {"inputs": {"tax": {"calculation": "taxcalc_test"}}},
}


class Rail:
    name = "charge"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def validate(self, credential: Credential, request: dict[str, Any]) -> Validation:
        self.calls.append("validate")
        if not credential.payload.get("valid"):
            raise VerificationFailedError("invalid credential")
        return Validation(credential, {}, self.name, dict(request))

    async def broadcast(self, credential: Credential, request: dict[str, Any]) -> Receipt:
        self.calls.append("broadcast")
        return Receipt.success("0xtest", method="tempo")


@pytest.fixture(params=["stripe", "tempo"])
def payment(request):
    client, payments = make_payments(
        deposit_addresses={"tempo": TEMPO_ADDRESS}, metadata={"configured": "yes"}
    )
    method = payments.spt.charge() if request.param == "stripe" else payments.tempo.charge()
    rail = Rail()
    if request.param == "tempo":
        method.intents["charge"] = rail
    return Mpp.create(method=method, realm="example.com", secret_key="secret"), client, rail


def credential(challenge: Challenge, **payload: Any) -> Credential:
    return Credential(challenge.to_echo(), payload or {"spt": "spt_test", "valid": True})


async def issue(server: Mpp, options: Any = None) -> Challenge:
    result = await server.charge(None, "0.50", payment_intent_options=options)
    assert isinstance(result, Challenge)
    return result


async def test_static_options_are_private_and_propagate_with_analytics(payment):
    server, client, rail = payment
    for options in (None, OPTIONS):
        challenge = await issue(server, options)
        assert "payment_intent_options" not in challenge.request
        await server.charge(
            credential(challenge).to_authorization(), "0.50", payment_intent_options=options
        )
        params, _ = client.payment_intents.calls[-1]
        assert {
            key: params["metadata"][key]
            for key in ("machine_payment", "mpp_sdk", "mpp_challenge_id", "mpp_intent")
        } == {
            "machine_payment": "true",
            "mpp_sdk": f"pympp/{version('pympp')}",
            "mpp_challenge_id": challenge.id,
            "mpp_intent": "charge",
        }
        if options:
            assert {key: params[key] for key in ("customer", "receipt_email", "hooks")} == {
                key: options[key] for key in ("customer", "receipt_email", "hooks")
            }
            assert params["metadata"]["order_id"] == "order_test"
    assert len(client.payment_intents.calls) == 2
    if server.method.name == "tempo":
        assert rail.calls == ["validate", "broadcast", "validate", "broadcast"]


async def test_deferred_resolver_skips_unpaid_and_rejected_credentials(payment):
    server, client, rail = payment
    resolver = AsyncMock(return_value=OPTIONS)
    challenge = await issue(server, resolver)
    expired = await server.charge(
        None, "0.50", expires=(datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    )
    assert isinstance(expired, Challenge)
    rejected = (
        "Payment broken",
        credential(replace(challenge, id="forged")).to_authorization(),
        credential(expired).to_authorization(),
    )
    for authorization in rejected:
        result = await server.charge(authorization, "0.50", payment_intent_options=resolver)
        assert isinstance(result, Challenge)
    with pytest.raises(VerificationFailedError):
        await server.charge(
            credential(challenge, spt="").to_authorization(),
            "0.50",
            payment_intent_options=resolver,
        )
    resolver.assert_not_called()
    assert not client.payment_intents.calls


async def test_resolver_runs_before_spt_payment_intent_or_tempo_broadcast(payment):
    server, client, rail = payment

    async def resolve(context: PaymentIntentContext):
        assert context.challenge.id == challenge.id and context.request == challenge.request
        assert not client.payment_intents.calls
        if server.method.name == "tempo":
            assert rail.calls == ["validate"]
        raise BadRequestError("invalid tax country")

    challenge = await issue(server, resolve)
    with pytest.raises(BadRequestError):
        await server.charge(
            credential(challenge).to_authorization(), "0.50", payment_intent_options=resolve
        )
    assert not client.payment_intents.calls and "broadcast" not in rail.calls


@pytest.mark.parametrize(
    "value",
    [
        {"amount": 1},
        {"confirm": False},
        {"idempotency_key": "x"},
        {"customer": ""},
        {"receipt_email": " "},
        {"metadata": {"bad": 1}},
        {"hooks": {"inputs": {"tax": {"calculation": ""}}}},
    ],
)
def test_options_reject_uncontrolled_or_invalid_values(value):
    with pytest.raises(BadRequestError):
        validate_options(value)


@pytest.mark.parametrize(
    "error,retries",
    [
        (stripe.InvalidRequestError("bad customer", "customer"), 1),
        (stripe.InvalidRequestError("bad amount", "amount"), 0),
        (stripe.APIConnectionError("timeout"), 0),
    ],
)
async def test_crypto_fallback_is_once_and_only_for_optional_field_rejection(error, retries):
    client, payments = make_payments(deposit_addresses={"tempo": TEMPO_ADDRESS})
    method, rail = payments.tempo.charge(), Rail()
    method.intents["charge"] = rail
    server = Mpp.create(method=method, realm="example.com", secret_key="secret")
    calls = []

    async def create(params, *, options):
        calls.append((params, options))
        if len(calls) == 1:
            raise error
        return await client.payment_intents.create(params, options=options)

    client.payment_intents.create_async = create
    challenge = await issue(server, OPTIONS)
    await server.charge(
        credential(challenge).to_authorization(), "0.50", payment_intent_options=OPTIONS
    )
    assert len(calls) == retries + 1
    if retries:
        assert calls[1][1]["idempotency_key"] == "0xtest_fallback"
        assert not {"customer", "receipt_email", "hooks"} & calls[1][0].keys()
        assert "configured" not in calls[1][0]["metadata"]


async def test_compose_and_direct_broadcast_accept_private_options(payment):
    server, client, rail = payment
    resolver = AsyncMock(return_value=OPTIONS)
    handler = server.compose(
        (server.method, {"amount": "0.50", "payment_intent_options": resolver})
    )
    unpaid = await handler.verify(None)
    assert isinstance(unpaid, ComposedChallenges)
    paid = credential(unpaid.challenges[0])
    await server.broadcast_credential(paid, payment_intent_options=resolver)
    await handler.verify(paid.to_authorization())
    assert resolver.await_count == 2 and len(client.payment_intents.calls) == 2


async def test_attempt_options_are_isolated(payment):
    server, client, rail = payment
    challenge = await issue(server)
    await asyncio.gather(
        *(
            server.charge(
                credential(challenge).to_authorization(),
                "0.50",
                payment_intent_options={"customer": customer},
            )
            for customer in ("cus_a", "cus_b")
        )
    )
    assert {params["customer"] for params, _ in client.payment_intents.calls} == {"cus_a", "cus_b"}


async def test_raw_http_spt_supports_options_without_owning_injected_client():
    calls = []

    def respond(request):
        calls.append(request.content.decode())
        return httpx.Response(200, json={"id": "pi", "status": "succeeded"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        intent = ChargeIntent(secret_key="sk_test", http_client=http)
        _, payments = make_payments()
        method = payments.spt.charge()
        method.intents["charge"] = intent
        server = Mpp.create(method=method, realm="example.com", secret_key="secret")
        challenge = await issue(server, OPTIONS)
        await server.charge(
            credential(challenge).to_authorization(), "0.50", payment_intent_options=OPTIONS
        )
        assert "hooks%5Binputs%5D%5Btax%5D%5Bcalculation%5D=taxcalc_test" in calls[0]
        assert intent._http_client is http and not http.is_closed


async def test_raw_http_spt_failure_does_not_expose_private_options():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                400, json={"error": {"message": "No such customer: cus_private"}}
            )
        )
    ) as http:
        intent = ChargeIntent(secret_key="sk_test", http_client=http)
        _, payments = make_payments()
        method = payments.spt.charge()
        method.intents["charge"] = intent
        server = Mpp.create(method=method, realm="example.com", secret_key="secret")
        challenge = await issue(server, OPTIONS)
        with pytest.raises(VerificationFailedError) as error:
            await server.charge(
                credential(challenge).to_authorization(), "0.50", payment_intent_options=OPTIONS
            )
        assert "cus_private" not in str(error.value)


async def test_legacy_crypto_rail_remains_supported():
    client, payments = make_payments(deposit_addresses={"tempo": TEMPO_ADDRESS})
    method, calls = payments.tempo.charge(), []

    class Legacy:
        name = "charge"

        async def verify(self, credential, request):
            calls.append("verify")
            return Receipt.success("0xlegacy", method="tempo")

    method.intents["charge"] = Legacy()
    server = Mpp.create(method=method, realm="example.com", secret_key="secret")

    def resolver(context):
        calls.append("resolve")
        return None

    challenge = await issue(server, resolver)
    await server.charge(
        credential(challenge).to_authorization(), "0.50", payment_intent_options=resolver
    )
    assert calls == ["resolve", "verify"] and len(client.payment_intents.calls) == 1


async def test_replacing_the_existing_tempo_callback_preserves_event_behavior():
    client, payments = make_payments(deposit_addresses={"tempo": TEMPO_ADDRESS})
    method, callback = payments.tempo.charge(), AsyncMock()
    method.on_payment_success = callback
    server = Mpp.create(method=method, realm="example.com", secret_key="secret")
    challenge = await issue(server, {"metadata": {"order": "123"}})

    # The actual rail rejects this malformed transaction before either callback;
    # use a legacy intent to isolate callback registration behavior.
    class Legacy:
        name = "charge"

        async def verify(self, credential, request):
            return Receipt.success("0xcallback", method="tempo")

    method.intents["charge"] = Legacy()
    await server.charge(
        credential(challenge).to_authorization(),
        "0.50",
        payment_intent_options={"metadata": {"order": "123"}},
    )
    callback.assert_awaited_once()
    assert client.payment_intents.calls[0][0]["metadata"]["order"] == "123"
