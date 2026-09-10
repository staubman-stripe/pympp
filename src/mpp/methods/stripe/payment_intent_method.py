"""Decorate a crypto rail with private, request-scoped Stripe input."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from mpp import Credential, Receipt
from mpp.methods.stripe.payment_intent_options import (
    PaymentIntentInput,
    prepare_options,
    resolve_options,
)
from mpp.server.intent import Intent, VerifiableIntent, broadcast_credential

RecordPayment = Callable[[Credential, dict[str, Any], Receipt, dict[str, Any]], Awaitable[None]]


def with_payment_intent_input(method: Any, record_payment: RecordPayment) -> Any:
    """Decorate a method in place without changing its public concrete type."""

    built_in_callback = method.on_payment_success

    def prepare_intent(
        intent: Intent | VerifiableIntent, input: dict[str, Any]
    ) -> tuple[Intent | VerifiableIntent, dict[str, Any]]:
        options = prepare_options(input.pop("payment_intent_options", None))
        # Retain the existing ability to replace the event callback: in that
        # case generic event dispatch owns it and this wrapper does no recording.
        callback = record_payment if method.on_payment_success is built_in_callback else None
        wrapper = (
            _VerifiablePaymentIntentIntent(intent, options, callback)
            if isinstance(intent, VerifiableIntent)
            else _PaymentIntentIntent(intent, options, callback)
        )
        return wrapper, input

    method.prepare_intent = prepare_intent
    # Generic server plumbing uses this only to avoid duplicating the existing
    # success callback: the prepared intent invokes it after settlement.
    method._handled_payment_success_callback = built_in_callback
    return method


class _PaymentIntentIntent:
    def __init__(
        self,
        intent: Intent | VerifiableIntent,
        options: PaymentIntentInput,
        record_payment: RecordPayment | None,
    ) -> None:
        self.name = intent.name
        self._intent = intent
        self._options = options
        self._record_payment = record_payment

    async def _record(
        self, credential: Credential, request: dict[str, Any], receipt: Receipt
    ) -> Receipt:
        options = await resolve_options(self._options, credential, request)
        if self._record_payment is not None:
            await self._record_payment(credential, request, receipt, options)
        return receipt

    async def verify(self, credential: Credential, request: dict[str, Any]) -> Receipt:
        # Legacy rails have no non-mutating validation to run before resolution.
        options = await resolve_options(self._options, credential, request)
        receipt = await cast(Intent, self._intent).verify(credential, request)
        if self._record_payment is not None:
            await self._record_payment(credential, request, receipt, options)
        return receipt


class _VerifiablePaymentIntentIntent(_PaymentIntentIntent):
    async def validate(self, credential: Credential, request: dict[str, Any]):
        assert isinstance(self._intent, VerifiableIntent)
        return await self._intent.validate(credential, request)

    async def broadcast(self, credential: Credential, request: dict[str, Any]) -> Receipt:
        # Server lifecycle has already run validate. Resolve immediately before
        # the terminal operation so errors prevent settlement.
        options = await resolve_options(self._options, credential, request)
        assert isinstance(self._intent, VerifiableIntent)
        receipt = await self._intent.broadcast(credential, request)
        if self._record_payment is not None:
            await self._record_payment(credential, request, receipt, options)
        return receipt

    async def verify(self, credential: Credential, request: dict[str, Any]) -> Receipt:
        return await broadcast_credential(intent=self, credential=credential, request=request)
