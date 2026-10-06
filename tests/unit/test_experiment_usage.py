"""Cost resolution for grant-funded calls.

These replace a set of tests that encoded the defect found on 2026-10-05: they asserted
that a provider-reported zero was indistinguishable from a missing cost, and that
OpenRouter's own ``usage.cost`` was a mere estimate. Together those made a second
``/generation`` GET a precondition of publishing a forecast that had already been paid
for -- against an endpoint that is only eventually consistent. The cases below pin the
corrected policy, including the shapes that previously failed.

Every test is offline. The HTTP layer is stubbed; no paid call is made.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from sibyl.platform import experiment_usage


def _capture(
    generation_id: str | None = None,
    reported: Decimal | None = None,
    *,
    is_byok: bool | None = None,
    upstream: Decimal | None = None,
) -> experiment_usage.CostCapture:
    capture = experiment_usage.CostCapture(
        generation_id=generation_id,
        reported_cost=reported,
        reported_upstream_cost=upstream,
        reported_is_byok=is_byok,
    )
    capture.ready.set()
    return capture


class _Response:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload, self.status_code = payload, status

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)  # type: ignore[arg-type]


def _stub_http(monkeypatch: pytest.MonkeyPatch, response: _Response) -> None:
    class Client:
        def __init__(self, *a: object, **k: object) -> None: ...
        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *a: object) -> bool:
            return False

        async def get(self, *a: object, **k: object) -> _Response:
            return response

    monkeypatch.setattr(httpx, "AsyncClient", Client)


class TestCallbackCapture:
    @pytest.mark.asyncio
    async def test_captures_only_cost_and_generation_id(self) -> None:
        response = SimpleNamespace(id="gen-x", usage=SimpleNamespace(cost=0.12), choices=["sensitive-output"])
        with experiment_usage.capture_grant_usage() as capture:
            await experiment_usage._GrantUsageObserver().async_log_success_event({}, response, None, None)
        assert (capture.generation_id, capture.reported_cost) == ("gen-x", Decimal("0.12"))
        assert "sensitive-output" not in repr(capture)

    @pytest.mark.asyncio
    async def test_a_zero_provider_cost_is_captured_not_dropped(self) -> None:
        """Zero is a fact. Under BYOK it is the NORMAL OpenRouter charge."""
        response = SimpleNamespace(id="gen-byok", usage=SimpleNamespace(cost=0))
        with experiment_usage.capture_grant_usage() as capture:
            await experiment_usage._GrantUsageObserver().async_log_success_event({}, response, None, None)
        assert capture.reported_cost == Decimal("0")

    @pytest.mark.asyncio
    async def test_captures_the_byok_split_from_usage_accounting(self) -> None:
        usage = {"cost": 0.0095, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.19}}
        response = SimpleNamespace(id="gen-byok", usage=usage)
        with experiment_usage.capture_grant_usage() as capture:
            await experiment_usage._GrantUsageObserver().async_log_success_event({}, response, None, None)
        assert capture.reported_is_byok is True
        assert capture.reported_upstream_cost == Decimal("0.19")


class TestGenerationRecord:
    @pytest.mark.asyncio
    async def test_preserves_the_byok_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real 2026-10-05 record: no OpenRouter credits, real upstream expense."""
        _stub_http(
            monkeypatch,
            _Response({"data": {"total_cost": 0, "upstream_inference_cost": 0.1873175, "is_byok": True}}),
        )
        record = await experiment_usage.generation_record("gen-1", "k")
        assert record is not None
        assert record.openrouter_charge == Decimal("0")
        assert record.upstream_cost == Decimal("0.1873175")
        assert record.is_byok is True
        # Billable follows the money, which under BYOK is upstream -- not OpenRouter's $0.
        assert record.billable == Decimal("0.1873175")

    def test_byok_billable_includes_the_openrouter_fee(self) -> None:
        # The key's limit meters both the upstream expense and OpenRouter's BYOK fee.
        record = experiment_usage.GenerationRecord(Decimal("0.0095"), Decimal("0.19"), True)
        assert record.billable == Decimal("0.1995")

    def test_byok_without_an_upstream_figure_is_not_billable(self) -> None:
        assert experiment_usage.GenerationRecord(Decimal("0.0095"), None, True).billable is None

    @pytest.mark.asyncio
    async def test_non_byok_billable_is_the_openrouter_charge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_http(monkeypatch, _Response({"data": {"total_cost": 0.42, "is_byok": False}}))
        record = await experiment_usage.generation_record("gen-1", "k")
        assert record is not None and record.billable == Decimal("0.42")

    @pytest.mark.asyncio
    async def test_a_404_is_absence_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_http(monkeypatch, _Response({}, status=404))
        assert await experiment_usage.generation_record("gen-1", "k") is None

    @pytest.mark.asyncio
    async def test_a_non_openrouter_id_is_not_looked_up(self) -> None:
        assert await experiment_usage.generation_record("chatcmpl-123", "k") is None


class TestResolution:
    @pytest.mark.asyncio
    async def test_positive_response_cost_confirms_despite_a_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """THE regression. This shape previously raised and discarded the forecast.

        The call succeeded, OpenRouter reported the account charge in the response, and
        the billing record simply was not queryable yet. That is now a confirmed charge.
        """
        _stub_http(monkeypatch, _Response({}, status=404))
        capture = _capture("gen-1", Decimal("0.42"), is_byok=False)
        resolution = await experiment_usage.resolve_grant_cost(capture, 0, "k")
        assert resolution.billable == Decimal("0.42")
        assert resolution.evidence == "provider_response"
        assert resolution.confirmed is True

    @pytest.mark.asyncio
    async def test_a_byok_split_on_the_response_confirms_without_a_lookup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def no_lookup(generation_id: str, api_key: str):
            raise AssertionError("the response already settled the cost")

        monkeypatch.setattr(experiment_usage, "generation_record", no_lookup)
        capture = _capture("gen-1", Decimal("0.0095"), is_byok=True, upstream=Decimal("0.19"))
        resolution = await experiment_usage.resolve_grant_cost(capture, 0, "k")
        assert resolution.billable == Decimal("0.1995")
        assert (resolution.openrouter_charge, resolution.upstream_cost) == (Decimal("0.0095"), Decimal("0.19"))
        assert resolution.confirmed is True

    @pytest.mark.asyncio
    async def test_a_byok_fee_alone_never_confirms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A positive usage.cost under BYOK is only OpenRouter's fee, not the expense.

        Confirming it would discharge the reservation at a fraction of the real spend.
        """
        _stub_http(monkeypatch, _Response({}, status=404))
        for capture in (
            _capture("gen-1", Decimal("0.0095"), is_byok=True),
            _capture("gen-1", Decimal("0.0095")),
        ):
            resolution = await experiment_usage.resolve_grant_cost(capture, 0, "k")
            assert resolution.billable == Decimal("0.0095")
            assert resolution.evidence == "provider_response_partial"
            assert resolution.confirmed is False

    @pytest.mark.asyncio
    async def test_zero_response_cost_is_preserved_but_not_confirmed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A $0 OpenRouter charge that does not say whether it was BYOK must not confirm.

        It is either a genuinely free route or a BYOK route billing upstream, and the
        response alone cannot tell them apart. Confirming it would discharge the
        reservation and report the run as free -- the under-accounting failure.
        """
        _stub_http(monkeypatch, _Response({}, status=404))
        resolution = await experiment_usage.resolve_grant_cost(_capture("gen-1", Decimal("0")), 0, "k")
        assert resolution.billable == Decimal("0")
        assert resolution.confirmed is False
        assert resolution.evidence == "provider_response_partial"

    @pytest.mark.asyncio
    async def test_the_generation_record_outranks_the_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # It is the only source carrying the BYOK split, so it wins when available.
        _stub_http(
            monkeypatch,
            _Response({"data": {"total_cost": 0, "upstream_inference_cost": 0.19, "is_byok": True}}),
        )
        resolution = await experiment_usage.resolve_grant_cost(_capture("gen-1", Decimal("0")), 0, "k")
        assert resolution.billable == Decimal("0.19")
        assert resolution.evidence == "provider_generation"
        assert resolution.confirmed is True
        assert resolution.is_byok is True

    @pytest.mark.asyncio
    async def test_framework_estimate_is_recorded_but_never_confirms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_http(monkeypatch, _Response({}, status=404))
        resolution = await experiment_usage.resolve_grant_cost(_capture("gen-1"), 0.5, "k")
        assert resolution.billable == Decimal("0.5")
        assert resolution.evidence == "framework_estimate"
        assert resolution.confirmed is False

    @pytest.mark.asyncio
    async def test_no_evidence_at_all_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_http(monkeypatch, _Response({}, status=404))
        resolution = await experiment_usage.resolve_grant_cost(_capture("gen-1"), 0, "k")
        assert resolution.billable is None
        assert resolution.evidence == "unknown"
        assert resolution.confirmed is False

    @pytest.mark.asyncio
    async def test_a_delayed_callback_still_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_http(monkeypatch, _Response({"data": {"total_cost": 0.18, "is_byok": False}}))
        with experiment_usage.capture_grant_usage() as capture:

            async def delayed() -> None:
                await asyncio.sleep(0.15)
                await experiment_usage._GrantUsageObserver().async_log_success_event(
                    {}, SimpleNamespace(id="gen-delayed", usage=SimpleNamespace(cost=0.18)), None, None
                )

            task = asyncio.create_task(delayed())
            resolution = await experiment_usage.resolve_grant_cost(capture, 0, "k")
            await task
        assert (resolution.billable, resolution.evidence) == (Decimal("0.18"), "provider_generation")


class TestReconciliation:
    """Confirming estimates afterwards, without re-spending.

    The point of reconciliation is that a run publishes on the evidence it has and the
    authoritative figure arrives later. These are offline: the generation GET is stubbed,
    and sleeps are injected so backoff is exercised without waiting.
    """

    def _ledger(self, tmp_path, *, generation_id: str | None = "gen-1"):
        from decimal import Decimal as D

        from sibyl.platform.experiment_funding import SpendLedger

        ledger = SpendLedger(tmp_path / "spend.sqlite3")
        ledger.start_run("run", D("50"))
        ledger.reserve("a1", "run", "openrouter/openai/t", "forecaster", D("1"))
        ledger.settle(
            "a1",
            D("0"),
            718,
            2014,
            generation_id,
            certainty="estimated",
            evidence_source="provider_response",
            openrouter_charge_usd=D("0"),
        )
        return ledger

    @pytest.mark.asyncio
    async def test_a_byok_estimate_becomes_a_confirmed_upstream_charge(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_http(
            monkeypatch,
            _Response({"data": {"total_cost": 0, "upstream_inference_cost": 0.1873175, "is_byok": True}}),
        )
        ledger = self._ledger(tmp_path)
        outcome = await experiment_usage.reconcile_pending(ledger, "run", "k")
        assert outcome == {"pending": 1, "reconciled": 1, "still_unknown": 0, "no_generation_id": 0}
        assert ledger.unreconciled("run") == []

    @pytest.mark.asyncio
    async def test_it_retries_a_not_yet_visible_record(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The observed production condition: 404, 404, then available."""
        calls = {"n": 0}

        async def flaky(generation_id: str, api_key: str):
            calls["n"] += 1
            if calls["n"] < 3:
                return None
            return experiment_usage.GenerationRecord(Decimal("0"), Decimal("0.19"), True)

        monkeypatch.setattr(experiment_usage, "generation_record", flaky)
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        ledger = self._ledger(tmp_path)
        outcome = await experiment_usage.reconcile_pending(ledger, "run", "k", sleep=fake_sleep)
        assert outcome["reconciled"] == 1
        assert calls["n"] == 3
        assert slept == [2.0, 4.0], "bounded exponential backoff"

    @pytest.mark.asyncio
    async def test_a_permanently_missing_record_keeps_its_reservation(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_http(monkeypatch, _Response({}, status=404))

        async def fake_sleep(seconds: float) -> None: ...

        ledger = self._ledger(tmp_path)
        outcome = await experiment_usage.reconcile_pending(ledger, "run", "k", sleep=fake_sleep)
        assert outcome["still_unknown"] == 1
        # Visible and still held, rather than quietly written off as free.
        assert ledger.unreconciled("run") == [("a1", "gen-1")]

    @pytest.mark.asyncio
    async def test_an_attempt_with_no_generation_id_is_reported_not_guessed(self, tmp_path) -> None:
        ledger = self._ledger(tmp_path, generation_id=None)
        outcome = await experiment_usage.reconcile_pending(ledger, "run", "k")
        assert outcome["no_generation_id"] == 1
        assert outcome["reconciled"] == 0

    @pytest.mark.asyncio
    async def test_it_never_issues_a_completion_request(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reconciliation must never re-spend to learn what a call cost."""
        import litellm

        def explode(*a: object, **k: object):
            raise AssertionError("reconciliation must not call a model")

        monkeypatch.setattr(litellm, "acompletion", explode, raising=False)
        monkeypatch.setattr(litellm, "completion", explode, raising=False)
        _stub_http(monkeypatch, _Response({"data": {"total_cost": 0.42, "is_byok": False}}))
        ledger = self._ledger(tmp_path)
        assert (await experiment_usage.reconcile_pending(ledger, "run", "k"))["reconciled"] == 1
