"""Which key pays for a call, and what the spend ledger records about it.

The model call itself is faked at the seam's lower edge, so these tests see exactly
which credential each request carried and in what order. No network is used.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from typing import Any

import litellm
import pytest
from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse

from sibyl.platform import experiment_usage, funding
from sibyl.platform.experiment_funding import FundingSource, SpendLedger
from sibyl.platform.key_fallback import FallbackOpenRouterLlm

MODEL = "openrouter/openai/test"


def _response(model: str = MODEL) -> TextTokenCostResponse:
    return TextTokenCostResponse(
        data="answer", prompt_tokens_used=10, completion_tokens_used=5, total_tokens_used=15, model=model, cost=0.2
    )


def _credit_error() -> Exception:
    return litellm.exceptions.APIError(
        status_code=402, message="insufficient credits", llm_provider="openrouter", model=MODEL
    )


def _rate_limit() -> Exception:
    return litellm.exceptions.RateLimitError(message="rate limited", llm_provider="openrouter", model=MODEL)


def _server_error() -> Exception:
    return litellm.exceptions.APIError(
        status_code=500, message="upstream broke", llm_provider="openrouter", model=MODEL
    )


class _FakeProvider:
    """Stands in for forecasting-tools' request; fails for the keys it is told to."""

    def __init__(self) -> None:
        self.keys: list[str | None] = []
        self.failures: dict[str | None, Exception] = {}

    async def __call__(self, llm: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
        key = llm.litellm_kwargs.get("api_key")
        self.keys.append(key)
        if key in self.failures:
            raise self.failures[key]
        return _response(llm.model)


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> _FakeProvider:
    fake = _FakeProvider()
    monkeypatch.setattr(funding, "_ORIGINAL_DIRECT_CALL", fake)
    monkeypatch.setattr(experiment_usage, "CALLBACK_WAIT_S", 0)
    return fake


def _funding(tmp_path: Path, *, grant: bool = True, budget: str = "10", reserve: str = "1") -> funding.Funding:
    ledger = SpendLedger(tmp_path / "spend.sqlite3")
    ledger.start_run("run", Decimal(budget) if grant else Decimal(0))
    configured = funding.Funding("run", ledger, "grant-key" if grant else None, Decimal(reserve))
    funding.configure_funding(configured)
    return configured


def _sources(configured: funding.Funding) -> list[tuple[str, str]]:
    with configured.ledger._connect() as db:
        return list(db.execute("SELECT source, status FROM attempts ORDER BY rowid"))


@pytest.mark.asyncio
async def test_without_funding_the_call_is_untouched(provider: _FakeProvider) -> None:
    llm = GeneralLlm(model=MODEL, api_key="own-key")
    await funding._funded_direct_call(llm, "prompt")
    assert provider.keys == ["own-key"]


@pytest.mark.asyncio
async def test_the_grant_pays_first_and_the_shared_client_keeps_its_key(
    tmp_path: Path, provider: _FakeProvider
) -> None:
    configured = _funding(tmp_path)
    llm = funding.label_stage(GeneralLlm(model=MODEL, api_key="own-key"), "forecaster")
    await funding._funded_direct_call(llm, "prompt")
    assert provider.keys == ["grant-key"]
    assert llm.litellm_kwargs["api_key"] == "own-key", "a shared roster client must never be rekeyed in place"
    with configured.ledger._connect() as db:
        assert db.execute("SELECT source, stage, status FROM attempts").fetchall() == [
            ("experiment-grant", "forecaster", "succeeded")
        ]


@pytest.mark.asyncio
async def test_an_out_of_credit_grant_moves_the_call_and_the_rest_of_the_run(
    tmp_path: Path, provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
    configured = _funding(tmp_path)
    provider.failures["grant-key"] = _credit_error()
    llm = GeneralLlm(model=MODEL, api_key="own-key")
    await funding._funded_direct_call(llm, "prompt")
    await funding._funded_direct_call(llm, "prompt")
    assert provider.keys == ["grant-key", "own-key", "own-key"]
    assert configured.grant_stopped is not None
    assert _sources(configured) == [
        ("experiment-grant", "rejected"),
        ("personal-key", "succeeded"),
        ("personal-key", "succeeded"),
    ]
    assert configured.ledger.summary("run")["remaining_local_usd"] == "95", "a refusal holds no grant money"


@pytest.mark.asyncio
async def test_a_rate_limited_grant_moves_only_that_call(tmp_path: Path, provider: _FakeProvider) -> None:
    configured = _funding(tmp_path)
    provider.failures["grant-key"] = _rate_limit()
    await funding._funded_direct_call(GeneralLlm(model=MODEL, api_key="own-key"), "prompt")
    assert provider.keys == ["grant-key", "own-key"]
    assert configured.grant_stopped is None


@pytest.mark.asyncio
async def test_a_grant_failure_another_key_cannot_fix_is_raised_as_usual(
    tmp_path: Path, provider: _FakeProvider
) -> None:
    configured = _funding(tmp_path)
    provider.failures["grant-key"] = _server_error()
    with pytest.raises(litellm.exceptions.APIError):
        await funding._funded_direct_call(GeneralLlm(model=MODEL, api_key="own-key"), "prompt")
    assert provider.keys == ["grant-key"], "the payer must not add a retry the client would not make"
    assert _sources(configured) == [("experiment-grant", "uncertain")]


@pytest.mark.asyncio
async def test_an_exhausted_grant_budget_sends_calls_to_the_ordinary_key(
    tmp_path: Path, provider: _FakeProvider
) -> None:
    configured = _funding(tmp_path, budget="1", reserve="1")
    llm = GeneralLlm(model=MODEL, api_key="own-key")
    await funding._funded_direct_call(llm, "prompt")  # the estimate holds the whole budget
    await funding._funded_direct_call(llm, "prompt")
    assert provider.keys == ["grant-key", "own-key"]
    assert configured.grant_stopped is not None


@pytest.mark.asyncio
async def test_the_metaculus_key_is_labelled_as_such(
    tmp_path: Path, provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "metaculus-key")
    configured = _funding(tmp_path, grant=False)
    await funding._funded_direct_call(GeneralLlm(model=MODEL, api_key="metaculus-key"), "prompt")
    assert _sources(configured) == [("metaculus-key", "succeeded")]


@pytest.mark.asyncio
async def test_with_the_grant_the_primary_wrapper_never_reaches_the_personal_key(
    tmp_path: Path, provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", funding._funded_direct_call)
    _funding(tmp_path)
    llm = FallbackOpenRouterLlm(model=MODEL, primary_api_key="metaculus-key", secondary_api_key="personal-key")
    assert await llm.invoke("prompt") == "answer"
    assert provider.keys == ["grant-key"]


@pytest.mark.asyncio
async def test_a_service_outside_openrouter_is_recorded_but_never_paid_by_the_grant(
    tmp_path: Path, provider: _FakeProvider
) -> None:
    configured = _funding(tmp_path)
    await funding._funded_direct_call(GeneralLlm(model="perplexity/sonar", api_key="perplexity-key"), "prompt")
    assert provider.keys == ["perplexity-key"]
    assert _sources(configured) == [("direct-service", "succeeded")]


@pytest.mark.asyncio
async def test_an_unwritable_ledger_never_stops_an_ordinary_call(
    tmp_path: Path, provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _funding(tmp_path, grant=False)

    def broken(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(configured.ledger, "reserve", broken)
    await funding._funded_direct_call(GeneralLlm(model=MODEL, api_key="own-key"), "prompt")
    assert provider.keys == ["own-key"]


@pytest.mark.asyncio
async def test_an_unwritable_ledger_keeps_the_grant_from_paying(
    tmp_path: Path, provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _funding(tmp_path)

    def broken(*args: Any, source: FundingSource = FundingSource.EXPERIMENT_GRANT, **kwargs: Any) -> None:
        if source is FundingSource.EXPERIMENT_GRANT:
            raise OSError("disk full")

    monkeypatch.setattr(configured.ledger, "reserve", broken)
    await funding._funded_direct_call(GeneralLlm(model=MODEL, api_key="own-key"), "prompt")
    assert provider.keys == ["own-key"]


@pytest.mark.asyncio
async def test_direct_service_calls_are_recorded_against_their_question(
    tmp_path: Path, provider: _FakeProvider
) -> None:
    configured = _funding(tmp_path, grant=False)

    async def search() -> str:
        return "results"

    with funding.spend_question(42):
        assert await funding.record_direct_call("exa/search", "gap_fill_v2", search) == "results"
    with configured.ledger._connect() as db:
        assert db.execute("SELECT source, model, question_id FROM attempts").fetchall() == [
            ("direct-service", "exa/search", "42")
        ]


class TestOpeningARun:
    def test_an_ordinary_run_opens_a_ledger_without_a_grant(self) -> None:
        opened = funding.open_run_funding("run")
        assert opened is not None and opened.grant_key is None

    def test_a_grant_budget_needs_the_grant_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(funding.GRANT_KEY_ENV, raising=False)
        with pytest.raises(ValueError, match="SIBYL_EXPERIMENT_OPENROUTER_API_KEY"):
            funding.open_run_funding("run", grant_budget_usd=Decimal("5"))

    def test_the_grant_key_must_differ_from_the_ordinary_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(funding.GRANT_KEY_ENV, "same")
        monkeypatch.setenv("OPENROUTER_API_KEY", "same")
        with pytest.raises(ValueError, match="distinct"):
            funding.open_run_funding("run", grant_budget_usd=Decimal("5"))

    def test_a_grant_run_checks_the_key_cap_before_starting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(funding.GRANT_KEY_ENV, "grant")
        checked: list[Decimal] = []
        monkeypatch.setattr(funding, "check_grant_key_cap", lambda key, limit: checked.append(limit) or limit)
        opened = funding.open_run_funding("run", grant_budget_usd=Decimal("5"))
        assert opened is not None and opened.grant_key == "grant"
        assert checked == [Decimal("5")]

    def test_a_per_call_reservation_above_the_budget_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(funding.GRANT_KEY_ENV, "grant")
        with pytest.raises(ValueError, match="per-call"):
            funding.open_run_funding("run", grant_budget_usd=Decimal("1"), grant_reserve_per_call_usd=Decimal("2"))


@pytest.mark.asyncio
async def test_a_call_outside_openrouter_never_waits_for_a_billing_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct SDKs never fire LiteLLM's callback, so waiting for it only adds delay."""
    import time

    _funding(tmp_path, grant=False)
    monkeypatch.setattr(experiment_usage, "CALLBACK_WAIT_S", 30)

    async def search() -> str:
        return "results"

    started = time.monotonic()
    await funding.record_direct_call("exa/search", "gap_fill_v2", search)
    assert time.monotonic() - started < 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 403, 404])
async def test_a_request_turned_away_before_the_model_runs_releases_its_hold(
    tmp_path: Path, provider: _FakeProvider, status: int
) -> None:
    configured = _funding(tmp_path)
    provider.failures["grant-key"] = litellm.exceptions.APIError(
        status_code=status, message="rejected", llm_provider="openrouter", model=MODEL
    )
    with pytest.raises(litellm.exceptions.APIError):
        await funding._funded_direct_call(GeneralLlm(model=MODEL, api_key="own-key"), "prompt")
    assert _sources(configured) == [("experiment-grant", "rejected")]
    assert configured.ledger.summary("run")["remaining_local_usd"] == "95"


@pytest.mark.asyncio
async def test_a_failure_that_may_have_been_billed_holds_until_written_off(
    tmp_path: Path, provider: _FakeProvider
) -> None:
    configured = _funding(tmp_path)
    provider.failures["grant-key"] = _server_error()
    with pytest.raises(litellm.exceptions.APIError):
        await funding._funded_direct_call(GeneralLlm(model=MODEL, api_key="own-key"), "prompt")
    assert configured.ledger.summary("run")["remaining_local_usd"] == "94"
    assert configured.ledger.write_off("trueup", Decimal("0.05"), "OpenRouter activity", expected_count=1) == 1
    # The $1 hold is replaced by what OpenRouter actually metered.
    assert configured.ledger.summary()["remaining_local_usd"] == "94.95"


@pytest.mark.asyncio
async def test_holds_of_requests_in_flight_do_not_stop_the_grant(tmp_path: Path, provider: _FakeProvider) -> None:
    """Concurrent calls fill a small budget with holds; capacity returns as they settle."""
    import asyncio

    configured = _funding(tmp_path, budget="2", reserve="1")
    release = asyncio.Event()
    started = 0

    async def slow(llm: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
        nonlocal started
        provider.keys.append(llm.litellm_kwargs.get("api_key"))
        started += 1
        if llm.litellm_kwargs.get("api_key") == "grant-key":
            await release.wait()
        return _response(llm.model)

    funding._ORIGINAL_DIRECT_CALL = slow  # restored by the provider fixture
    calls = [
        asyncio.create_task(funding._funded_direct_call(GeneralLlm(model=MODEL, api_key="own-key"), "p"))
        for _ in range(3)
    ]
    while started < 3:
        await asyncio.sleep(0)
    release.set()
    await asyncio.gather(*calls)
    assert sorted(str(key) for key in provider.keys) == ["grant-key", "grant-key", "own-key"]
    assert configured.grant_stopped is None


@pytest.mark.asyncio
async def test_the_fallback_wrapper_offers_the_grant_once(
    tmp_path: Path, provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grant rate-limited, Metaculus key out of credit: the personal key pays, the grant is not asked twice."""
    monkeypatch.setattr(GeneralLlm, "_mockable_direct_call_to_model", funding._funded_direct_call)
    monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "metaculus-key")
    _funding(tmp_path)
    provider.failures["grant-key"] = _rate_limit()
    provider.failures["metaculus-key"] = _credit_error()
    llm = FallbackOpenRouterLlm(
        model=MODEL, primary_api_key="metaculus-key", secondary_api_key="personal-key", allowed_tries=1
    )
    assert await llm.invoke("prompt") == "answer"
    assert provider.keys == ["grant-key", "metaculus-key", "personal-key"]


class TestStartingARunSafely:
    @pytest.mark.parametrize("second_budget", [None, Decimal("5")], ids=["no-grant", "same-budget"])
    def test_a_reused_run_id_is_refused_whatever_the_budget(
        self, monkeypatch: pytest.MonkeyPatch, second_budget: Decimal | None
    ) -> None:
        monkeypatch.setenv(funding.GRANT_KEY_ENV, "grant")
        monkeypatch.setattr(funding, "check_grant_key_cap", lambda key, limit: limit)
        funding.open_run_funding("same", grant_budget_usd=Decimal("5"))
        with pytest.raises(ValueError, match="already in the spend ledger"):
            funding.open_run_funding("same", grant_budget_usd=second_budget)

    def test_whitespace_does_not_make_the_personal_key_look_distinct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(funding.GRANT_KEY_ENV, "sk-same")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-same\n")
        with pytest.raises(ValueError, match="distinct"):
            funding.open_run_funding("run", grant_budget_usd=Decimal("5"))


class TestGrantKeyCap:
    """The cap OpenRouter enforces on the grant key is the only hard bound on its spend."""

    @staticmethod
    def _key(monkeypatch: pytest.MonkeyPatch, **data: Any) -> None:
        import httpx

        body = {"limit": 100, "limit_remaining": 50, "include_byok_in_limit": True, "limit_reset": None, **data}

        def fake_get(url: str, **kwargs: Any) -> httpx.Response:
            assert url == "https://openrouter.ai/api/v1/key"
            return httpx.Response(200, json={"data": body}, request=httpx.Request("GET", url))

        monkeypatch.setattr(funding.httpx, "get", fake_get)

    def test_a_bounded_key_with_room_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._key(monkeypatch)
        assert funding.check_grant_key_cap("k", Decimal("45")) == Decimal("50")

    @pytest.mark.parametrize(
        "data",
        [
            {"limit": None},
            {"limit": 100.01},
            {"include_byok_in_limit": False},
            {"include_byok_in_limit": "true"},
            {"limit_reset": "daily"},
        ],
        ids=["uncapped", "above-allocation", "byok-excluded", "byok-not-boolean", "resetting"],
    )
    def test_a_key_that_does_not_bound_the_grant_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, data: dict[str, Any]
    ) -> None:
        self._key(monkeypatch, **data)
        with pytest.raises(ValueError):
            funding.check_grant_key_cap("k", Decimal("1"))

    def test_the_budget_plus_reserve_must_fit_what_is_left(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._key(monkeypatch, limit_remaining=50)
        funding.check_grant_key_cap("k", Decimal("45"))  # 45 + 5 = 50 exactly
        with pytest.raises(ValueError, match="cannot cover"):
            funding.check_grant_key_cap("k", Decimal("45.01"))


@pytest.mark.asyncio
async def test_a_charge_above_the_per_call_hold_stops_the_grant(
    tmp_path: Path, provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = _funding(tmp_path, reserve="0.1")  # the fake response reports 0.2
    llm = GeneralLlm(model=MODEL, api_key="own-key")
    await funding._funded_direct_call(llm, "prompt")
    assert configured.grant_stopped is not None and "exceeded" in configured.grant_stopped
    await funding._funded_direct_call(llm, "prompt")
    assert provider.keys == ["grant-key", "own-key"]


@pytest.mark.asyncio
async def test_a_direct_service_failure_propagates_unchanged_and_is_not_called_refused(tmp_path: Path) -> None:
    """The research call sites catch these exact exceptions, so the wrapper must not alter them."""
    import asyncio

    configured = _funding(tmp_path, grant=False)

    async def timed_out() -> str:
        raise asyncio.TimeoutError

    async def rate_limited() -> str:
        raise Exception("AskNews: Too Many Requests")

    with pytest.raises(asyncio.TimeoutError):
        await funding.record_direct_call("exa/search", "gap_fill_v2", timed_out)
    with pytest.raises(Exception, match="Too Many Requests"):
        await funding.record_direct_call("asknews/search_news", "asknews", rate_limited)
    assert _sources(configured) == [("direct-service", "uncertain"), ("direct-service", "uncertain")]


@pytest.mark.asyncio
async def test_a_billed_answer_that_then_fails_keeps_its_generation_id(
    tmp_path: Path, provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenRouter answered and billed; the framework then rejected the empty answer."""
    configured = _funding(tmp_path)

    async def billed_then_empty(llm: GeneralLlm, prompt: Any) -> TextTokenCostResponse:
        capture = experiment_usage._ACTIVE_CAPTURE.get()
        assert capture is not None
        capture.generation_id = "gen-billed"
        raise RuntimeError("LLM answer is an empty string")

    monkeypatch.setattr(funding, "_ORIGINAL_DIRECT_CALL", billed_then_empty)
    with pytest.raises(RuntimeError):
        await funding._funded_direct_call(GeneralLlm(model=MODEL, api_key="own-key"), "prompt")
    with configured.ledger._connect() as db:
        assert db.execute("SELECT status, provider_request_id FROM attempts").fetchone() == ("uncertain", "gen-billed")


@pytest.mark.asyncio
async def test_a_revoked_grant_key_stops_the_grant(tmp_path: Path, provider: _FakeProvider) -> None:
    configured = _funding(tmp_path)
    provider.failures["grant-key"] = litellm.exceptions.AuthenticationError(
        message="invalid key", llm_provider="openrouter", model=MODEL
    )
    llm = GeneralLlm(model=MODEL, api_key="own-key")
    await funding._funded_direct_call(llm, "prompt")
    await funding._funded_direct_call(llm, "prompt")
    assert provider.keys == ["grant-key", "own-key", "own-key"]
    assert configured.grant_stopped is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [413, 422, 429])
async def test_these_refusals_also_release_their_hold(tmp_path: Path, provider: _FakeProvider, status: int) -> None:
    configured = _funding(tmp_path)
    provider.failures["grant-key"] = litellm.exceptions.APIError(
        status_code=status, message="refused", llm_provider="openrouter", model=MODEL
    )
    try:
        await funding._funded_direct_call(GeneralLlm(model=MODEL, api_key="own-key"), "prompt")
    except litellm.exceptions.APIError:
        pass  # 413/422 are raised; 429 moves the call to the ordinary key
    assert _sources(configured)[0] == ("experiment-grant", "rejected")
    assert configured.ledger.summary("run")["remaining_local_usd"] == "95"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        asyncio.TimeoutError(),
        asyncio.CancelledError(),
        ConnectionError("reset"),
        litellm.exceptions.APIError(status_code=408, message="timeout", llm_provider="openrouter", model=MODEL),
        litellm.exceptions.APIError(status_code=502, message="bad gateway", llm_provider="openrouter", model=MODEL),
    ],
    ids=["timeout", "cancelled", "connection", "408", "502"],
)
async def test_a_failure_that_may_have_been_billed_keeps_its_hold(
    tmp_path: Path, provider: _FakeProvider, failure: BaseException
) -> None:
    configured = _funding(tmp_path)
    provider.failures["grant-key"] = failure  # type: ignore[assignment]
    with pytest.raises(type(failure)):
        await funding._funded_direct_call(GeneralLlm(model=MODEL, api_key="own-key"), "prompt")
    assert _sources(configured) == [("experiment-grant", "uncertain")]
    assert configured.ledger.summary("run")["remaining_local_usd"] == "94"


@pytest.mark.asyncio
async def test_a_charge_equal_to_the_hold_does_not_stop_the_grant(tmp_path: Path, provider: _FakeProvider) -> None:
    configured = _funding(tmp_path, reserve="0.2")  # the fake response reports exactly 0.2
    await funding._funded_direct_call(GeneralLlm(model=MODEL, api_key="own-key"), "prompt")
    assert configured.grant_stopped is None


class TestStartupFailures:
    def test_an_unreadable_cap_lookup_refuses_the_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        def unreachable(url: str, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectError("down")

        monkeypatch.setattr(funding.httpx, "get", unreachable)
        with pytest.raises(ValueError, match="cap check failed"):
            funding.check_grant_key_cap("k", Decimal("5"))

    def test_a_cap_lookup_without_limit_remaining_refuses_the_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        monkeypatch.setattr(
            funding.httpx,
            "get",
            lambda url, **kw: httpx.Response(200, json={"data": {"limit": 50}}, request=httpx.Request("GET", url)),
        )
        with pytest.raises(ValueError, match="cap check failed"):
            funding.check_grant_key_cap("k", Decimal("5"))

    @pytest.mark.parametrize("budget", ["0", "-1", "95.01", "NaN"])
    def test_an_impossible_grant_budget_is_refused(self, budget: str) -> None:
        with pytest.raises(ValueError):
            funding.open_run_funding("run", grant_budget_usd=Decimal(budget))

    def test_an_unwritable_ledger_leaves_an_ordinary_run_unaccounted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(funding.SPEND_LEDGER_ENV, str(tmp_path))  # a directory, not a file
        assert funding.open_run_funding("run") is None

    def test_an_unwritable_ledger_refuses_a_grant_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(funding.SPEND_LEDGER_ENV, str(tmp_path))
        monkeypatch.setenv(funding.GRANT_KEY_ENV, "grant")
        monkeypatch.setattr(funding, "check_grant_key_cap", lambda key, limit: limit)
        with pytest.raises(ValueError, match="not writable"):
            funding.open_run_funding("run", grant_budget_usd=Decimal("5"))

    def test_a_spend_summary_that_cannot_be_read_never_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import decimal

        configured = _funding(tmp_path)

        def corrupt(*args: Any, **kwargs: Any) -> None:
            raise decimal.InvalidOperation("bad stored amount")

        monkeypatch.setattr(configured.ledger, "sources", corrupt)
        funding.log_spend_summary(configured)
