"""What the free tier refuses to run, and what it keeps.

The rule is substitutability, not cost. A provider whose value is a paid SEARCH TOOL --
a corpus, a server-side web tool, a grounding index -- cannot be made free by pointing it
at a cheaper model, because no `:free` slug comes with a search index attached. Those are
vetoed. A provider that is free or public DATA with an LLM stage bolted on can be, so
those stay and their LLM stage moves to the free utility model.

Getting this wrong is expensive in one direction and silently useless in the other, and
neither shows up as an exception.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import BinaryQuestion

from sibyl.config import settings
from sibyl.config.models import FREE_TIER_DISABLED_PROVIDERS, ModelTier, get_roster
from sibyl.config.settings import MODEL_TIER_ENV
from sibyl.research.orchestrator import ResearchOrchestrator

# The five the plan keeps available in the free tier, with the flag that turns each on.
FREE_SAFE_PROVIDER_FLAGS = [
    settings.FINANCIAL_DATA_ENABLED_ENV,
    settings.PREDICTION_MARKETS_ENABLED_ENV,
    settings.RESOLUTION_SOURCE_ENABLED_ENV,
    settings.REFERENCE_CLASS_ENABLED_ENV,
    settings.TS_ANCHOR_ENABLED_ENV,
]
# The six the plan force-disables. AskNews is absent from the flag set because it is a
# rung of the unflagged primary ladder, covered by its own test below.
PAID_SEARCH_PROVIDER_FLAGS = [
    settings.NATIVE_SEARCH_ENABLED_ENV,
    settings.GEMINI_SEARCH_ENABLED_ENV,
    settings.ANTHROPIC_SEARCH_ENABLED_ENV,
    settings.GAP_FILL_ENABLED_ENV,
    settings.GAP_FILL_V2_ENABLED_ENV,
]


def _orchestrator(custom_provider: object | None = None) -> ResearchOrchestrator:
    return ResearchOrchestrator(
        default_llm=MagicMock(),
        summarizer_llm=MagicMock(),
        custom_provider=custom_provider,  # type: ignore[arg-type]
        is_benchmarking=False,
    )


@pytest.fixture
def free_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the suite-wide prod pin for the tests that are about the free tier."""
    monkeypatch.setenv(MODEL_TIER_ENV, ModelTier.FREE.value)


class TestPaidSearchProvidersAreVetoed:
    @pytest.mark.parametrize("flag", PAID_SEARCH_PROVIDER_FLAGS)
    def test_disabled_even_with_the_flag_explicitly_on(
        self, monkeypatch: pytest.MonkeyPatch, free_tier: None, flag: str
    ) -> None:
        monkeypatch.setenv(flag, "true")
        assert _orchestrator()._provider_enabled(flag) is False

    @pytest.mark.parametrize("flag", PAID_SEARCH_PROVIDER_FLAGS)
    def test_the_same_flag_is_honoured_in_prod(self, monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
        monkeypatch.setenv(MODEL_TIER_ENV, ModelTier.PROD.value)
        monkeypatch.setenv(flag, "true")
        assert _orchestrator()._provider_enabled(flag) is True

    @pytest.mark.parametrize("flag", PAID_SEARCH_PROVIDER_FLAGS)
    def test_the_veto_is_logged_so_a_thin_bundle_is_explained(
        self, monkeypatch: pytest.MonkeyPatch, free_tier: None, caplog: pytest.LogCaptureFixture, flag: str
    ) -> None:
        # Without the line, a narrowed provider set reads as a bad question rather than
        # as a deliberate choice.
        monkeypatch.setenv(flag, "true")
        with caplog.at_level("INFO", logger="sibyl.research.orchestrator"):
            _orchestrator()._provider_enabled(flag)
        assert any("FREE TIER" in r.getMessage() and flag in r.getMessage() for r in caplog.records)

    def test_the_veto_list_is_exactly_the_paid_search_providers(self) -> None:
        assert FREE_TIER_DISABLED_PROVIDERS == set(PAID_SEARCH_PROVIDER_FLAGS)


class TestFreeSafeProvidersSurvive:
    @pytest.mark.parametrize("flag", FREE_SAFE_PROVIDER_FLAGS)
    def test_still_enabled_in_the_free_tier_when_flagged_on(
        self, monkeypatch: pytest.MonkeyPatch, free_tier: None, flag: str
    ) -> None:
        monkeypatch.setenv(flag, "true")
        assert _orchestrator()._provider_enabled(flag) is True

    @pytest.mark.parametrize("flag", FREE_SAFE_PROVIDER_FLAGS)
    def test_off_by_default_in_both_tiers(self, monkeypatch: pytest.MonkeyPatch, free_tier: None, flag: str) -> None:
        monkeypatch.delenv(flag, raising=False)
        assert _orchestrator()._provider_enabled(flag) is False

    def test_the_two_lists_do_not_overlap(self) -> None:
        assert not set(FREE_SAFE_PROVIDER_FLAGS) & set(PAID_SEARCH_PROVIDER_FLAGS)


class TestPrimaryLadderIsDisabled:
    """The paid path with no flag on it, and so the one a tier guard most easily misses.

    AskNews -> Exa -> Perplexity runs as the primary provider whether or not any add-on
    flag is set. Every rung is paid. Without this, a free run would bill on the single
    provider nobody opted into.
    """

    def test_free_tier_selects_no_primary_provider(self, free_tier: None) -> None:
        _provider, name = _orchestrator()._select_research_provider()
        assert name == "none"

    @pytest.mark.asyncio
    async def test_the_disabled_primary_returns_empty_rather_than_raising(self, free_tier: None) -> None:
        # An absent primary is a narrower bundle, not a failed run.
        provider, _name = _orchestrator()._select_research_provider()
        assert await provider(MagicMock()) == ""

    def test_prod_tier_still_selects_a_primary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The tier gate must not fire in prod -- with a key present, a rung is chosen.

        The key is set explicitly because the ladder returns "none" when NO provider is
        credentialed, which is indistinguishable from the free tier's veto by name alone.
        Without it this test passed on a developer machine with a populated .env and
        failed in CI, which has no secrets -- the exact ambient-credential dependence the
        suite's autouse flag isolation exists to prevent.
        """
        monkeypatch.setenv(MODEL_TIER_ENV, ModelTier.PROD.value)
        monkeypatch.setenv(settings.OPENROUTER_API_KEY_ENV, "test-key-not-used-for-a-call")
        _provider, name = _orchestrator()._select_research_provider()
        assert name != "none"

    def test_a_custom_provider_still_wins_in_the_free_tier(self, free_tier: None) -> None:
        # The fixture-backtest path passes a custom provider; the veto must not eat it.
        async def _custom(_question: object) -> str:
            return "custom research"

        _provider, name = _orchestrator(custom_provider=_custom)._select_research_provider()
        assert name == "custom"


class TestFreeRosterShape:
    def test_every_free_slug_is_a_free_slug(self) -> None:
        """The property that matters, rather than the specific models, which will rotate."""
        roster = get_roster(ModelTier.FREE)
        slugs = [llm.model for llm in roster.forecasters] + [
            roster.summarizer.model,
            roster.parser.model,
            roster.researcher.model,
            roster.stacker.model,
            roster.stacker_fallback.model,
            roster.analyzer.model,
            roster.market_ranker_config["model"],
            roster.market_query_author_config["model"],
        ]
        assert slugs, "roster exposed no models"
        for slug in slugs:
            assert slug.endswith(":free"), slug

    def test_no_prod_slug_is_a_free_slug(self) -> None:
        roster = get_roster(ModelTier.PROD)
        for llm in roster.forecasters:
            assert not llm.model.endswith(":free"), llm.model

    def test_no_forecaster_shares_the_utility_model(self) -> None:
        """The rate-limit fault from the 2026-09-02 run, as an assertion.

        gemma was a forecaster AND all five utility slots, so one run asked a single
        shared free pool for summarize, parse, stack, analyze and forecast, and was 429'd
        every time. Keeping them separate spreads the load across four pools.
        """
        from sibyl.config.models import FREE_UTILITY_MODEL

        roster = get_roster(ModelTier.FREE)
        forecaster_slugs = {llm.model for llm in roster.forecasters}
        assert FREE_UTILITY_MODEL not in forecaster_slugs, (
            f"{FREE_UTILITY_MODEL} serves the utility slots; using it as a forecaster too "
            "concentrates the whole run on one provider's free quota"
        )

    def test_forecasters_come_from_distinct_vendors(self) -> None:
        """Three members from one vendor correlate, which is an expensive way to ask one
        model three times -- and they share a rate-limit pool, so they fail together."""
        vendors = [llm.model.split("/")[1] for llm in get_roster(ModelTier.FREE).forecasters]
        assert len(set(vendors)) == len(vendors), f"duplicate vendor in the free roster: {vendors}"

    @pytest.mark.parametrize("tier", list(ModelTier))
    def test_no_tier_lists_the_same_forecaster_twice(self, tier: ModelTier) -> None:
        """A duplicate slug is an ensemble of one wearing a costume.

        It would clear the min-forecasters guard, produce three identical rationales, and
        report zero spread -- so conditional stacking would never fire and the run would
        look healthy while being a single model asked three times.
        """
        slugs = [llm.model for llm in get_roster(tier).forecasters]
        assert len(set(slugs)) == len(slugs), f"{tier.value} roster repeats a model: {slugs}"

    def test_enough_free_forecasters_to_clear_the_publish_guard(self) -> None:
        # Below MIN_FORECASTERS_TO_PUBLISH the free lane would fetch questions, run
        # models, and publish nothing -- a green run that did no work.
        assert len(get_roster(ModelTier.FREE).forecasters) >= settings.MIN_FORECASTERS_TO_PUBLISH

    def test_both_tiers_populate_every_slot(self) -> None:
        import dataclasses

        for tier in ModelTier:
            roster = get_roster(tier)
            for field in dataclasses.fields(roster):
                assert getattr(roster, field.name), f"{tier.value}: {field.name} is empty"


class TestFreeTierDropsAreNotAlertable:
    """A dropped forecaster reddens a prod run and not a free one.

    Every OpenRouter `:free` slug shares one upstream pool with the whole free tier, so a
    429 tracks the hour rather than anything this repo did. Treating it as alertable made
    `make e2e_free` permanently red for a reason that says nothing about the code, and a
    signal that is always red stops being read.

    The exemption has to be NARROW, which is what most of these tests check: it covers
    exactly `forecasters_dropped`, in exactly the free tier, and nothing else. Widening it
    by accident would silence real failures in the lane meant to catch them.
    """

    class _Counters:
        """A stand-in exposing the eleven counters alertable_total reads."""

        def __init__(self, **overrides: int) -> None:
            for name in (
                "_forecasters_dropped_count",
                "_questions_failed_to_publish",
                "_stacker_primary_failed_count",
                "_stacker_fallback_used_count",
                "_stacker_fallback_failed_count",
                "_research_provider_failure_count",
                "_summarizer_failure_count",
                "_gap_fill_v2_error_count",
                "_prediction_market_degraded_count",
                "_prediction_market_source_loss_count",
                "_provider_degradation_count",
            ):
                setattr(self, name, 0)
            for name, value in overrides.items():
                assert hasattr(self, name), name
                setattr(self, name, value)

    def _alertable(self, monkeypatch: pytest.MonkeyPatch, tier: ModelTier, **counters: int) -> int:
        from sibyl.telemetry.degradation import alertable_total

        monkeypatch.setenv(MODEL_TIER_ENV, tier.value)
        return alertable_total(self._Counters(**counters))  # type: ignore[arg-type]

    def test_drops_do_not_alert_in_the_free_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._alertable(monkeypatch, ModelTier.FREE, _forecasters_dropped_count=3) == 0

    def test_drops_still_alert_in_prod(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._alertable(monkeypatch, ModelTier.PROD, _forecasters_dropped_count=3) == 3

    @pytest.mark.parametrize(
        "counter",
        [
            "_questions_failed_to_publish",
            "_stacker_primary_failed_count",
            "_stacker_fallback_used_count",
            "_stacker_fallback_failed_count",
            "_research_provider_failure_count",
            "_summarizer_failure_count",
            "_gap_fill_v2_error_count",
            "_prediction_market_degraded_count",
            "_prediction_market_source_loss_count",
            "_provider_degradation_count",
        ],
    )
    def test_every_other_counter_still_alerts_in_the_free_tier(
        self, monkeypatch: pytest.MonkeyPatch, counter: str
    ) -> None:
        """The narrowness check, one counter at a time.

        A failure to PUBLISH is not a free-tier fact of life; nor is a research provider
        erroring. Only the shared-pool 429 is, and only it is exempt.
        """
        assert self._alertable(monkeypatch, ModelTier.FREE, **{counter: 2}) == 2

    def test_drops_do_not_mask_a_real_failure_in_the_same_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The exemption subtracts the drops, not the whole total.
        total = self._alertable(
            monkeypatch, ModelTier.FREE, _forecasters_dropped_count=5, _questions_failed_to_publish=2
        )
        assert total == 2

    def test_drops_are_still_counted_and_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not alertable is not invisible.

        The whole evidence trail stays: the summary line still carries the count, and it
        is a parsed telemetry contract besides.
        """
        from sibyl.telemetry.degradation import format_degradation_summary

        monkeypatch.setenv(MODEL_TIER_ENV, ModelTier.FREE.value)
        summary = format_degradation_summary(self._Counters(_forecasters_dropped_count=3))  # type: ignore[arg-type]
        assert "forecasters_dropped=3" in summary


class TestGapFillAndCruxSearchStayOffOnTheFreeTier:
    """The research run itself must honour the veto, not only ``_provider_enabled``.

    Both gap-fill passes read their flags directly, so a free run given the MiniBench
    workflow's settings paid for GPT-5.6 Terra on the personal key (2026-10-18 free
    rehearsal).
    """

    @staticmethod
    async def _run(monkeypatch: pytest.MonkeyPatch) -> tuple[AsyncMock, AsyncMock]:
        monkeypatch.setenv(settings.GAP_FILL_ENABLED_ENV, "true")
        monkeypatch.setenv(settings.GAP_FILL_V2_ENABLED_ENV, "true")
        orch = _orchestrator()
        provider = AsyncMock(return_value="A" * 2000)
        question = MagicMock(spec=BinaryQuestion)
        question.id_of_question = 1
        question.page_url = "https://example.com/q/1"
        question.question_text = "Will X happen?"
        v1 = AsyncMock(return_value="v1 addendum")
        v2 = AsyncMock(return_value="v2 findings")
        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "custom")]),
            patch("sibyl.research.targeted.run_gap_fill_pass", v1),
            patch("sibyl.research.agentic_gap_fill.run_gap_fill_v2", v2),
        ):
            await orch.run_research(question)
        return v1, v2

    @pytest.mark.asyncio
    async def test_neither_gap_fill_pass_runs_on_the_free_tier(
        self, monkeypatch: pytest.MonkeyPatch, free_tier: None
    ) -> None:
        v1, v2 = await self._run(monkeypatch)
        v1.assert_not_awaited()
        v2.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_both_run_on_the_prod_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MODEL_TIER_ENV, ModelTier.PROD.value)
        v1, v2 = await self._run(monkeypatch)
        v1.assert_awaited_once()
        v2.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gap_fill_v2_refuses_the_free_tier_at_its_own_entry(
        self, monkeypatch: pytest.MonkeyPatch, free_tier: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        from sibyl.research import agentic_gap_fill

        monkeypatch.setenv(settings.GAP_FILL_V2_ENABLED_ENV, "true")
        loop = AsyncMock()
        monkeypatch.setattr(agentic_gap_fill, "run_agentic_loop", loop)
        with caplog.at_level("INFO", logger=agentic_gap_fill.__name__):
            assert (
                await agentic_gap_fill.run_gap_fill_v2(MagicMock(spec=BinaryQuestion), "bundle", is_benchmarking=False)
                == ""
            )
        loop.assert_not_awaited()
        assert "FREE TIER: gap-fill v2 skipped" in caplog.text

    @pytest.mark.asyncio
    async def test_crux_search_is_skipped_on_the_free_tier(
        self, monkeypatch: pytest.MonkeyPatch, free_tier: None
    ) -> None:
        from sibyl.aggregation import routing

        crux = AsyncMock(return_value="what they disagree about")
        monkeypatch.setattr(routing, "extract_disagreement_crux", crux)
        result = await routing._targeted_research_for_crux(MagicMock(), MagicMock(), MagicMock(), [])
        assert result == ""
        crux.assert_not_awaited()
