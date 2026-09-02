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

from unittest.mock import MagicMock

import pytest

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
        monkeypatch.setenv(MODEL_TIER_ENV, ModelTier.PROD.value)
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
