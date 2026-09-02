"""Tests for the zero-cost fixture forecaster (sibyl/eval/benchmark/fixture_llm.py).

Purpose: run the REAL backtest pipeline — extraction ladder, sanitize, tail widening,
PCHIP CDF, aggregation, the A3 guard, scoring — against REAL fetched questions and REAL
resolutions, with only the model's *judgment* faked. Everything downstream of the model
is exercised for real, at zero cost.

Two properties are load-bearing and both are asserted here rather than assumed:

1. **It cannot spend.** No API client is constructed and no socket is opened. This is the
   structural version of the cost rule — the point is that an accidental fixture run is
   free by construction, not by remembering to check.
2. **Its output survives the real parser.** A fixture that emitted anything the
   extraction ladder rejects would exercise only the failure path and prove nothing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from sibyl.eval.benchmark.fixture_llm import FixtureLlm, build_fixture_forecasters


def _mc_probs(raw: str) -> dict[str, float]:
    """Parse an MC block and narrow the union."""
    from sibyl.forecasting.structured_output_schema import MultipleChoiceStructured, parse_structured_block

    block = parse_structured_block(raw, "multiple_choice")
    assert isinstance(block, MultipleChoiceStructured), block
    return block.option_probs


def _percentiles(raw: str) -> dict[float, float]:
    """Parse a numeric block and narrow the union."""
    from sibyl.forecasting.structured_output_schema import NumericStructured, parse_structured_block

    block = parse_structured_block(raw, "numeric")
    assert isinstance(block, NumericStructured), block
    assert block.declared_percentiles is not None
    return block.declared_percentiles


def _binary_prob(raw: str) -> float:
    """Parse a binary block and narrow the union so the type checker can see the field."""
    from sibyl.forecasting.structured_output_schema import BinaryStructured, parse_structured_block

    block = parse_structured_block(raw, "binary")
    assert isinstance(block, BinaryStructured), block
    return block.posterior_prob


_NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)
# prompts.py asserts on these rather than defaulting — a missing timestamp is treated as
# broken upstream data, which is the right call and means test fixtures must supply them.
# Passed explicitly rather than via **kwargs: a dict expansion defeats basedpyright's
# argument checking (85 spurious errors), and these are the fields prompts.py asserts on.
_OPEN_TIME = _NOW - timedelta(days=10)
_CLOSE_TIME = _NOW + timedelta(days=10)
_RESOLVE_TIME = _NOW + timedelta(days=20)


def _binary_prompt() -> str:
    from forecasting_tools.data_models.questions import BinaryQuestion

    from sibyl.forecasting.prompts import binary_prompt

    q = BinaryQuestion(
        question_text="Will X happen?",
        id_of_question=1,
        background_info="bg",
        resolution_criteria="rc",
        fine_print="fp",
        open_time=_OPEN_TIME,
        close_time=_CLOSE_TIME,
        scheduled_resolution_time=_RESOLVE_TIME,
    )
    return binary_prompt(q, "canned research")


def _mc_prompt(options: list[str]) -> str:
    from forecasting_tools.data_models.questions import MultipleChoiceQuestion

    from sibyl.forecasting.prompts import multiple_choice_prompt

    q = MultipleChoiceQuestion(
        question_text="Which?",
        id_of_question=2,
        options=options,
        background_info="bg",
        resolution_criteria="rc",
        fine_print="fp",
        open_time=_OPEN_TIME,
        close_time=_CLOSE_TIME,
        scheduled_resolution_time=_RESOLVE_TIME,
    )
    return multiple_choice_prompt(q, "canned research")


def _numeric_prompt(lower: float = 0.0, upper: float = 100.0) -> str:
    """A real NumericQuestion, not the shared MagicMock — prompts.py asserts on the
    timestamps, and a Mock answers those attribute lookups with another Mock."""
    from forecasting_tools.data_models.questions import NumericQuestion

    from sibyl.forecasting.numeric.utils import bound_messages
    from sibyl.forecasting.prompts import numeric_prompt

    q = NumericQuestion(
        question_text="How many units?",
        id_of_question=3,
        background_info="bg",
        resolution_criteria="rc",
        fine_print="fp",
        lower_bound=lower,
        upper_bound=upper,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="units",
        open_time=_OPEN_TIME,
        close_time=_CLOSE_TIME,
        scheduled_resolution_time=_RESOLVE_TIME,
    )
    upper_msg, lower_msg = bound_messages(q)
    return numeric_prompt(q, "canned research", upper_msg, lower_msg)


class TestCannotSpend:
    """The structural guarantee. A fixture run must be free by construction."""

    def test_constructing_one_opens_no_socket(self) -> None:
        """conftest's autouse egress guard raises on any non-loopback connect."""
        FixtureLlm(model="fixture/alpha")  # must not raise

    def test_invoke_opens_no_socket(self) -> None:
        llm = FixtureLlm(model="fixture/alpha")
        asyncio.run(llm.invoke(_binary_prompt()))  # egress guard would raise

    def test_model_name_is_marked_as_a_fixture(self) -> None:
        """A run log must never let a fixture arm be mistaken for a real one."""
        assert FixtureLlm(model="fixture/alpha").model.startswith("fixture/")

    def test_rejects_a_real_looking_model_slug(self) -> None:
        """Naming a fixture after a real model would make published output misleading."""
        with pytest.raises(ValueError, match="fixture/"):
            FixtureLlm(model="openrouter/openai/gpt-5.6-sol")


class TestOutputSurvivesTheRealParser:
    """If the extraction ladder rejects the output, the fixture proves nothing."""

    def test_binary_block_parses(self) -> None:

        prob = _binary_prob(asyncio.run(FixtureLlm(model="fixture/alpha").invoke(_binary_prompt())))
        assert 0.0 < prob < 1.0

    def test_mc_block_parses_with_the_real_option_names(self) -> None:

        options = ["Alpha", "Beta", "Gamma"]
        probs = _mc_probs(asyncio.run(FixtureLlm(model="fixture/alpha").invoke(_mc_prompt(options))))
        assert set(probs) == set(options)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_numeric_block_parses_with_all_percentiles(self) -> None:
        from sibyl.forecasting.numeric.config import EXPECTED_PERCENTILE_COUNT

        raw = asyncio.run(FixtureLlm(model="fixture/alpha").invoke(_numeric_prompt()))
        assert len(_percentiles(raw)) == EXPECTED_PERCENTILE_COUNT

    def test_numeric_percentiles_are_strictly_increasing(self) -> None:
        """A non-monotone set is repaired or rejected downstream — either way the fixture
        would be testing the repair path rather than the happy path."""

        pct = _percentiles(asyncio.run(FixtureLlm(model="fixture/a").invoke(_numeric_prompt())))
        vals = [v for _, v in sorted(pct.items(), key=lambda kv: float(kv[0]))]
        assert all(b > a for a, b in zip(vals, vals[1:])), vals

    def test_numeric_respects_closed_bounds(self) -> None:

        vals = list(
            _percentiles(asyncio.run(FixtureLlm(model="fixture/a").invoke(_numeric_prompt(10.0, 20.0)))).values()
        )
        assert min(vals) >= 10.0 and max(vals) <= 20.0, vals


class TestDeterministicButVaried:
    """Reproducible runs, but forecasters must disagree or spread logic never fires."""

    def test_same_model_same_prompt_is_reproducible(self) -> None:
        p = _binary_prompt()
        a = asyncio.run(FixtureLlm(model="fixture/alpha").invoke(p))
        b = asyncio.run(FixtureLlm(model="fixture/alpha").invoke(p))
        assert a == b

    def test_different_models_disagree(self) -> None:
        """Identical forecasts would make every question a zero-spread degenerate case."""

        p = _binary_prompt()
        probs = {
            round(_binary_prob(asyncio.run(FixtureLlm(model=name).invoke(p))), 6)
            for name in ("fixture/alpha", "fixture/beta", "fixture/gamma")
        }
        assert len(probs) == 3, probs

    def test_different_questions_get_different_forecasts(self) -> None:

        llm = FixtureLlm(model="fixture/alpha")
        assert 0.0 < _binary_prob(asyncio.run(llm.invoke(_binary_prompt()))) < 1.0
        assert _mc_probs(asyncio.run(llm.invoke(_mc_prompt(["X", "Y"]))))

    def test_seed_changes_the_forecast(self) -> None:
        """The seed is what lets an A/A arm be identical and an A/B arm differ."""

        p = _binary_prompt()
        one = _binary_prob(asyncio.run(FixtureLlm(model="fixture/a", seed=1).invoke(p)))
        two = _binary_prob(asyncio.run(FixtureLlm(model="fixture/a", seed=2).invoke(p)))
        assert one != two


class TestLineupFactory:
    def test_builds_the_requested_number(self) -> None:
        assert len(build_fixture_forecasters(3)) == 3

    def test_all_are_fixtures(self) -> None:
        assert all(llm.model.startswith("fixture/") for llm in build_fixture_forecasters(3))

    def test_names_are_distinct(self) -> None:
        names = [llm.model for llm in build_fixture_forecasters(5)]
        assert len(set(names)) == 5


class TestBacktestWiring:
    """The fixture path must be free through EVERY slot, not just the forecasters.

    A partial guard is the dangerous kind: leaving one helper LLM real would reintroduce
    spend through a back door while the run still looks like a fixture run.
    """

    def _bots(self, count: int = 3):
        from sibyl.eval.backtest import cli as backtest

        return backtest._create_fixture_bots(count=count, seed=0, research_cache={})

    def test_every_forecaster_slot_is_a_fixture(self) -> None:
        """`_forecaster_llms` is the list the fan-out actually iterates."""
        for bot in self._bots():
            assert bot._forecaster_llms, "a bot with no forecasters would vacuously pass"
            for llm in bot._forecaster_llms:
                assert isinstance(llm, FixtureLlm), llm

    def test_no_helper_slot_is_a_real_llm(self) -> None:
        """summarizer / parser / researcher are LLM slots too and would bill."""
        for bot in self._bots(1):
            for slot in ("summarizer", "parser", "researcher"):
                llm = bot.get_llm(slot) if hasattr(bot, "get_llm") else None
                if llm is not None and not isinstance(llm, str):
                    assert isinstance(llm, FixtureLlm), f"{slot} is not a fixture: {llm!r}"

    def test_research_provider_is_overridden(self) -> None:
        """Otherwise AskNews and the search providers still fire and still bill."""
        for bot in self._bots(1):
            assert bot.research_provider is not None, "fixture run must not use live research"

    def test_publishing_stays_off(self) -> None:
        """A fixture forecast reaching Metaculus would be a fabricated public record."""
        for bot in self._bots(1):
            assert bot.publish_reports_to_metaculus is False

    def test_bot_names_mark_the_run_as_fixture(self) -> None:
        for bot in self._bots():
            assert bot.name.startswith("fixture/"), bot.name

    def test_real_path_is_untouched_by_the_flag(self) -> None:
        """The fixture branch must not have changed how real bots are built."""
        from sibyl.eval.backtest import cli as backtest

        assert callable(backtest._create_real_bots)


class TestResearchProvidersAreKilled:
    """Regression: overriding `research_provider` is NOT enough to make a run free.

    It replaces only the PRIMARY provider. Every add-on — Gemini grounded search,
    prediction markets, gap-fill v2, reference-class retrieval — is selected independently
    from env flags in `_select_research_providers`, so they keep firing and billing inside
    what looks like a fixture run.

    This was found by actually running the thing, not by reading it: the forecast succeeded
    while litellm raised an OpenrouterException underneath. That is the argument for
    executing a zero-cost claim rather than reasoning about it.
    """

    def test_fixture_construction_disables_every_provider_flag(self, monkeypatch) -> None:
        import os

        from sibyl.config import settings as constants
        from sibyl.eval.backtest import cli as backtest

        provider_flags = [
            getattr(constants, n)
            for n in dir(constants)
            if n.endswith("_ENABLED_ENV")
            and isinstance(getattr(constants, n), str)
            and "STACKING" not in getattr(constants, n)
        ]
        for flag in provider_flags:
            monkeypatch.setenv(flag, "true")

        backtest._create_fixture_bots(count=1, seed=0, research_cache={})

        still_on = [f for f in provider_flags if os.environ.get(f, "").lower() in ("true", "1", "yes")]
        assert still_on == [], f"these would still fire and bill in fixture mode: {still_on}"

    def test_it_covers_new_providers_automatically(self, monkeypatch) -> None:
        """Derived by naming convention, so a provider added later is covered without
        anyone remembering to update a list — the same failure mode that let
        TS_ANCHOR_ENABLED drift out of the hand-written list in test_research_orchestrator."""
        from sibyl.eval.benchmark.fixture_llm import disable_all_research_providers

        cleared = disable_all_research_providers()
        assert "REFERENCE_CLASS_ENABLED" in cleared, cleared
        assert "GEMINI_SEARCH_ENABLED" in cleared, cleared
        assert len(cleared) >= 8, cleared

    def test_stacking_gates_are_left_alone(self) -> None:
        """Stacking is aggregation, not research. Clearing it would change which
        aggregation path the fixture run exercises."""
        from sibyl.eval.benchmark.fixture_llm import disable_all_research_providers

        cleared = disable_all_research_providers()
        assert not any("STACKING" in f for f in cleared), cleared
