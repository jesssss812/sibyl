"""Tier selection, the refusal matrix, and the mode aliases.

The refusal is the load-bearing part. A free forecast is not a cheap version of a prod
forecast, it is a much worse one, and publishing it to a SCORED tournament damages a
public record that cannot be retracted. So the guard has to fire before anything is
built, and it has to fire on every scored mode rather than on a list someone maintains.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sibyl import cli as cli_module
from sibyl.cli import _MODE_ALIASES, _SCORED_MODES, _default_max_questions, _refuse_free_tier_on_scored_tournament
from sibyl.config.models import ModelTier
from sibyl.config.settings import DEFAULT_MAX_QUESTIONS_PER_RUN, FREE_TIER_MAX_QUESTIONS, MODEL_TIER_ENV


class TestRefusalMatrix:
    @pytest.mark.parametrize("mode", sorted(_SCORED_MODES))
    def test_free_tier_refuses_every_scored_mode(self, mode: str) -> None:
        with pytest.raises(SystemExit) as exc:
            _refuse_free_tier_on_scored_tournament(ModelTier.FREE, mode)
        assert exc.value.code != 0

    def test_free_tier_allows_the_sandbox_mode(self) -> None:
        # Nothing in bot-testing-area is scored, which is the whole reason the free lane
        # publishes there.
        _refuse_free_tier_on_scored_tournament(ModelTier.FREE, "test")

    @pytest.mark.parametrize("mode", sorted(set(_MODE_ALIASES.values())))
    def test_prod_tier_refuses_nothing(self, mode: str) -> None:
        _refuse_free_tier_on_scored_tournament(ModelTier.PROD, mode)

    def test_every_canonical_mode_is_either_scored_or_the_sandbox(self) -> None:
        """No mode may be silently neither, which is how a new one would slip the guard."""
        canonical = set(_MODE_ALIASES.values())
        assert canonical - _SCORED_MODES == {"test"}

    def test_the_refusal_names_the_way_out(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("ERROR", logger="sibyl.cli"):
            with pytest.raises(SystemExit):
                _refuse_free_tier_on_scored_tournament(ModelTier.FREE, "tournament")
        message = " ".join(r.getMessage() for r in caplog.records)
        assert "--tier prod" in message and MODEL_TIER_ENV in message


class TestModeAliases:
    @pytest.mark.parametrize(
        "spelling, canonical",
        [
            ("tournament", "tournament"),
            ("minibench", "minibench"),
            ("cup", "cup"),
            ("metaculus_cup", "cup"),
            ("quarterly_cup", "cup"),
            ("test", "test"),
            ("test_questions", "test"),
        ],
    )
    def test_old_spellings_still_resolve(self, spelling: str, canonical: str) -> None:
        # The old names are in the workflow yamls, in docs/runbooks/operations.md and in the
        # operator's shell history. Breaking a dispatch someone types from memory buys
        # nothing.
        assert _MODE_ALIASES[spelling] == canonical


class TestMaxQuestionsDefault:
    def test_free_tier_gets_the_small_cap(self) -> None:
        # OpenRouter allows ~50 :free requests a day without a balance and one question
        # costs several, so the cap is what stops a test run eating the day's quota.
        assert _default_max_questions(ModelTier.FREE) == FREE_TIER_MAX_QUESTIONS
        assert FREE_TIER_MAX_QUESTIONS < DEFAULT_MAX_QUESTIONS_PER_RUN

    def test_prod_tier_keeps_the_cap_it_always_had(self) -> None:
        assert _default_max_questions(ModelTier.PROD) == DEFAULT_MAX_QUESTIONS_PER_RUN


class TestTierResolutionThroughArgv:
    """`--tier` overrides the env var; the guard runs before a client is built."""

    def _run(self, argv: list[str], env_tier: str | None, monkeypatch: pytest.MonkeyPatch) -> int | None:
        if env_tier is None:
            monkeypatch.delenv(MODEL_TIER_ENV, raising=False)
        else:
            monkeypatch.setenv(MODEL_TIER_ENV, env_tier)
        monkeypatch.setattr(sys, "argv", ["cli", *argv])
        # If the refusal works, NONE of these are reached. Patching them proves it: a
        # guard that fires after the bot is built has already done the expensive thing.
        with (
            patch("sibyl.cli.SibylBot", MagicMock()) as bot_class,
            patch("sibyl.cli.MetaculusApi", MagicMock()),
            patch("sibyl.cli.apply_publish_hardening"),
            patch("sibyl.cli.apply_fetch_hardening"),
            patch("sibyl.cli.verify_metaculus_api_identity"),
            patch("sibyl.cli.CreditTelemetry", MagicMock()),
        ):
            try:
                cli_module.main()
            except SystemExit as exc:
                assert not bot_class.called, "refused only AFTER building the bot"
                return int(exc.code) if isinstance(exc.code, int) else 1
        return None

    def test_env_prod_still_cannot_run_dormant_tournament(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The configured tournament expired; prod tier must not bypass dormancy.
        assert self._run(["tournament"], "prod", monkeypatch) == 2

    def test_flag_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # env says prod, flag says free -> the flag wins, so a scored mode is refused.
        assert self._run(["tournament", "--tier", "free"], "prod", monkeypatch) == 2

    def test_env_free_refuses_a_scored_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._run(["minibench"], "free", monkeypatch) == 2

    def test_unset_env_defaults_to_free_and_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._run(["cup"], None, monkeypatch) == 2

    def test_positional_and_flag_forms_agree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._run(["--mode", "metaculus_cup"], None, monkeypatch) == 2

    def test_contradictory_mode_spellings_are_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # argparse exits 2 on parser.error; the point is that it does not silently pick one.
        assert self._run(["tournament", "--mode", "minibench"], "prod", monkeypatch) == 2

    def test_a_grant_budget_is_refused_on_the_free_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A free run must not be able to spend, whichever key would pay.
        assert self._run(["test", "--grant-budget-usd", "5"], "free", monkeypatch) == 2

    def test_a_grant_budget_without_the_grant_key_is_refused_before_the_bot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SIBYL_EXPERIMENT_OPENROUTER_API_KEY", raising=False)
        assert self._run(["test", "--tier", "prod", "--grant-budget-usd", "5"], None, monkeypatch) == 2

    def test_a_refused_grant_leaves_no_archive_behind(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("SIBYL_EXPERIMENT_OPENROUTER_API_KEY", raising=False)
        argv = ["test", "--tier", "prod", "--grant-budget-usd", "5", "--run-id", "r1", "--archive-dir", str(tmp_path)]
        assert self._run(argv, None, monkeypatch) == 2
        assert not (tmp_path / "r1").exists(), "a rerun with the same run ID must not be blocked"

    def test_the_tier_flag_governs_research_gating_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`--tier prod` with SIBYL_MODEL_TIER unset must not gate research as free."""
        from sibyl.config.models import ModelTier, active_tier

        seen: list[ModelTier] = []

        def record_then_stop(tier: ModelTier) -> None:
            seen.append(active_tier())
            raise SystemExit(0)  # the tier is settled by now; nothing after it matters here

        monkeypatch.setattr(cli_module, "_log_tier_banner", record_then_stop)
        self._run(["test", "--tier", "prod"], None, monkeypatch)
        assert seen == [ModelTier.PROD]


class TestOneQuestionAtATime:
    """`--one-at-a-time` reaches the bot; without it the batch runs as before."""

    def _bot_kwargs(self, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        seen: list[dict[str, object]] = []

        def record_then_stop(*args: object, **kwargs: object) -> None:
            seen.append(kwargs)
            raise SystemExit(0)

        monkeypatch.delenv(MODEL_TIER_ENV, raising=False)
        monkeypatch.setattr(sys, "argv", ["cli", *argv])
        with (
            patch("sibyl.cli.SibylBot", side_effect=record_then_stop),
            patch("sibyl.cli.MetaculusApi", MagicMock()),
            patch("sibyl.cli.apply_publish_hardening"),
            patch("sibyl.cli.apply_fetch_hardening"),
            patch("sibyl.cli.verify_metaculus_api_identity"),
            patch("sibyl.cli.CreditTelemetry", MagicMock()),
            pytest.raises(SystemExit),
        ):
            cli_module.main()
        return seen[0]

    def test_the_flag_reaches_the_bot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._bot_kwargs(["test", "--one-at-a-time"], monkeypatch)["one_question_at_a_time"] is True

    def test_off_without_the_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._bot_kwargs(["test"], monkeypatch)["one_question_at_a_time"] is False
