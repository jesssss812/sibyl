"""The suite must be hermetic against the developer's own .env.

``metaculus_bot/config.py`` calls ``load_dotenv()``, so a populated ``.env`` — which
every developer who has actually configured the bot has — leaks its feature flags into
the test process. Six tests silently depended on the flags being unset, i.e. on the
developer NOT having configured the bot. That is the same class of defect as an
un-guarded network egress: it works until someone's machine differs.

Fixed the same way conftest handles egress: an autouse fixture, structural rather than
conventional. Tests that want a flag opt IN via monkeypatch; nothing has to remember to
opt out.

The fixture derives its list from ``constants`` at runtime rather than hard-coding it,
so a newly added provider flag is covered without anyone remembering to update a list —
which is exactly how the hand-rolled enumeration in test_research_orchestrator.py came
to miss TS_ANCHOR_ENABLED.
"""

from __future__ import annotations

import os

from metaculus_bot import constants
from metaculus_bot.constants import env_flag_enabled

# The three per-type stacking gates are deliberately forced ON by conftest's
# _enable_per_type_stacking fixture, so they are out of scope for the isolation guard.
_OWNED_ELSEWHERE = {
    constants.BINARY_STACKING_ENABLED_ENV,
    constants.MC_STACKING_ENABLED_ENV,
    constants.NUMERIC_STACKING_ENABLED_ENV,
}


def _all_feature_flag_names() -> list[str]:
    return [
        v
        for n in dir(constants)
        if n.endswith("_ENABLED_ENV")
        for v in [getattr(constants, n)]
        if v not in _OWNED_ELSEWHERE
    ]


class TestFeatureFlagsAreCleared:
    def test_the_constants_module_actually_exposes_flags(self) -> None:
        """Guards the guard: if the naming convention changes, this test fails loudly
        rather than the fixture silently clearing nothing."""
        assert len(_all_feature_flag_names()) >= 5

    def test_no_feature_flag_is_set_during_tests(self) -> None:
        leaked = {name: os.environ[name] for name in _all_feature_flag_names() if name in os.environ}
        assert leaked == {}, f"feature flags leaked into the test process: {leaked}"

    def test_flags_read_as_their_code_defaults(self) -> None:
        for name in _all_feature_flag_names():
            assert env_flag_enabled(name) is False
            assert env_flag_enabled(name, default=True) is True


class TestOptInStillWorks:
    def test_monkeypatch_can_still_enable_a_flag(self, monkeypatch) -> None:
        monkeypatch.setenv(constants.NATIVE_SEARCH_ENABLED_ENV, "true")
        assert env_flag_enabled(constants.NATIVE_SEARCH_ENABLED_ENV) is True

    def test_the_flag_is_cleared_again_for_the_next_test(self) -> None:
        assert constants.NATIVE_SEARCH_ENABLED_ENV not in os.environ


class TestStackingGatesAreOutOfScope:
    """The carve-out is deliberate; pin it so nobody "fixes" it back into the sweep."""

    def test_stacking_gates_stay_enabled(self) -> None:
        for name in _OWNED_ELSEWHERE:
            assert os.environ.get(name) == "true", f"{name} is owned by conftest's _enable_per_type_stacking fixture"
