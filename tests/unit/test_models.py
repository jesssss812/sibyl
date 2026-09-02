"""The model roster and the tier that selects it.

The tier is the repo's spend guard expressed in code rather than in prose: FREE cannot
bill, PROD can, and which one a run gets is decided here. These tests exist because the
failure mode is silent in both directions -- a prod run downgraded to free models
publishes bad forecasts to a scored tournament, and a free run upgraded to prod spends
real money -- and neither shows up as an exception.
"""

from __future__ import annotations

import pytest

from sibyl.config.models import ModelTier, active_tier
from sibyl.config.settings import MODEL_TIER_ENV


class TestActiveTier:
    def test_defaults_to_free_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The deliberate behaviour change: a run that forgets the flag cannot spend."""
        monkeypatch.delenv(MODEL_TIER_ENV, raising=False)
        assert active_tier() is ModelTier.FREE

    def test_empty_and_whitespace_are_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An unset GitHub secret arrives as "", not as absent.
        for raw in ("", "   "):
            monkeypatch.setenv(MODEL_TIER_ENV, raw)
            assert active_tier() is ModelTier.FREE, raw

    @pytest.mark.parametrize("raw", ["prod", "PROD", " Prod "])
    def test_prod_is_selected_explicitly_and_case_insensitively(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv(MODEL_TIER_ENV, raw)
        assert active_tier() is ModelTier.PROD

    @pytest.mark.parametrize("raw", ["production", "paid", "cheap", "true", "1"])
    def test_an_unrecognised_tier_raises_rather_than_guessing(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        """Neither fallback is safe, so there is no fallback.

        Defaulting an unrecognised value to FREE would silently downgrade a prod run and
        publish free-model forecasts to a scored tournament; defaulting to PROD would
        silently spend. "production" is the specific typo worth naming -- it is the word
        a person reaches for, and it is not the value.
        """
        monkeypatch.setenv(MODEL_TIER_ENV, raw)
        with pytest.raises(ValueError, match="not a known model tier"):
            active_tier()

    def test_the_error_names_the_valid_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MODEL_TIER_ENV, "production")
        with pytest.raises(ValueError, match="free, prod"):
            active_tier()


class TestModelTier:
    def test_is_a_string_enum_so_it_survives_env_and_yaml_round_trips(self) -> None:
        # The workflows set this as a YAML string; StrEnum means `tier == "prod"` holds
        # without a .value at every comparison site.
        assert ModelTier.PROD == "prod"
        assert ModelTier.FREE == "free"

    def test_exactly_two_tiers(self) -> None:
        # A third tier would need its own roster and its own refusal rules; adding one
        # should be a deliberate change that fails this test first.
        assert sorted(t.value for t in ModelTier) == ["free", "prod"]
