"""The market and reference-class LLM stages follow the run's tier.

They read fixed prod configs, so a free-tier run paid for GPT-6 Luna on the personal
key (seen in the 2026-10-18 free rehearsal). Each stage must use the active tier's
roster slot.
"""

from typing import Any

import pytest

from sibyl.config.models import ModelTier, set_run_tier
from sibyl.research import prediction_market, reference_class


@pytest.fixture
def tier():
    def _set(value: ModelTier) -> None:
        set_run_tier(value)

    yield _set
    set_run_tier(None)


@pytest.mark.parametrize("stage", [prediction_market._query_author_config, prediction_market._ranker_config])
def test_market_stages_use_a_free_model_on_the_free_tier(tier, stage) -> None:
    tier(ModelTier.FREE)
    assert stage()["model"].endswith(":free")


@pytest.mark.parametrize("stage", [prediction_market._query_author_config, prediction_market._ranker_config])
def test_market_stages_keep_the_prod_model_on_the_prod_tier(tier, stage) -> None:
    tier(ModelTier.PROD)
    assert stage()["model"] == "openrouter/openai/gpt-6-luna"


@pytest.mark.asyncio
async def test_reference_class_ranker_uses_a_free_model_on_the_free_tier(tier, monkeypatch) -> None:
    tier(ModelTier.FREE)
    built: list[dict[str, Any]] = []

    def fake_build(**kwargs: Any) -> Any:
        built.append(kwargs)
        return object()

    async def fake_retry(call: Any, **_: Any) -> str:
        return "[]"

    monkeypatch.setattr("sibyl.platform.key_fallback.build_llm_with_openrouter_fallback", fake_build)
    monkeypatch.setattr("sibyl.platform.llm_stage.label_stage", lambda llm, stage: llm)
    monkeypatch.setattr("sibyl.platform.llm_retry.invoke_with_transient_retry", fake_retry)

    assert await reference_class._rank_with_llm("rank these") == "[]"
    assert built[0]["model"].endswith(":free")
