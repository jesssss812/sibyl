"""The forecaster roster stays on vendors every paying key can route.

The grant's OpenRouter account reaches Google only through a BYOK route with no
quota, so a Google forecaster 429s there and falls back to the personal key on
every question.
"""

from sibyl.config.models import FORECASTER_LLMS, FORECASTER_MODEL_NAMES


def test_roster_is_sol_opus_and_sonnet():
    assert FORECASTER_MODEL_NAMES == ["gpt-6-sol", "claude-opus-5.5", "claude-sonnet-5.5"]


def test_no_forecaster_routes_to_google():
    assert not [llm.model for llm in FORECASTER_LLMS if "/google/" in llm.model]


def test_sonnet_slot_uses_openrouter_effort_shape():
    sonnet = FORECASTER_LLMS[2]
    assert sonnet.model == "openrouter/anthropic/claude-sonnet-5.5"
    assert sonnet.litellm_kwargs.get("reasoning") == {"effort": "high"}
    assert "reasoning_effort" not in sonnet.litellm_kwargs
