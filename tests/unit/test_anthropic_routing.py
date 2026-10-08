"""The Anthropic slots route through OpenRouter, with OpenRouter's effort shape.

`forecasting_tools` sets ``litellm.drop_params = True`` process-globally, so a
transport's unsupported kwarg is discarded rather than rejected. On OpenRouter the
effort must be the ``reasoning={"effort": ...}`` dict; the scalar
``reasoning_effort`` is the Anthropic-direct spelling and would leave the slot at
default effort with nothing in the logs to say so.
"""

import pytest

from sibyl.config.models import FORECASTER_LLMS, FORECASTER_MODEL_NAMES, STACKER_LLM

_ANTHROPIC_SLOTS = [
    ("forecaster", FORECASTER_LLMS[1], "openrouter/anthropic/claude-opus-5.5"),
    ("stacker", STACKER_LLM, "openrouter/anthropic/claude-opus-5.5"),
]


@pytest.mark.parametrize(("label", "llm", "expected"), _ANTHROPIC_SLOTS)
def test_anthropic_slots_use_openrouter_and_its_effort_shape(label: str, llm: object, expected: str) -> None:
    assert llm.model == expected, label  # type: ignore[attr-defined]
    kwargs = llm.litellm_kwargs  # type: ignore[attr-defined]
    assert kwargs.get("reasoning") == {"effort": "xhigh"}, label
    assert "reasoning_effort" not in kwargs, label


def test_forecaster_display_names_strip_the_route() -> None:
    assert "claude-opus-5.5" in FORECASTER_MODEL_NAMES
    assert all("/" not in name for name in FORECASTER_MODEL_NAMES)
