"""Invariants for the 2026-08-31 move of Anthropic slots off OpenRouter.

These are regression tests for a SILENT failure mode, which is why they exist at
all. `forecasting_tools` sets ``litellm.drop_params = True`` process-globally, and
litellm's Anthropic transform advertises ``reasoning_effort`` but NOT the
OpenRouter-shaped ``reasoning={"effort": ...}`` dict, nor ``verbosity``. Carrying
either kwarg across the swap therefore does not raise — it is quietly discarded,
and the slot runs at default effort with nothing in the logs to say so.

A cheap assertion here is the only thing standing between that and a forecaster
silently downgrading itself in production.
"""

import pytest

from sibyl.config.models import FORECASTER_LLMS, STACKER_LLM
from sibyl.platform.key_fallback import should_route_via_primary_key

# Every live LLM slot that is expected to speak to Anthropic directly.
_DIRECT_ANTHROPIC_SLOTS = [
    ("forecaster", FORECASTER_LLMS[1]),
    ("stacker", STACKER_LLM),
]

_ALL_LIVE_SLOTS = [("forecaster", llm) for llm in FORECASTER_LLMS] + [("stacker", STACKER_LLM)]


@pytest.mark.parametrize(("label", "llm"), _ALL_LIVE_SLOTS, ids=lambda v: v if isinstance(v, str) else "")
def test_no_live_slot_routes_anthropic_through_openrouter(label: str, llm: object) -> None:
    model = llm.model  # type: ignore[attr-defined]
    assert not model.startswith("openrouter/anthropic/"), (
        f"{label} slot {model!r} still routes Anthropic via OpenRouter; the primary key is "
        "exhausted and the fallback OpenRouter balance cannot cover the max_tokens reservation"
    )


@pytest.mark.parametrize(("label", "llm"), _DIRECT_ANTHROPIC_SLOTS, ids=lambda v: v if isinstance(v, str) else "")
def test_direct_slots_use_anthropic_prefix(label: str, llm: object) -> None:
    assert llm.model.startswith("anthropic/"), f"{label}: {llm.model!r}"  # type: ignore[attr-defined]


@pytest.mark.parametrize(("label", "llm"), _DIRECT_ANTHROPIC_SLOTS, ids=lambda v: v if isinstance(v, str) else "")
def test_direct_slots_bypass_the_primary_key_wrapper(label: str, llm: object) -> None:
    """A bare ``anthropic/`` slug must fall through to a plain GeneralLlm.

    ``should_route_via_primary_key`` gates on an ``openrouter/`` prefix, so this
    also documents WHY calling ``build_llm_with_openrouter_fallback`` on a direct
    slug is safe rather than accidental.
    """
    assert should_route_via_primary_key(llm.model) is False  # type: ignore[attr-defined]


@pytest.mark.parametrize(("label", "llm"), _DIRECT_ANTHROPIC_SLOTS, ids=lambda v: v if isinstance(v, str) else "")
def test_direct_slots_send_scalar_reasoning_effort_not_the_openrouter_dict(label: str, llm: object) -> None:
    kwargs = llm.litellm_kwargs  # type: ignore[attr-defined]
    assert "reasoning" not in kwargs, (
        f"{label}: the OpenRouter dict form reasoning={{'effort': ...}} is NOT a supported "
        "Anthropic param and drop_params=True discards it silently — use reasoning_effort=..."
    )
    assert kwargs.get("reasoning_effort") == "xhigh", f"{label}: {kwargs.get('reasoning_effort')!r}"


@pytest.mark.parametrize(("label", "llm"), _DIRECT_ANTHROPIC_SLOTS, ids=lambda v: v if isinstance(v, str) else "")
def test_direct_slots_do_not_carry_openrouter_only_verbosity(label: str, llm: object) -> None:
    kwargs = llm.litellm_kwargs  # type: ignore[attr-defined]
    assert "verbosity" not in kwargs
    assert "verbosity" not in (kwargs.get("extra_body") or {}), (
        f"{label}: verbosity is an OpenRouter-only knob with no Anthropic equivalent"
    )


@pytest.mark.parametrize(("label", "llm"), _DIRECT_ANTHROPIC_SLOTS, ids=lambda v: v if isinstance(v, str) else "")
def test_direct_slot_model_ids_use_anthropics_dash_form(label: str, llm: object) -> None:
    """Anthropic ids are dash-separated; OpenRouter's dotted form 404s upstream."""
    bare = llm.model.split("/", 1)[1]  # type: ignore[attr-defined]
    assert "." not in bare, f"{label}: {bare!r} looks like an OpenRouter id (dots), not an Anthropic one"


@pytest.mark.parametrize(("label", "llm"), _DIRECT_ANTHROPIC_SLOTS, ids=lambda v: v if isinstance(v, str) else "")
def test_effort_xhigh_survives_local_anthropic_request_transform(label: str, llm: object) -> None:
    """The local adapter must retain xhigh; remote availability needs a paid check."""
    bare = llm.model.split("/", 1)[1]  # type: ignore[attr-defined]
    # The bundled cost map can lag model support. Exercise the request transformer,
    # which is the behavior we need; absence of pricing metadata is not rejection.
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    params = AnthropicConfig().map_openai_params({"reasoning_effort": "xhigh"}, {}, bare, False)
    assert params["thinking"] == {"type": "adaptive"}, label
    assert params["output_config"] == {"effort": "xhigh"}, label



def test_forecaster_display_names_survive_the_reroute() -> None:
    """performance_analysis maps 'Forecaster N' back to a model via these names."""
    from sibyl.config.models import FORECASTER_MODEL_NAMES

    assert "claude-opus-5" in FORECASTER_MODEL_NAMES
    assert all("/" not in name for name in FORECASTER_MODEL_NAMES)


def test_opus_five_xhigh_request_transformation_is_adaptive() -> None:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    params = AnthropicConfig().map_openai_params(
        {"reasoning_effort": "xhigh"}, {}, "claude-opus-5", False)
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "xhigh"}
