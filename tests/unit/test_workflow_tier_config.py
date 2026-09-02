"""The tier each workflow runs at, asserted from the yaml.

A missing tier pin does not break a run, it DOWNGRADES one -- and the failure surfaces
as a refusal at dispatch time, or worse, as a prod run that quietly used free models.
Both are cheap to prevent here and expensive to notice in a log.

Deliberately parsed as text rather than with a yaml library: these files are also read by
GitHub, and what matters is the literal line, not a normalised document.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).parents[2] / ".github" / "workflows"
PROD_WORKFLOWS = [
    "run_bot_on_tournament.yaml",
    "run_bot_on_minibench.yaml",
    "run_bot_on_metaculus_cup.yaml",
    "test_bot_basic.yaml",
]
FREE_WORKFLOW = "test_bot.yaml"

# Keys that buy access to a paid model or a paid research API. The free lane must pass
# none of them: the tier guard already vetoes those providers, but withholding the keys
# means a bug in the guard still cannot spend.
PAID_KEY_NAMES = [
    "OAI_ANTH_OPENROUTER_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "PERPLEXITY_API_KEY",
    "EXA_API_KEY",
    "ASKNEWS_CLIENT_ID",
    "ASKNEWS_SECRET",
    "GOOGLE_API_KEY",
]


def _text(name: str) -> str:
    return (WORKFLOWS / name).read_text()


@pytest.mark.parametrize("name", PROD_WORKFLOWS)
def test_prod_workflows_pin_the_prod_tier(name: str) -> None:
    assert "SIBYL_MODEL_TIER: 'prod'" in _text(name)


def test_the_free_lane_pins_the_free_tier() -> None:
    assert "SIBYL_MODEL_TIER: 'free'" in _text(FREE_WORKFLOW)


def test_the_free_lane_runs_test_mode_on_the_free_tier() -> None:
    assert re.search(r"python -m sibyl test --tier free\b", _text(FREE_WORKFLOW))


@pytest.mark.parametrize("key", PAID_KEY_NAMES)
def test_the_free_lane_passes_no_paid_key(key: str) -> None:
    assert key not in _text(FREE_WORKFLOW), f"{FREE_WORKFLOW} must not receive {key}"


@pytest.mark.parametrize(
    "flag",
    [
        "NATIVE_SEARCH_ENABLED",
        "GEMINI_SEARCH_ENABLED",
        "ANTHROPIC_SEARCH_ENABLED",
        "GAP_FILL_ENABLED",
        "GAP_FILL_V2_ENABLED",
    ],
)
def test_the_free_lane_enables_no_paid_search_provider(flag: str) -> None:
    assert f"{flag}: 'true'" not in _text(FREE_WORKFLOW)


@pytest.mark.parametrize("name", [*PROD_WORKFLOWS, FREE_WORKFLOW])
def test_no_bot_workflow_has_a_schedule(name: str) -> None:
    """Manual dispatch only. A schedule is something the operator adds deliberately."""
    text = _text(name)
    assert not re.search(r"^\s*schedule:", text, re.MULTILINE)
    assert not re.search(r"^\s*-?\s*cron:", text, re.MULTILINE)
