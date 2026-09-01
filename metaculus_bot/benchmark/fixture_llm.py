"""Zero-cost fixture forecaster: the real pipeline, a fake model.

Runs the entire backtest path — extraction ladder, sanitize, tail widening, PCHIP CDF,
aggregation, the A3 guard, scoring against real resolutions — with only the model's
*judgment* replaced. Questions and outcomes come from the live Metaculus API, which is
read-only and free; nothing else in the path is faked.

Why this exists rather than "just run it cheaply"
-------------------------------------------------
The eval layer is the part of this project that has to be trusted, and you cannot
validate an eval harness without data. A fixture arm generates that data at zero cost
*and* with a known ground truth, which real models can never give you at any price:

* **A/A null test.** Two arms with identical fixture config must produce identical
  forecasts, so the paired harness must report exactly no difference. If it reports one,
  the harness is broken — and that is only detectable when you control both arms.
* **Interval coverage.** With a seeded generator you set the true effect size, so you can
  check whether a bootstrap CI actually covers it ~95% of the time at n=60 — which is
  precisely where `ablation/scoring_report.py` warns its own percentile intervals may
  under-cover.

**It is free by construction, not by convention.** No API client is built and no socket
is opened, so an accidental fixture run costs nothing. That is deliberate: the repo's
cost rule is otherwise prose in `AGENTS.md`, and prose depends on remembering.

The input contract is the real prompt
-------------------------------------
``invoke()`` receives the same prompt string a real forecaster gets, and infers the
question type and parameters from it. That is a feature: the fixture consumes the
production prompt, so a prompt change that breaks the machine-readable contract breaks
the fixture too, loudly, in a free test rather than in a paid run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.general_llm import ModelInputType

logger = logging.getLogger(__name__)

# Every fixture model carries this prefix so a run log, a published comment, or a scored
# row can never let a fixture arm be mistaken for a real one.
FIXTURE_MODEL_PREFIX = "fixture/"

_DEFAULT_NAMES = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")

# Bound phrasings emitted by numeric/utils.bound_messages. Parsed rather than guessed so
# the fixture stays inside a closed range instead of leaning on downstream repair — a
# forecast that only survives because sanitize clamped it exercises the repair path, not
# the happy path, and would quietly weaken every number the run produces.
_UPPER_CLOSED_RE = re.compile(r"outcome can not be higher than ([-\d.eE+]+)")
_LOWER_CLOSED_RE = re.compile(r"outcome can not be lower than ([-\d.eE+]+)")
_UPPER_OPEN_RE = re.compile(r"The upper bound is open: ([-\d.eE+]+) is the top")
_LOWER_OPEN_RE = re.compile(r"The lower bound is open: ([-\d.eE+]+) is the bottom")

# The MC prompt lists the real option names in its schema example; they must be echoed
# back exactly, since the parser matches on them.
_OPTION_PROBS_RE = re.compile(r'"option_probs":\s*\{(.*?)\}', re.DOTALL)
_OPTION_KEY_RE = re.compile(r'"([^"]+)"\s*:')


def _as_float(text: str | None) -> float | None:
    if text is None:
        return None
    # The bound messages end in a full stop, and the character class captures it, so
    # "20.0." reaches here and float() rejects it. Stripping is what stops a parse failure
    # from silently degrading to the [0, 100] fallback — which is worse than a crash,
    # because the run still completes and every numeric forecast is quietly off-scale.
    try:
        value = float(text.rstrip("."))
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


class FixtureLlm(GeneralLlm):
    """A deterministic stand-in for ``GeneralLlm``. Never touches the network.

    Subclassed rather than duck-typed. Duck-typing looked safer — "don't inherit the real
    client" — but it was wrong twice: ``GeneralLlm.__init__`` builds no client and opens no
    socket (the repo's own tests construct one under the egress guard on every run), and
    the framework does string operations on LLM values, so a non-``GeneralLlm`` blows up
    with ``AttributeError: 'FixtureLlm' object has no attribute 'startswith'`` deep inside
    purpose validation.

    Only ``invoke`` is overridden. That is the single method that would reach the network,
    so overriding it is precisely the guarantee — everything else behaves as the framework
    expects, which is what makes this a drop-in rather than a lookalike.
    """

    def __init__(self, model: str, seed: int = 0) -> None:
        if not model.startswith(FIXTURE_MODEL_PREFIX):
            raise ValueError(
                f"fixture model names must start with {FIXTURE_MODEL_PREFIX!r} so a fixture arm is never "
                f"mistaken for a real one in logs or scored output; got {model!r}"
            )
        super().__init__(model=model, temperature=None)
        self.seed = seed

    # -- deterministic pseudo-randomness -----------------------------------------
    #
    # Seeded by (model, seed, prompt) so a rerun reproduces exactly, different models
    # disagree (spread metrics and the stacking route need genuine disagreement to fire
    # at all), and a different `seed` yields a different arm — which is what makes an A/A
    # comparison identical and an A/B comparison differ by construction.

    def _unit(self, prompt: str, salt: str) -> float:
        digest = hashlib.sha256(f"{self.model}|{self.seed}|{salt}|{prompt}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    async def invoke(self, prompt: ModelInputType, system_prompt: str | None = None) -> str:
        """Return a response whose fenced STRUCTURED FORECAST block the real parser accepts.

        Signature mirrors ``GeneralLlm.invoke`` exactly. A narrowed override would type-check
        locally and then break at the one call site that passes ``system_prompt``.
        """
        prompt = prompt if isinstance(prompt, str) else str(prompt)
        if '"question_type": "numeric"' in prompt or "declared_percentiles" in prompt:
            return self._numeric(prompt)
        if '"question_type": "multiple_choice"' in prompt or "option_probs" in prompt:
            return self._multiple_choice(prompt)
        return self._binary(prompt)

    # -- per-question-type generators --------------------------------------------

    def _wrap(self, payload: dict) -> str:
        """Emit prose then the fenced block, in the shape the prompt demands.

        The prose matters: the extraction ladder's first rung looks for the fenced block
        inside a full response, so a bare JSON document would exercise a different rung
        than production takes.
        """
        return (
            "FIXTURE RESPONSE — deterministic stand-in, no model was called.\n\n"
            "Reasoning is intentionally omitted; this arm exists to exercise the pipeline "
            "below the model, not to forecast.\n\n"
            "── STRUCTURED FORECAST (machine-readable; REQUIRED) ──\n\n"
            "```json\n" + json.dumps(payload, indent=2) + "\n```"
        )

    def _binary(self, prompt: str) -> str:
        # Held inside [0.05, 0.95] so the fixture's own spread, not a clamp, drives the
        # value — and so the A3 guard is reachable without being unconditionally tripped.
        prob = round(0.05 + 0.90 * self._unit(prompt, "binary"), 4)
        return self._wrap({"question_type": "binary", "posterior_prob": prob})

    def _multiple_choice(self, prompt: str) -> str:
        options = self._extract_options(prompt)
        weights = [self._unit(prompt, f"mc:{i}:{name}") + 0.05 for i, name in enumerate(options)]
        total = sum(weights)
        probs = {name: round(w / total, 6) for name, w in zip(options, weights)}
        # Absorb rounding drift into the largest option so the set sums to exactly 1.0,
        # which the parser requires.
        drift = round(1.0 - sum(probs.values()), 6)
        if drift and probs:
            top = max(probs, key=lambda k: probs[k])
            probs[top] = round(probs[top] + drift, 6)
        return self._wrap({"question_type": "multiple_choice", "option_probs": probs})

    def _numeric(self, prompt: str) -> str:
        from metaculus_bot.numeric.config import STANDARD_PERCENTILES

        lo, hi = self._extract_bounds(prompt)
        span = hi - lo
        # Centre the distribution somewhere in the middle 60% of the range and give it a
        # width that varies by model, so arms disagree on both location and spread.
        centre = lo + span * (0.25 + 0.5 * self._unit(prompt, "centre"))
        # Kept narrow on purpose. A wider spread piles percentiles against the bounds,
        # and a clamped ladder is a degenerate distribution that tests the repair path
        # instead of the pipeline.
        width = span * (0.04 + 0.08 * self._unit(prompt, "width"))

        values: list[float] = []
        for p in STANDARD_PERCENTILES:
            # A logistic quantile keeps the shape plausible and monotone by construction.
            z = -1.0 if p <= 0.0 else (1.0 if p >= 1.0 else (p - 0.5))
            values.append(centre + width * z * 2.0)

        # Clamp into the closed range, then force strict monotonicity. Ties and inversions
        # are what the sanitize layer repairs, and a fixture that leaned on that repair
        # would be measuring the repair rather than the pipeline.
        eps = max(abs(span), 1.0) * 1e-6
        clamped: list[float] = []
        for v in values:
            v = min(max(v, lo), hi)
            if clamped and v <= clamped[-1]:
                v = clamped[-1] + eps
            clamped.append(min(v, hi))
        # If the ladder ran into the ceiling, walk it back down from the top.
        for i in range(len(clamped) - 2, -1, -1):
            if clamped[i] >= clamped[i + 1]:
                clamped[i] = clamped[i + 1] - eps

        declared = {f"{p:g}": round(v, 6) for p, v in zip(STANDARD_PERCENTILES, clamped)}
        return self._wrap({"question_type": "numeric", "declared_percentiles": declared, "outcome_type": "continuous"})

    # -- prompt parsing ------------------------------------------------------------

    @staticmethod
    def _extract_options(prompt: str) -> list[str]:
        """Recover the real option names from the prompt's schema example."""
        match = _OPTION_PROBS_RE.search(prompt)
        if match:
            names = _OPTION_KEY_RE.findall(match.group(1))
            if names:
                return names
        logger.warning("FIXTURE: could not recover MC option names from prompt; using a 2-option fallback")
        return ["Option A", "Option B"]

    @staticmethod
    def _extract_bounds(prompt: str) -> tuple[float, float]:
        """Recover numeric bounds, widening past an open bound the way a forecaster may.

        Falls back to [0, 100] only if the prompt carries no bound message at all, which
        would itself be a prompt-contract change worth noticing.
        """
        upper = _as_float(m.group(1)) if (m := _UPPER_CLOSED_RE.search(prompt)) else None
        lower = _as_float(m.group(1)) if (m := _LOWER_CLOSED_RE.search(prompt)) else None

        if upper is None and (m := _UPPER_OPEN_RE.search(prompt)):
            top = _as_float(m.group(1))
            # An open ceiling means mass may sit above the displayed range.
            upper = None if top is None else top + abs(top) * 0.25 + 1.0
        if lower is None and (m := _LOWER_OPEN_RE.search(prompt)):
            bottom = _as_float(m.group(1))
            lower = None if bottom is None else bottom - abs(bottom) * 0.25 - 1.0

        if lower is None or upper is None or not upper > lower:
            logger.warning("FIXTURE: no usable numeric bounds in prompt; falling back to [0, 100]")
            return 0.0, 100.0
        return lower, upper


def build_fixture_forecasters(count: int = 3, *, seed: int = 0) -> list[FixtureLlm]:
    """A lineup of distinctly-named fixture forecasters.

    Distinct names are required, not cosmetic: identically-seeded arms would agree on
    every question, so spread would be zero everywhere and the aggregation and stacking
    routes would never be exercised.
    """
    if count < 1:
        raise ValueError(f"need at least one fixture forecaster; got {count}")
    names = [_DEFAULT_NAMES[i] if i < len(_DEFAULT_NAMES) else f"m{i}" for i in range(count)]
    return [FixtureLlm(model=f"{FIXTURE_MODEL_PREFIX}{n}", seed=seed) for n in names]


async def fixture_research_provider(question: object) -> str:
    """Canned research, so the research fan-out costs nothing either.

    Deliberately terse and clearly labelled: a fixture run must never produce a bundle
    that could be mistaken for real evidence if it lands in an archive.
    """
    return (
        "## Fixture Research\n\n"
        "Deterministic placeholder. No research provider was called and no model saw this "
        "text. Present only so the pipeline has a bundle of the right shape."
    )


def disable_all_research_providers() -> list[str]:
    """Turn every research provider OFF for the current process. Returns what it cleared.

    Setting a custom ``research_provider`` replaces only the PRIMARY provider; the add-ons
    are chosen independently from env flags in ``_select_research_providers``, so they keep
    firing — and billing — in what looks like a fixture run. That is exactly the partial
    guard this module exists to avoid, and it was caught by a smoke run rather than by
    reasoning, which is the argument for running the thing you claim is free.

    The flag list is derived from ``constants`` by naming convention rather than hardcoded,
    so a provider added later is disabled automatically instead of quietly reopening the
    hole. Same approach as the autouse fixture in ``tests/conftest.py``.
    """
    import os

    from metaculus_bot import constants as _constants

    cleared: list[str] = []
    for attr in dir(_constants):
        if not attr.endswith("_ENABLED_ENV"):
            continue
        env_name = getattr(_constants, attr)
        if not isinstance(env_name, str):
            continue
        # Stacking gates are aggregation, not research, and turning them off would change
        # which aggregation path the fixture run exercises.
        if "STACKING" in env_name:
            continue
        os.environ[env_name] = "false"
        cleared.append(env_name)

    logger.info("FIXTURE: disabled %d research/provider flags: %s", len(cleared), ", ".join(sorted(cleared)))
    return cleared
