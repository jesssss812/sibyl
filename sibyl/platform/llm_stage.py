"""Which pipeline stage an LLM client serves, for the spend ledger's breakdown.

Kept apart from the funding module so client builders can label what they build
without importing the funding machinery.
"""

from __future__ import annotations

import weakref

from forecasting_tools import GeneralLlm

_STAGES: weakref.WeakKeyDictionary[GeneralLlm, str] = weakref.WeakKeyDictionary()


def label_stage(llm: GeneralLlm, stage: str) -> GeneralLlm:
    """Name the stage ``llm`` serves, including any secondary client it falls back to."""
    try:
        _STAGES[llm] = stage
    except TypeError:  # a test double that cannot be weakly referenced
        return llm
    secondary = getattr(llm, "_secondary_llm", None)
    if isinstance(secondary, GeneralLlm):
        _STAGES[secondary] = stage
    return llm


def stage_of(llm: GeneralLlm) -> str:
    return _STAGES.get(llm, "llm")
