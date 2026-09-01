"""The reference-class provider is registered, gated, and headed like every other provider.

Separate from test_reference_class.py, which covers the provider's own logic. These tests
pin its integration with ResearchOrchestrator: the env gate, the section header, and the
one property that distinguishes it from the other leakage-sensitive providers — it stays
ON under benchmarking instead of being hard-disabled.
"""

from __future__ import annotations

import pytest
from forecasting_tools import GeneralLlm

from sibyl.config.settings import REFERENCE_CLASS_ENABLED_ENV
from sibyl.research.orchestrator import ResearchOrchestrator


def _orchestrator(is_benchmarking: bool = False) -> ResearchOrchestrator:
    llm = GeneralLlm(model="test-model", temperature=0.0)
    return ResearchOrchestrator(
        default_llm=llm,
        summarizer_llm=llm,
        is_benchmarking=is_benchmarking,
    )


def _provider_names(orch: ResearchOrchestrator) -> list[str]:
    return [name for _, name in orch._select_research_providers()]


class TestEnvGate:
    def test_absent_when_flag_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(REFERENCE_CLASS_ENABLED_ENV, raising=False)
        assert "reference_class" not in _provider_names(_orchestrator())

    def test_absent_when_flag_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REFERENCE_CLASS_ENABLED_ENV, "false")
        assert "reference_class" not in _provider_names(_orchestrator())

    def test_present_when_flag_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REFERENCE_CLASS_ENABLED_ENV, "true")
        assert "reference_class" in _provider_names(_orchestrator())


class TestSurvivesBenchmarking:
    """The whole strategic point: this provider is measurable offline.

    Prediction markets and the resolution-source fetcher return "" under
    is_benchmarking=True, so a backtest cannot price them. Reference-class retrieval
    filters by resolve time instead of switching itself off, so it stays in the run.
    """

    def test_registered_under_benchmarking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REFERENCE_CLASS_ENABLED_ENV, "true")
        assert "reference_class" in _provider_names(_orchestrator(is_benchmarking=True))


class TestHeader:
    def test_has_a_dedicated_h2_header(self) -> None:
        from sibyl.research.reference_class import SECTION_HEADER

        header = ResearchOrchestrator._provider_header("reference_class")
        assert header == SECTION_HEADER
        assert header.startswith("## "), "provider headers must be h2 (see _demote_headings)"

    def test_header_is_not_the_generic_fallback(self) -> None:
        assert ResearchOrchestrator._provider_header("reference_class") != "## Research (reference_class)"
