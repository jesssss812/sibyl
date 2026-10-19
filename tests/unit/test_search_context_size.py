"""OpenAI web search's context size is a setting a workflow can lower."""

import pytest

from sibyl.config import settings


@pytest.mark.parametrize("raw", [None, "", "  "])
def test_unset_means_high(raw: str | None) -> None:
    assert settings.search_context_size(raw) == "high"


@pytest.mark.parametrize("raw", ["low", "medium", "high", " Medium "])
def test_known_sizes_are_accepted(raw: str) -> None:
    assert settings.search_context_size(raw) == raw.strip().lower()


def test_an_unknown_size_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="NATIVE_SEARCH_CONTEXT_SIZE"):
        settings.search_context_size("huge")


def test_the_native_search_llm_sends_the_configured_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from sibyl.research import providers

    monkeypatch.setattr(settings, "NATIVE_SEARCH_CONTEXT_SIZE", "medium")
    captured: dict = {}
    monkeypatch.setattr(providers, "build_llm_with_openrouter_fallback", lambda **kw: captured.update(kw) or kw)
    monkeypatch.setattr(providers, "label_stage", lambda llm, stage: llm)
    providers.build_native_search_llm()
    assert captured["web_search_options"] == {"search_context_size": "medium"}
