"""Stage labels name the pipeline step in the spend ledger's breakdown."""

from sibyl.config.models import ModelTier, get_roster
from sibyl.platform.key_fallback import FallbackOpenRouterLlm
from sibyl.platform.llm_stage import label_stage, stage_of


def test_every_prod_roster_slot_is_labelled() -> None:
    roster = get_roster(ModelTier.PROD)
    assert {stage_of(llm) for llm in roster.forecasters} == {"forecaster"}
    assert stage_of(roster.parser) == "parser"
    assert stage_of(roster.stacker) == "stacker"
    assert stage_of(roster.stacker_fallback) == "stacker_fallback"
    assert stage_of(roster.summarizer) == "researcher/summarizer", "one client serving two slots names both"


def test_a_fallback_client_labels_its_secondary_too() -> None:
    llm = FallbackOpenRouterLlm(model="openrouter/openai/test", primary_api_key="a", secondary_api_key="b")
    label_stage(llm, "native_search")
    assert llm._secondary_llm is not None
    assert stage_of(llm._secondary_llm) == "native_search"


def test_an_unlabelled_client_is_reported_as_llm() -> None:
    llm = FallbackOpenRouterLlm(model="openrouter/openai/test", primary_api_key="a", secondary_api_key=None)
    assert stage_of(llm) == "llm"
