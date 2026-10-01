"""CLI spend refusal happens before logs, clients, or stage dispatch."""
from unittest.mock import patch

from sibyl.eval.ablation.cli import main


def test_default_ablation_refuses_paid_stages_when_tier_unset(monkeypatch):
    monkeypatch.delenv("SIBYL_MODEL_TIER", raising=False)
    with patch("sibyl.eval.ablation.cli._configure_logging") as logs, \
         patch("sibyl.eval.ablation.cli.run_ablation") as runner:
        assert main([]) == 2
    logs.assert_not_called()
    runner.assert_not_called()


def test_score_only_is_available_without_prod_tier(monkeypatch):
    monkeypatch.delenv("SIBYL_MODEL_TIER", raising=False)
    with patch("sibyl.eval.ablation.cli._configure_logging"), \
         patch("sibyl.eval.ablation.cli.run_ablation", return_value=0) as runner:
        assert main(["--stages", "score"]) == 0
    runner.assert_called_once()
