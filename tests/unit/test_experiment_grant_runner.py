"""Offline preflight for the isolated grant-funded MiniBench entry point."""

import argparse
from pathlib import Path

import pytest

from sibyl.ops.experiment_grant import checked_session, preflight_research
from sibyl.platform.experiment_funding import GRANT_KEY_ENV


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_id="test-run",
        ledger=tmp_path / "spend.sqlite3",
        max_cost_usd="3",
        reserve_per_call_usd="1",
    )


def test_key_is_required_only_for_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GRANT_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=GRANT_KEY_ENV):
        checked_session(_args(tmp_path))


def test_grant_key_must_differ_from_existing_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GRANT_KEY_ENV, "dummy-grant")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-grant")
    with pytest.raises(ValueError, match="distinct"):
        checked_session(_args(tmp_path))


def test_direct_paid_research_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_SEARCH_ENABLED", "true")
    with pytest.raises(ValueError, match="ANTHROPIC_SEARCH_ENABLED"):
        preflight_research()


def test_per_call_reservation_must_fit_run_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GRANT_KEY_ENV, "dummy-grant")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
    args = _args(tmp_path)
    args.max_cost_usd = "0.5"
    with pytest.raises(ValueError, match="per-call"):
        checked_session(args)
