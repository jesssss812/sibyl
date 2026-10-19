"""The MiniBench polling loop: one run at a time, skipped pulls, and its stop conditions."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from sibyl.ops import minibench_loop as loop

_T0 = datetime(2026, 10, 19, 6, 0, tzinfo=timezone.utc)


class _Clock:
    def __init__(self) -> None:
        self.now = _T0

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class _Ledger:
    def __init__(self, remaining: list[Decimal], personal: Decimal = Decimal(0), per_run: Decimal = Decimal(0)) -> None:
        self.remaining = remaining
        self.personal = personal
        self.per_run = per_run

    def grant_spend_usd(self, run_ids: list[str]) -> Decimal:
        return self.per_run * len(run_ids)

    def grant_remaining_usd(self) -> Decimal:
        return self.remaining.pop(0) if len(self.remaining) > 1 else self.remaining[0]

    def personal_spend_usd(self, run_ids: list[str]) -> Decimal:
        return self.personal


def _config(**overrides) -> loop.LoopConfig:
    values = {
        "loop_id": "mb",
        "interval": timedelta(minutes=15),
        "until": _T0 + timedelta(hours=2),
        "max_questions": 10,
        "grant_budget_usd": Decimal("15"),
        "archive_dir": Path("backtests/minibench_runs"),
        "stop_below_usd": Decimal("20"),
        "max_personal_usd": Decimal("2"),
    }
    values.update(overrides)
    return loop.LoopConfig(**values)


def _drive(config, ledger, run_minutes: list[float], codes: list[int] | None = None):
    clock = _Clock()
    starts: list[datetime] = []
    commands: list[list[str]] = []
    messages: list[str] = []
    durations = list(run_minutes)
    exit_codes = list(codes or [])

    def run(command: list[str], env: dict[str, str]) -> int:
        starts.append(clock.now)
        commands.append(command)
        clock.now += timedelta(minutes=durations.pop(0) if durations else 1)
        return exit_codes.pop(0) if exit_codes else 0

    reason = loop.run_loop(config, ledger, run=run, env={}, clock=clock, sleep=clock.sleep, say=messages.append)
    return reason, starts, commands, messages


def test_pulls_every_interval_and_never_overlaps():
    reason, starts, _, messages = _drive(_config(), _Ledger([Decimal(90)]), [20, 5, 5])
    # The 06:15 pull came due while the first run was still working, so it was skipped.
    assert [s.strftime("%H:%M") for s in starts[:3]] == ["06:00", "06:30", "06:45"]
    assert any("skipped 1 pull" in m for m in messages)
    assert "until" in reason


def test_each_run_is_one_at_a_time_on_the_grant():
    _, _, commands, _ = _drive(_config(until=_T0 + timedelta(minutes=1)), _Ledger([Decimal(90)]), [1])
    command = commands[0]
    assert command[2:4] == ["sibyl", "minibench"]
    assert "--one-at-a-time" in command
    assert command[command.index("--tier") + 1] == "prod"
    assert command[command.index("--grant-budget-usd") + 1] == "15"


def test_stops_before_the_grant_falls_below_the_floor():
    reason, starts, _, _ = _drive(_config(), _Ledger([Decimal(60), Decimal(36), Decimal(34)]), [1, 1, 1])
    assert len(starts) == 2
    assert "below --stop-below-usd" in reason


def test_stops_when_the_personal_key_has_paid_too_much():
    reason, starts, _, _ = _drive(_config(), _Ledger([Decimal(90)], personal=Decimal("2.5")), [1])
    assert starts == []
    assert "personal key" in reason


def test_a_refused_run_stops_the_loop_and_a_degraded_one_does_not():
    reason, starts, _, _ = _drive(_config(), _Ledger([Decimal(90)]), [1, 1, 1], codes=[1, 2])
    assert len(starts) == 2
    assert "refused" in reason


def test_next_pull_counts_the_boundaries_passed_over():
    assert loop.next_pull(_T0, _T0 + timedelta(minutes=31), timedelta(minutes=15)) == (
        _T0 + timedelta(minutes=45),
        2,
    )


def test_workflow_settings_are_the_literal_flags_of_the_run_step():
    settings = loop.workflow_settings()
    assert settings["SIBYL_MODEL_TIER"] == "prod"
    # The first round's cost cuts (docs/decisions/2026-10-18-first-minibench-round.md).
    assert settings["GAP_FILL_ENABLED"] == "true"
    assert settings["GAP_FILL_V2_ENABLED"] == "false"
    assert settings["PREDICTION_MARKETS_ENABLED"] == "false"
    assert settings["NATIVE_SEARCH_CONTEXT_SIZE"] == "medium"
    assert not [value for value in settings.values() if "${{" in value]


def test_refuses_without_an_existing_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert loop.main(["--ledger", str(tmp_path / "missing.sqlite3")]) == loop.REFUSED_EXIT_CODE
    assert "no spend ledger" in capsys.readouterr().out


def test_missing_secrets_names_research_credentials_and_the_grant_key():
    env = {"METACULUS_TOKEN": "t", "ASKNEWS_CLIENT_ID": " "}
    missing = loop.missing_secrets(env)
    assert "ASKNEWS_CLIENT_ID" in missing  # blank counts as missing
    assert loop.GRANT_KEY_ENV in missing
    assert "METACULUS_TOKEN" not in missing
    assert "OAI_ANTH_OPENROUTER_KEY" not in missing  # deliberately unset this season


def test_a_timed_out_run_does_not_stop_the_loop():
    reason, starts, _, messages = _drive(_config(), _Ledger([Decimal(90)]), [1, 1], codes=[loop.TIMED_OUT_EXIT_CODE])
    assert len(starts) >= 2
    assert any("run-timeout" in m for m in messages)
    assert "until" in reason


def test_stream_stops_a_run_at_its_timeout(tmp_path: Path):
    import io
    import sys

    log = io.StringIO()
    code = loop._stream([sys.executable, "-c", "import time; time.sleep(30)"], {}, log, timedelta(seconds=0.5))
    assert code == loop.TIMED_OUT_EXIT_CODE


class _BrokenLedger(_Ledger):
    def __init__(self, failures: int) -> None:
        super().__init__([Decimal(90)])
        self.failures = failures

    def grant_remaining_usd(self) -> Decimal:
        if self.failures:
            self.failures -= 1
            raise OSError("database is locked")
        return Decimal(90)


def test_an_unreadable_ledger_skips_that_pull_and_the_loop_survives():
    reason, starts, _, messages = _drive(_config(), _BrokenLedger(failures=1), [1, 1])
    assert starts and starts[0].strftime("%H:%M") == "06:15"
    assert any("could not read the spend ledger" in m for m in messages)
    assert "until" in reason


def test_a_run_that_cannot_start_does_not_kill_the_loop():
    clock = _Clock()
    attempts: list[datetime] = []

    def run(command: list[str], env: dict[str, str]) -> int:
        attempts.append(clock.now)
        if len(attempts) == 1:
            raise FileNotFoundError("python")
        return 0

    reason = loop.run_loop(
        _config(), _Ledger([Decimal(90)]), run=run, env={}, clock=clock, sleep=clock.sleep, say=lambda m: None
    )
    assert len(attempts) > 1
    assert "until" in reason


def test_starts_with_a_warning_when_only_provider_credentials_are_missing(tmp_path, monkeypatch, capsys):
    ledger = tmp_path / "ledger.sqlite3"
    ledger.write_bytes(b"")
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.setenv(loop.GRANT_KEY_ENV, "grant")
    monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
    started: list[loop.LoopConfig] = []
    monkeypatch.setattr(loop, "run_loop", lambda config, *a, **k: started.append(config) or "stopped")
    argv = ["--ledger", str(ledger), "--archive-dir", str(tmp_path / "runs")]
    assert loop.main(argv) == 0
    assert started
    assert "ASKNEWS_CLIENT_ID" in capsys.readouterr().out


def test_refuses_without_the_grant_key(tmp_path, monkeypatch, capsys):
    ledger = tmp_path / "ledger.sqlite3"
    ledger.write_bytes(b"")
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.setenv(loop.GRANT_KEY_ENV, "")
    assert loop.main(["--ledger", str(ledger)]) == loop.REFUSED_EXIT_CODE
    assert loop.GRANT_KEY_ENV in capsys.readouterr().out


def test_stops_before_a_run_could_pass_the_round_budget():
    config = _config(round_budget_usd=Decimal("50"), grant_budget_usd=Decimal("20"))
    reason, starts, _, _ = _drive(config, _Ledger([Decimal(90)], per_run=Decimal("12")), [1, 1, 1, 1])
    # 0+20, 12+20 and 24+20 fit under 50; 36+20 would not.
    assert len(starts) == 3
    assert "--round-budget-usd" in reason


def test_loop_ledger_reads_the_real_ledger(tmp_path: Path):
    from sibyl.platform.experiment_funding import SpendLedger

    path = tmp_path / "spend.sqlite3"
    ledger = SpendLedger(path)
    ledger.start_run("mb-1", Decimal("20"))
    ledger.reserve("a", "mb-1", "openrouter/test", "forecast", Decimal("2"))
    view = loop.Ledger(path)
    assert view.grant_spend_usd(["mb-1", "mb-2"]) == Decimal("2")
    assert view.personal_spend_usd(["mb-1"]) == Decimal("0")
    assert view.grant_remaining_usd() == Decimal("93")


def test_a_rehearsal_is_one_pull_on_the_testing_area():
    reason, starts, commands, _ = _drive(_config(mode="test", once=True), _Ledger([Decimal(90)]), [1])
    assert len(starts) == 1
    assert commands[0][3] == "test"
    assert "--one-at-a-time" in commands[0]
    assert "once" in reason


def test_a_free_rehearsal_runs_the_free_tier_without_a_grant():
    config = _config(mode="test", once=True, tier="free", round_budget_usd=Decimal("1"))
    reason, starts, commands, _ = _drive(config, _Ledger([Decimal(1)], per_run=Decimal("50")), [1])
    command = commands[0]
    assert command[command.index("--tier") + 1] == "free"
    assert "--grant-budget-usd" not in command
    assert len(starts) == 1  # grant floors and round caps do not apply to a free run
    assert "once" in reason


def test_the_free_tier_is_refused_on_minibench(tmp_path: Path, capsys):
    ledger = tmp_path / "ledger.sqlite3"
    ledger.write_bytes(b"")
    assert loop.main(["--ledger", str(ledger), "--tier", "free"]) == loop.REFUSED_EXIT_CODE
    assert "--mode test" in capsys.readouterr().out
