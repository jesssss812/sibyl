"""Poll MiniBench on a fixed interval and forecast new questions one at a time.

MiniBench questions open at random hours and stay open for about three hours, so a
question is lost if nothing looks for it in time. Every ``--interval-minutes`` this
starts one ``python -m sibyl minibench --one-at-a-time`` run: it lists the open
questions this bot has not forecast, sorts them by scheduled close and forecasts them
one after another. Runs never overlap. A pull that comes due while a run is still
working through its queue is skipped, and the next one is taken at the following
interval.

Each run gets the MiniBench workflow's own settings (the literal ``env`` of its "Run
bot" step), so a local run uses the method the workflow would. Secrets still come from
``.env``; a missing provider credential is reported at start and does not stop the
loop, but a missing grant key does. The loop stops before a run whose budget would
take this loop's grant use past ``--round-budget-usd``, when the grant's local remaining allocation falls below
``--stop-below-usd``, when this loop's runs have paid more than
``--max-personal-usd`` from the personal key, when a run is refused (exit 2), or at
``--until``.

Every run spends and publishes to a scored tournament: start the loop only with the
operator's go for this round. Run it in a visible terminal; output streams there and
is kept in ``<archive-dir>/<loop-id>.log``.

    python -m sibyl.ops.minibench_loop --until 2026-10-23T00:00Z
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, TextIO

import yaml

WORKFLOW = Path(".github/workflows/run_bot_on_minibench.yaml")
RUN_STEP = "Run bot"
REFUSED_EXIT_CODE = 2
TIMED_OUT_EXIT_CODE = 124
GRANT_KEY_ENV = "SIBYL_EXPERIMENT_OPENROUTER_API_KEY"
# Spent for the season and deliberately unset (docs/architecture/overview.md).
NOT_REQUIRED_SECRETS = frozenset({"OAI_ANTH_OPENROUTER_KEY"})


def _run_step_env(path: Path) -> Mapping[str, Any]:
    workflow = yaml.safe_load(path.read_text())
    steps = workflow["jobs"]["forecast_job"]["steps"]
    return next(step for step in steps if step.get("name") == RUN_STEP).get("env") or {}


def workflow_settings(path: Path = WORKFLOW) -> dict[str, str]:
    """The literal env values of the workflow's run step; secret references are left out."""
    return {key: str(value) for key, value in _run_step_env(path).items() if "${{" not in str(value)}


def missing_secrets(env: Mapping[str, str], path: Path = WORKFLOW) -> list[str]:
    """Credentials the workflow's run step gets that this environment lacks, plus the grant key.

    A missing research credential does not stop a run: the provider is skipped and the
    run publishes with less research, a different method from the workflow's.
    """
    wanted = {key for key, value in _run_step_env(path).items() if "${{" in str(value)} - NOT_REQUIRED_SECRETS
    return sorted(key for key in wanted | {GRANT_KEY_ENV} if not env.get(key, "").strip())


@dataclass(frozen=True)
class LoopConfig:
    loop_id: str
    interval: timedelta
    until: datetime | None
    max_questions: int
    grant_budget_usd: Decimal
    archive_dir: Path
    stop_below_usd: Decimal
    max_personal_usd: Decimal
    run_timeout: timedelta = timedelta(hours=3)
    round_budget_usd: Decimal | None = None
    # "test" forecasts the unscored bot-testing area instead: a rehearsal of the same path.
    mode: str = "minibench"
    once: bool = False
    # "free" runs the free roster with no grant: a rehearsal that costs nothing. The bot
    # refuses scored modes on the free tier, so it goes with mode "test".
    tier: str = "prod"


def run_command(config: LoopConfig, run_id: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sibyl",
        config.mode,
        "--tier",
        config.tier,
        "--one-at-a-time",
        "--max-questions",
        str(config.max_questions),
        "--run-id",
        run_id,
        "--archive-dir",
        str(config.archive_dir),
    ]
    if config.tier == "prod":
        command += ["--grant-budget-usd", str(config.grant_budget_usd)]
    return command


def next_pull(started: datetime, now: datetime, interval: timedelta) -> tuple[datetime, int]:
    """The first interval boundary after ``now``, and how many boundaries were passed over."""
    elapsed = now - started
    passed = int(elapsed / interval)
    return started + (passed + 1) * interval, passed


class SpendView(Protocol):
    def grant_remaining_usd(self) -> Decimal: ...

    def personal_spend_usd(self, run_ids: list[str]) -> Decimal: ...

    def grant_spend_usd(self, run_ids: list[str]) -> Decimal: ...


@dataclass
class Ledger:
    """What the loop needs from the spend ledger: grant headroom and personal-key spend."""

    path: Path

    def _open(self) -> Any:
        from sibyl.platform.experiment_funding import SpendLedger  # noqa: PLC0415

        return SpendLedger(self.path)

    def grant_remaining_usd(self) -> Decimal:
        return Decimal(str(self._open().summary()["remaining_local_usd"]))

    def grant_spend_usd(self, run_ids: list[str]) -> Decimal:
        ledger = self._open()
        return sum((ledger.run_liability_usd(run_id) for run_id in run_ids), Decimal(0))

    def personal_spend_usd(self, run_ids: list[str]) -> Decimal:
        from sibyl.platform.experiment_funding import FundingSource  # noqa: PLC0415

        ledger = self._open()
        return sum(
            (
                Decimal(str(ledger.summary(run_id, source=FundingSource.PERSONAL_KEY)["charged_usd"]))
                for run_id in run_ids
            ),
            Decimal(0),
        )


def run_loop(
    config: LoopConfig,
    ledger: SpendView,
    *,
    run: Callable[[list[str], dict[str, str]], int],
    env: dict[str, str],
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    say: Callable[[str], None] = print,
) -> str:
    """Pull until a stop condition holds; return why it stopped."""
    started = clock()
    boundary = 0  # index of the interval boundary the next run starts on
    run_ids: list[str] = []
    while True:
        now = clock()
        if config.until is not None and now >= config.until:
            return f"reached --until {config.until.isoformat()}"
        try:
            remaining = ledger.grant_remaining_usd()
            personal = ledger.personal_spend_usd(run_ids)
            round_spend = ledger.grant_spend_usd(run_ids)
        except Exception as exc:
            # Without the ledger the limits below cannot be checked, so start no run.
            remaining = personal = round_spend = None
            say(f"[loop] could not read the spend ledger ({type(exc).__name__}: {exc}); skipping this pull")
        if (
            config.tier == "prod"
            and remaining is not None
            and remaining - config.grant_budget_usd < config.stop_below_usd
        ):
            return f"grant remaining ${remaining} would fall below --stop-below-usd ${config.stop_below_usd}"
        # A run must start with its whole budget available: a run that exhausts its grant
        # budget sends its remaining calls to the personal key.
        if (
            config.tier == "prod"
            and config.round_budget_usd is not None
            and round_spend is not None
            and round_spend + config.grant_budget_usd > config.round_budget_usd
        ):
            return (
                f"this loop has used ${round_spend} of the grant; another run's ${config.grant_budget_usd} "
                f"would pass --round-budget-usd ${config.round_budget_usd}"
            )
        if personal is not None and personal > config.max_personal_usd:
            return f"personal key paid ${personal} in this loop, over --max-personal-usd ${config.max_personal_usd}"

        code = 0
        run_id = ""
        if remaining is not None:
            run_id = f"{config.loop_id}-{now.strftime('%Y%m%dT%H%M%SZ')}"
            say(f"[loop] {now.isoformat()} pull {len(run_ids) + 1}: run {run_id} (grant remaining ${remaining})")
            run_ids.append(run_id)
            try:
                code = run(run_command(config, run_id), env)
            except Exception as exc:
                say(f"[loop] run {run_id} could not be started ({type(exc).__name__}: {exc}); continuing")
        if code == TIMED_OUT_EXIT_CODE:
            say(f"[loop] run {run_id} passed --run-timeout-minutes and was stopped; continuing")
        elif code == REFUSED_EXIT_CODE:
            return f"run {run_id} was refused (exit {code}); fix the cause before restarting"
        elif code != 0:
            say(f"[loop] run {run_id} exited {code} (a degraded run still published what it could); continuing")

        if config.once:
            return "ran once (--once)"
        pull_at, passed = next_pull(started, clock(), config.interval)
        if passed > boundary:
            say(f"[loop] skipped {passed - boundary} pull(s) while the queue was still running")
        boundary = passed + 1
        wait = (pull_at - clock()).total_seconds()
        if config.until is not None and pull_at >= config.until:
            return f"reached --until {config.until.isoformat()}"
        say(f"[loop] next pull at {pull_at.isoformat()}")
        if wait > 0:
            sleep(wait)


def _stream(command: list[str], env: dict[str, str], log: TextIO, timeout: timedelta) -> int:
    """Run one pull, copying its output to the terminal and the log; stop it at ``timeout``."""
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    timer = threading.Timer(timeout.total_seconds(), process.kill)
    timer.start()
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = process.wait()
    finally:
        timer.cancel()
    # A kill shows as a negative code; report it as a timeout.
    return TIMED_OUT_EXIT_CODE if code < 0 else code


def _utc(raw: str) -> datetime:
    value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sibyl.ops.minibench_loop",
        description="Poll MiniBench on a fixed interval and forecast new questions one at a time.",
    )
    parser.add_argument("--interval-minutes", type=float, default=15)
    parser.add_argument("--until", type=_utc, default=None, help="Stop starting runs at this UTC time")
    parser.add_argument("--max-questions", type=int, default=5, help="Per-run question cap (soonest close first)")
    parser.add_argument(
        "--grant-budget-usd",
        type=Decimal,
        default=Decimal("20"),
        # About $1.30 a question, plus up to about $10 held by calls in flight. A run that
        # exhausts its grant budget sends the rest of its calls to the personal key.
        help="Grant budget for each run; cover --max-questions with room for in-flight holds",
    )
    parser.add_argument("--archive-dir", type=Path, default=Path("backtests/minibench_runs"))
    parser.add_argument("--stop-below-usd", type=Decimal, default=Decimal("20"), help="Grant headroom to keep")
    parser.add_argument("--max-personal-usd", type=Decimal, default=Decimal("2"), help="Personal-key spend cap")
    parser.add_argument(
        "--round-budget-usd", type=Decimal, default=None, help="Most of the grant this loop may use, in total"
    )
    parser.add_argument("--run-timeout-minutes", type=float, default=180, help="Stop a run that takes longer")
    parser.add_argument(
        "--mode",
        choices=["minibench", "test"],
        default="minibench",
        help="test forecasts the unscored bot-testing area: a rehearsal of the same path",
    )
    parser.add_argument("--once", action="store_true", help="Make one pull and stop")
    parser.add_argument(
        "--tier",
        choices=["prod", "free"],
        default="prod",
        help="free runs the free roster with no grant (only with --mode test): a rehearsal that costs nothing",
    )
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--loop-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    from sibyl.platform.funding import DEFAULT_SPEND_LEDGER, SPEND_LEDGER_ENV  # noqa: PLC0415

    args = _parser().parse_args(argv)
    ledger_path = args.ledger or Path(os.getenv(SPEND_LEDGER_ENV, "").strip() or DEFAULT_SPEND_LEDGER)
    # A missing ledger would be created empty, hiding everything the grant already spent.
    if not ledger_path.is_file():
        print(f"ERROR: no spend ledger at {ledger_path}; run from the checkout that holds it or pass --ledger")
        return REFUSED_EXIT_CODE
    if Path(".env.local").exists():
        print("ERROR: .env.local overrides the workflow's settings; move it aside for the loop")
        return REFUSED_EXIT_CODE
    from sibyl.config.env import load_environment  # noqa: PLC0415

    load_environment()
    if args.tier == "free" and args.mode != "test":
        print("ERROR: the free tier cannot forecast a scored tournament; use --mode test")
        return REFUSED_EXIT_CODE
    missing = missing_secrets(os.environ)
    if args.tier == "free":
        missing = [key for key in missing if key != GRANT_KEY_ENV]
    elif GRANT_KEY_ENV in missing:
        print(f"ERROR: {GRANT_KEY_ENV} is not set; the loop runs on the grant")
        return REFUSED_EXIT_CODE
    if missing:
        # The run skips a provider whose credential is absent and still publishes.
        print(f"WARNING: running without credentials the MiniBench workflow has: {', '.join(missing)}")
    loop_id = args.loop_id or f"minibench-loop-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%MZ')}"
    config = LoopConfig(
        loop_id=loop_id,
        interval=timedelta(minutes=args.interval_minutes),
        until=args.until,
        max_questions=args.max_questions,
        grant_budget_usd=args.grant_budget_usd,
        archive_dir=args.archive_dir,
        stop_below_usd=args.stop_below_usd,
        max_personal_usd=args.max_personal_usd,
        run_timeout=timedelta(minutes=args.run_timeout_minutes),
        round_budget_usd=args.round_budget_usd,
        mode=args.mode,
        once=args.once,
        tier=args.tier,
    )
    env = {**os.environ, **workflow_settings(), SPEND_LEDGER_ENV: str(ledger_path.resolve()), "PYTHONUNBUFFERED": "1"}
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    with open(args.archive_dir / f"{loop_id}.log", "a") as log:

        def say(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        reason = run_loop(
            config,
            Ledger(ledger_path),
            run=lambda command, run_env: _stream(command, run_env, log, config.run_timeout),
            env=env,
            say=say,
        )
        say(f"[loop] stopped: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
