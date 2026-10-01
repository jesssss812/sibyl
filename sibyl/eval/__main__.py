"""``python -m sibyl.eval <command>`` — one door onto the offline eval tools.

Before this, the five tools were five unrelated invocations: ``python -m
sibyl.eval.backtest``, ``python -m sibyl.eval.ablation.cli``, ``python -m
sibyl.eval.ensemble``, ``python -m sibyl.eval.performance``, ``python -m
sibyl.aggregation.calibration.fit_platt_cli``. Nothing tied them together, so there was
no place to discover that the others existed.

**The dispatcher does not re-parse the sub-tool's flags.** It takes the first argument as
the command and hands the REST to that tool untouched, by rewriting ``sys.argv`` the way
the tool already expects to find it. Argparse subparsers would have meant restating every
flag of every tool here and keeping five copies in sync -- and the plan's requirement is
that existing flags are unchanged, which is much easier to guarantee by not touching
them.

Every old invocation still works. ``python -m sibyl.eval.backtest`` is untouched; this is
an additional entry point, not a replacement, so nothing in anyone's shell history breaks.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

# name -> (one-line help, thunk taking the remaining argv and returning an exit code)
_Command = tuple[str, Callable[[list[str]], int]]


def _run_backtest(argv: list[str]) -> int:
    from sibyl.eval.backtest.cli import main

    return _with_argv("backtest", argv, main)


def _run_ablation(argv: list[str]) -> int:
    from sibyl.eval.ablation.cli import main

    # The one tool that already returns an exit code, and takes argv directly.
    return main(argv)


def _run_ensemble(argv: list[str]) -> int:
    from sibyl.eval.ensemble.cli import main

    return _with_argv("ensemble", argv, main)


def _run_performance(argv: list[str]) -> int:
    from sibyl.eval.performance.cli import main

    main(argv)
    return 0


def _run_fit_platt(argv: list[str]) -> int:
    from sibyl.aggregation.calibration.fit_platt_cli import _build_parser, run

    return int(run(_build_parser().parse_args(argv)))


def _run_preflight(argv: list[str]) -> int:
    from sibyl.eval.power import main
    return main(argv)


def _run_compare(argv: list[str]) -> int:
    from sibyl.eval.compare import main

    return main(argv)


def _with_argv(name: str, argv: list[str], fn: Callable[[], None]) -> int:
    """Call a sub-tool that reads ``sys.argv`` itself.

    Restores argv in a finally so a SystemExit from the tool's own argparse -- a bad flag,
    or --help -- does not leave the process with a rewritten argv.
    """
    saved = sys.argv
    sys.argv = [f"python -m sibyl.eval {name}", *argv]
    try:
        fn()
    finally:
        sys.argv = saved
    return 0


COMMANDS: dict[str, _Command] = {
    "backtest": ("Score the bot against resolved questions", _run_backtest),
    "ablation": ("Compare aggregation arms on cached forecasts", _run_ablation),
    "ensemble": ("Forecaster correlation and ensemble simulation", _run_ensemble),
    "performance": ("Calibration against live resolutions", _run_performance),
    "fit-platt": ("Fit Platt calibration parameters", _run_fit_platt),
    "preflight": ("Prospective per-type power check (offline)", _run_preflight),
    "compare": ("Paired comparison of two backtest data files", _run_compare),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [f"  {name:<{width}}  {help_text}" for name, (help_text, _fn) in sorted(COMMANDS.items())]
    return (
        "usage: python -m sibyl.eval <command> [args...]\n\n"
        "commands:\n" + "\n".join(lines) + "\n\n"
        "Flags belong to the command, not to this dispatcher:\n"
        "  python -m sibyl.eval backtest --help\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(_usage(), end="")
        return 0
    name, rest = args[0], args[1:]
    entry = COMMANDS.get(name)
    if entry is None:
        print(f"unknown command {name!r}\n\n{_usage()}", end="", file=sys.stderr)
        return 2
    return entry[1](rest)


if __name__ == "__main__":
    raise SystemExit(main())
