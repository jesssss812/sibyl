"""The eval dispatcher: every command reaches its tool, and the old doors still open.

The dispatcher's whole job is to add a front door without changing any room behind it,
so the tests are mostly about what it does NOT do -- it must not consume a sub-tool's
flags, must not leave sys.argv rewritten, and must not become the only way in.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

import pytest

from sibyl.eval.__main__ import COMMANDS, main

OLD_ENTRY_POINTS = [
    "sibyl.eval.backtest",
    "sibyl.eval.ensemble",
    "sibyl.eval.ablation.cli",
    "sibyl.aggregation.calibration.fit_platt_cli",
]
_BAD_FLAG = "--definitely-not-a-real-flag"

# Runs every case in ONE child process. The tools configure logging and touch module
# state on import, so they stay out of the test process; but a process per case paid the
# ~9s forecasting-tools import 24 times. Each case runs the way `python -m` would: the
# dispatcher's main() for `sibyl.eval <command>`, runpy for an old module entry point.
_DRIVER = """
import contextlib, io, json, runpy, sys
from sibyl.eval.__main__ import main

def run(fn):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = fn()
        except SystemExit as exc:
            code = exc.code
    code = 0 if code is None else code if isinstance(code, int) else 1
    return {"returncode": code, "stdout": out.getvalue(), "stderr": err.getvalue()}

def as_module(module, args):
    def fn():
        saved = sys.argv
        sys.argv = [module, *args]
        try:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        finally:
            sys.argv = saved
    return fn

cases = json.loads(sys.argv[1])
results = {}
for key, kind, target, args in cases:
    fn = (lambda t=target, a=args: main([t, *a])) if kind == "dispatch" else as_module(target, args)
    results[key] = run(fn)
print("RESULTS:" + json.dumps(results))
"""


class _Result(TypedDict):
    returncode: int
    stdout: str
    stderr: str


@pytest.fixture(scope="module")
def results() -> dict[str, _Result]:
    cases = [
        *([f"help:{name}", "dispatch", name, ["--help"]] for name in COMMANDS),
        *([f"badflag:{name}", "dispatch", name, [_BAD_FLAG]] for name in COMMANDS),
        *([f"module:{module}", "module", module, ["--help"]] for module in OLD_ENTRY_POINTS),
    ]
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, json.dumps(cases)], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    line = next(line for line in reversed(proc.stdout.splitlines()) if line.startswith("RESULTS:"))
    return json.loads(line.removeprefix("RESULTS:"))


class TestDispatch:
    def test_bare_invocation_lists_the_commands(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == 0
        out = capsys.readouterr().out
        for name in COMMANDS:
            assert name in out

    def test_unknown_command_is_an_error_not_a_default(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Silently falling through to a default would run the WRONG eval tool.
        assert main(["bogus"]) == 2
        assert "unknown command" in capsys.readouterr().err

    def test_argv_is_restored_even_when_a_tool_exits(self) -> None:
        # A sub-tool's --help raises SystemExit from inside argparse. If the dispatcher
        # leaked its rewritten argv, every later call in the same process would inherit it.
        before = list(sys.argv)
        with pytest.raises(SystemExit):
            main(["backtest", "--help"])
        assert sys.argv == before

    @pytest.mark.parametrize("name", sorted(COMMANDS))
    def test_every_command_reaches_a_real_parser(self, name: str, results: dict[str, _Result]) -> None:
        """--help must come from the TOOL, not from the dispatcher.

        Run out-of-process (see ``_DRIVER``): --help exits, and several of these tools
        configure logging or touch module state on import.
        """
        result = results[f"help:{name}"]
        assert result["returncode"] == 0, result["stderr"]
        assert "usage:" in result["stdout"]
        assert "commands:" not in result["stdout"], "--help printed the dispatcher's usage, not the tool's"

    @pytest.mark.parametrize("name", sorted(COMMANDS))
    def test_the_dispatcher_does_not_claim_the_tools_flags(self, name: str, results: dict[str, _Result]) -> None:
        # If the dispatcher parsed flags itself, an unknown one would be ITS error.
        result = results[f"badflag:{name}"]
        assert result["returncode"] != 0
        assert "unknown command" not in result["stderr"]


class TestOldEntryPointsStillWork:
    """This is an additional door, not a replacement. Shell history must keep working."""

    @pytest.mark.parametrize("module", OLD_ENTRY_POINTS)
    def test_direct_module_invocation_still_parses(self, module: str, results: dict[str, _Result]) -> None:
        result = results[f"module:{module}"]
        assert result["returncode"] == 0, result["stderr"]
        assert "usage:" in result["stdout"]

    def test_the_package_entry_point_runs_as_a_real_process(self) -> None:
        """One true `python -m sibyl.eval`, so the driver above cannot hide a broken door."""
        result = subprocess.run([sys.executable, "-m", "sibyl.eval"], capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr
        assert "commands:" in result.stdout


class TestAgentsMdListsEveryCommand:
    """AGENTS.md is the first thing an agent reads; a command missing there goes unused."""

    def test_entry_point_line_names_exactly_the_dispatched_commands(self) -> None:
        agents_md = (Path(__file__).parents[3] / "AGENTS.md").read_text()
        match = re.search(r"python -m sibyl\.eval <([^>]+)>", agents_md)
        assert match, "AGENTS.md no longer shows the `python -m sibyl.eval <command>` line"
        assert set(match.group(1).split("|")) == set(COMMANDS)
