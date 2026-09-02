"""The eval dispatcher: every command reaches its tool, and the old doors still open.

The dispatcher's whole job is to add a front door without changing any room behind it,
so the tests are mostly about what it does NOT do -- it must not consume a sub-tool's
flags, must not leave sys.argv rewritten, and must not become the only way in.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from sibyl.eval.__main__ import COMMANDS, main


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "sibyl.eval", *args], capture_output=True, text=True, timeout=180)


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
    def test_every_command_reaches_a_real_parser(self, name: str) -> None:
        """--help must come from the TOOL, not from the dispatcher.

        Run out-of-process: --help exits, and several of these tools configure logging
        or touch module state on import.
        """
        result = _run([name, "--help"])
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout

    @pytest.mark.parametrize("name", sorted(COMMANDS))
    def test_the_dispatcher_does_not_claim_the_tools_flags(self, name: str) -> None:
        # If the dispatcher parsed flags itself, an unknown one would be ITS error.
        result = _run([name, "--definitely-not-a-real-flag"])
        assert result.returncode != 0
        assert "unknown command" not in result.stderr


class TestOldEntryPointsStillWork:
    """This is an additional door, not a replacement. Shell history must keep working."""

    @pytest.mark.parametrize(
        "module",
        [
            "sibyl.eval.backtest",
            "sibyl.eval.ensemble",
            "sibyl.eval.ablation.cli",
            "sibyl.aggregation.calibration.fit_platt_cli",
        ],
    )
    def test_direct_module_invocation_still_parses(self, module: str) -> None:
        result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
