"""The suite must not touch the repository it is run from, even inside a git hook.

A hook exports GIT_DIR for the repository being pushed. This runs the hook tests in a
child pytest with GIT_DIR pointed at a decoy repository, and checks the decoy is
untouched afterwards.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ISOLATED = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=decoy@example.invalid", "-c", "user.name=Decoy", "-c", "core.hooksPath=", *args],
        cwd=repo,
        env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")} | _ISOLATED,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_hook_tests_leave_the_repository_named_by_git_dir_alone(tmp_path: Path) -> None:
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    _git(decoy, "init", "-q", "-b", "work", ".")
    (decoy / "keep.txt").write_text("keep\n")
    _git(decoy, "add", ".")
    _git(decoy, "commit", "-q", "-m", "decoy")
    before = (_git(decoy, "rev-parse", "HEAD"), (decoy / ".git" / "config").read_text())

    hook_tests = sorted(str(path) for path in (_REPO_ROOT / "tests" / "unit").glob("test_*hook*.py"))
    assert hook_tests, "no hook tests found to run"
    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *hook_tests],
        cwd=_REPO_ROOT,
        # Only GIT_DIR, as a hook sets it: git then treats the test's cwd as the work tree.
        env={**os.environ, "GIT_DIR": str(decoy / ".git")},
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stdout[-2000:] + child.stderr[-2000:]
    after = (_git(decoy, "rev-parse", "HEAD"), (decoy / ".git" / "config").read_text())
    assert after == before, "a hook test committed to, or reconfigured, the repository GIT_DIR named"
