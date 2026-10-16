"""Guards on the pre-push hook that skips a suite run the same files already passed.

``make test`` records the tree id of a clean checkout that passed; the pre-push hook
skips the suite when the checkout is still that tree. The failure that matters is a
skip that should not happen: pushing files that never passed. So most of these run the
script for real in a throwaway repo and check it refuses to skip on a dirty checkout, a
changed tree, or a failed run. A stand-in ``uv`` on PATH records whether the suite ran
and sets its exit code, so the real suite never starts. The repos are isolated from the
operator's git config as in ``test_no_commit_to_main_hook.py``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_RELPATH = "scripts/hooks/tested_tree.sh"
_SCRIPT = _REPO_ROOT / _SCRIPT_RELPATH
_ISOLATED = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=test@example.invalid", "-c", "user.name=Test", "-c", "core.hooksPath=", *args],
        cwd=repo,
        env={**os.environ, **_ISOLATED},
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "feature", ".")
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "seed.txt").write_text("seed\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


@pytest.fixture
def fake_uv(tmp_path: Path) -> Path:
    """A ``uv`` that logs each call and exits with the code in ``uv_exit``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{tmp_path}/uv_calls"\nexit "$(cat "{tmp_path}/uv_exit")"\n')
    uv.chmod(0o755)
    (tmp_path / "uv_exit").write_text("0")
    return tmp_path


def _run(repo: Path, fake_uv: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **_ISOLATED, "PATH": f"{fake_uv / 'bin'}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run([str(_SCRIPT), *args], cwd=repo, env=env, capture_output=True, text=True, check=False)


def _suite_runs(fake_uv: Path) -> int:
    calls = fake_uv / "uv_calls"
    return len(calls.read_text().splitlines()) if calls.exists() else 0


class TestCurrent:
    def test_a_clean_checkout_has_its_tree_id(self, repo: Path, fake_uv: Path) -> None:
        assert _run(repo, fake_uv, "current").stdout.strip() == _git(repo, "rev-parse", "HEAD^{tree}").strip()

    def test_an_ignored_file_does_not_make_it_dirty(self, repo: Path, fake_uv: Path) -> None:
        (repo / "ignored.txt").write_text("local only\n")
        assert _run(repo, fake_uv, "current").stdout.strip()

    @pytest.mark.parametrize("change", ["modified", "untracked", "staged"])
    def test_a_dirty_checkout_has_none(self, repo: Path, fake_uv: Path, change: str) -> None:
        if change == "modified":
            (repo / "seed.txt").write_text("edited\n")
        elif change == "untracked":
            (repo / "new.txt").write_text("new\n")
        else:
            (repo / "new.txt").write_text("new\n")
            _git(repo, "add", "new.txt")
        assert _run(repo, fake_uv, "current").stdout.strip() == ""


class TestPrePush:
    def test_runs_the_suite_and_then_skips_the_same_tree(self, repo: Path, fake_uv: Path) -> None:
        assert _run(repo, fake_uv, "pre-push").returncode == 0
        assert _suite_runs(fake_uv) == 1
        second = _run(repo, fake_uv, "pre-push")
        assert second.returncode == 0
        assert "skipping" in second.stdout
        assert _suite_runs(fake_uv) == 1

    def test_a_recorded_pass_skips_the_suite(self, repo: Path, fake_uv: Path) -> None:
        tree = _run(repo, fake_uv, "current").stdout.strip()
        _run(repo, fake_uv, "record", tree)
        assert _run(repo, fake_uv, "pre-push").returncode == 0
        assert _suite_runs(fake_uv) == 0

    def test_a_new_commit_runs_the_suite_again(self, repo: Path, fake_uv: Path) -> None:
        _run(repo, fake_uv, "pre-push")
        (repo / "seed.txt").write_text("changed\n")
        _git(repo, "commit", "-q", "-am", "change")
        _run(repo, fake_uv, "pre-push")
        assert _suite_runs(fake_uv) == 2

    def test_a_dirty_checkout_always_runs_the_suite(self, repo: Path, fake_uv: Path) -> None:
        _run(repo, fake_uv, "pre-push")
        (repo / "seed.txt").write_text("uncommitted\n")
        _run(repo, fake_uv, "pre-push")
        _run(repo, fake_uv, "pre-push")
        assert _suite_runs(fake_uv) == 3

    def test_a_failed_run_blocks_the_push_and_is_not_remembered(self, repo: Path, fake_uv: Path) -> None:
        (fake_uv / "uv_exit").write_text("1")
        assert _run(repo, fake_uv, "pre-push").returncode != 0
        (fake_uv / "uv_exit").write_text("0")
        _run(repo, fake_uv, "pre-push")
        assert _suite_runs(fake_uv) == 2

    def test_runs_the_same_command_as_ci(self, repo: Path, fake_uv: Path) -> None:
        _run(repo, fake_uv, "pre-push")
        assert (fake_uv / "uv_calls").read_text().strip() == "run --frozen pytest --cov=sibyl"


class TestRecord:
    def test_a_tree_that_changed_during_the_run_is_not_recorded(self, repo: Path, fake_uv: Path) -> None:
        tree = _run(repo, fake_uv, "current").stdout.strip()
        (repo / "seed.txt").write_text("edited mid-run\n")
        _git(repo, "commit", "-q", "-am", "mid-run")
        _run(repo, fake_uv, "record", tree)
        _run(repo, fake_uv, "pre-push")
        assert _suite_runs(fake_uv) == 1


class TestWiring:
    def test_the_pre_push_hook_calls_the_script(self) -> None:
        config = yaml.safe_load((_REPO_ROOT / ".pre-commit-config.yaml").read_text())
        hooks = [hook for repo in config["repos"] for hook in repo["hooks"] if hook.get("id") == "pytest-full-suite"]
        assert len(hooks) == 1
        assert hooks[0]["entry"] == f"{_SCRIPT_RELPATH} pre-push"
        assert hooks[0]["stages"] == ["pre-push"]
        assert hooks[0]["language"] == "script"

    def test_git_records_the_executable_bit(self) -> None:
        recorded = subprocess.run(
            ["git", "ls-files", "-s", "--", _SCRIPT_RELPATH], cwd=_REPO_ROOT, capture_output=True, text=True
        ).stdout
        assert recorded.startswith("100755 "), f"git must record mode 100755 for {_SCRIPT_RELPATH}"

    def test_make_test_records_a_pass(self) -> None:
        recipe = (_REPO_ROOT / "Makefile").read_text().split("\ntest:\n", 1)[1].split("\n\n", 1)[0]
        assert f"{_SCRIPT_RELPATH} current" in recipe
        assert f"&& {_SCRIPT_RELPATH} record" in recipe
