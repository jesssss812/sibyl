"""The pre-push type check runs the command CI's ``lint`` job runs.

The hook exists so a type error stops the push instead of turning the PR red. If its
command drifts from CI's, a push can pass locally and still fail there.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_pre_push_type_check_matches_ci() -> None:
    config = yaml.safe_load((_REPO_ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"] if hook.get("id") == "basedpyright"]
    assert len(hooks) == 1
    assert hooks[0]["stages"] == ["pre-push"]

    ci = yaml.safe_load((_REPO_ROOT / ".github" / "workflows" / "ci.yaml").read_text())
    ci_commands = [step.get("run") for step in ci["jobs"]["lint"]["steps"]]
    assert hooks[0]["entry"] in ci_commands
