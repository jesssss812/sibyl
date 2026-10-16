#!/usr/bin/env bash
#
# Remember which source tree last passed the full suite, so the pre-push hook does not
# run it again on identical files.
#
# WHY: an agent runs `make test` before each commit, then the pre-push hook ran the same
# ~7-minute suite on the same files. A tree id names the exact content of every tracked
# file, so a match means those files already passed. Only a CLEAN checkout gets an id:
# tests read the working tree, and an uncommitted or untracked file would make the id
# describe files other than the ones tested. Ignored files (.env, backtests/) do not
# count, as CI never sees them either.
#
#   tested_tree.sh current        print the tree id when the checkout is clean, else nothing
#   tested_tree.sh record <tree>  remember <tree> as passed, if it is still the current tree
#   tested_tree.sh pre-push       skip when the current tree already passed, else run the suite
#
# The stamp lives under the git dir, per worktree, and is never committed.

set -euo pipefail

stamp="$(git rev-parse --git-path sibyl-tested-tree)"

current() {
  if [[ -z "$(git status --porcelain --untracked-files=normal)" ]]; then
    git rev-parse 'HEAD^{tree}'
  fi
}

record() {
  local tree="${1:-}"
  if [[ -n "${tree}" && "${tree}" == "$(current)" ]]; then
    printf '%s\n' "${tree}" >"${stamp}"
  fi
}

case "${1:-}" in
  current)
    current
    ;;
  record)
    record "${2:-}"
    ;;
  pre-push)
    tree="$(current)"
    if [[ -n "${tree}" && -f "${stamp}" && "$(cat "${stamp}")" == "${tree}" ]]; then
      echo "pytest: tree ${tree:0:12} already passed the full suite; skipping"
      exit 0
    fi
    uv run --frozen pytest --cov=sibyl
    record "${tree}"
    ;;
  *)
    echo "usage: $0 current | record <tree> | pre-push" >&2
    exit 2
    ;;
esac
