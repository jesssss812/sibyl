---
name: sibyl-review
description: Independently review Sibyl code changes for correctness, design simplicity, maintainability, modularity, and stale comments or artifacts. Use for branch, diff, or working-tree reviews; do not use to implement fixes.
---

# Sibyl review

Review the scope named by the user. If none is given, review branch changes since
`clean-fork` and any staged or unstaged changes. Record the exact refs or files
reviewed. Treat an untracked file as in scope only when the user names it or it
clearly belongs to the requested change.

Use an independent reviewer when subagents are available. Start it with no prior
conversation history, and pass only the repository path, review scope, `AGENTS.md`,
the criteria below, and read-only constraints. Do not send earlier conclusions,
implementation rationale, or suspected defects. If a subagent is unavailable,
perform the same review directly and say so.

Read the diff and enough surrounding code, callers, tests, and docs to verify a
finding. Prioritize: (1) correctness, (2) design simplicity, (3) ease of maintenance
and extension, (4) modularity and cleanliness, (5) outdated comments or artifacts.
Look for contract mismatches and edge cases before style. Run only offline tests
that help resolve a specific uncertainty. Never invoke live or paid APIs, publish,
dispatch workflows, or modify files during the review.

Report actionable findings first, ordered by severity. Give each finding an exact
file and line, a concrete failure scenario, and why the existing tests do not rule
it out. Separate confirmed defects from design suggestions. Include the reviewed
scope and any material coverage limits. If no actionable issue is found, say so
plainly without inventing suggestions.
