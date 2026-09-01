#!/usr/bin/env bash
#
# pre-commit hook: refuse a commit whose STAGED DIFF contains something shaped like a
# live credential.
#
# WHY: .gitignore protects the paths we thought of. It does not protect a key pasted
# into a source file, a notebook, a test fixture, a doc, or a `.env.template` that
# someone filled in "just to try it". The known-bad sequence is four innocuous steps —
# key lands in a tracked template, file gets renamed, the staged copy survives the
# rename, and `git checkout --` resurrects it — after which the key is on a remote and
# has to be rotated. That is a real incident from a prior project, not a hypothetical.
#
# Scope is deliberately the STAGED DIFF, not the worktree: this asks "is a secret about
# to enter history", which is the only question that matters at commit time. Untracked
# and ignored files never appear in `git diff --cached`, so a correctly-ignored `.env`
# costs nothing here.
#
# Only ADDED lines (^+) are scanned. A removed line matching a pattern is a key being
# taken OUT — the remediation commit — which must never be blocked.
#
# TWO DETECTORS, because one is not enough:
#
#   1. VENDOR PREFIX — a value whose own shape identifies it (ank_, sk-, AIza...).
#      Precise enough to match anywhere on the line, including bare.
#
#   2. ASSIGNMENT CONTEXT — a credential-ish NAME assigned a long opaque VALUE.
#      This is what catches the credentials with no distinctive shape, which for this
#      repo is most of them: METACULUS_TOKEN is opaque and ASKNEWS_CLIENT_ID is a plain
#      UUID. Matching those by value shape alone is not an option — this repo's tracked
#      fixtures contain bare UUIDs and its docs are full of 40-hex commit SHAs, so a
#      bare-shape rule would fire constantly. Requiring the assignment is what separates
#      "a UUID" from "a UUID being used as a credential".
#
# FALSE POSITIVES ARE A SECURITY BUG HERE, not just an annoyance: a hook that cries wolf
# trains people to reflexively pass --no-verify, which converts it into nothing. Hence
# the placeholder allowlist below, and hence `.env.template` — which is TRACKED and ships
# `METACULUS_TOKEN=1234567890` — must pass cleanly.
#
# Tests: tests/test_secret_hook.py drives this as a black box over throwaway git repos.

set -euo pipefail

# --- Detector 1: values whose own prefix identifies them -------------------------
#   ank_      AskNews secret
#   sk-/sk_   OpenAI / Anthropic / OpenRouter (sk-, sk-ant-, sk-or-) and Stripe
#             (sk_live_, sk_test_). Both separators, one pattern — routing Stripe
#             here avoids adding a bare `KEY` to cred_name, which would match
#             SORT_KEY / CACHE_KEY / PRIMARY_KEY and reintroduce false positives.
#   AIza      Google API keys
#   ghp_/gho_ GitHub tokens
#   xoxb-     Slack bot tokens
vendor_patterns=(
  '\bank_[A-Za-z0-9]{20,}'
  '\bsk[-_][A-Za-z0-9_-]{20,}'
  '\bAIza[A-Za-z0-9_-]{30,}'
  '\bgh[pos]_[A-Za-z0-9]{20,}'
  '\bxoxb-[A-Za-z0-9-]{20,}'
)

# --- Detector 2: credential-ish NAME = long opaque VALUE -------------------------
# Case-insensitive, accepts `=` or `:` (yaml/CI configs assign with a colon). The value
# stops at whitespace or a quote, so `SECRET = os.getenv("X")` yields a short `os.getenv(`
# rather than swallowing the whole expression.
# `[A-Z0-9_]*` after the name is load-bearing: without it the name had to be
# IMMEDIATELY followed by the separator, so every suffixed variant missed — including
# SECRET_KEY (the Django/Flask default) and AWS_ACCESS_KEY_ID.
# The optional closing quote before the separator catches JSON/dict literals, where a
# `"` sits between the key and the colon.
cred_name='(CLIENT_SECRET|CLIENT_ID|ACCESS_KEY|PRIVATE_KEY|SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY)[A-Z0-9_]*'
assignment_re="${cred_name}[\"']?[[:space:]]*[:=][[:space:]]*[\"']?[^\"'[:space:],)}]{12,}"

# Values that are obviously not live credentials. Anchored at the start of the value.
#   1234567890  the placeholder .env.template actually ships
#   your_/<..>/${..}/os.*  documentation, templates, CI interpolation, env indirection
placeholder_re='^(1234567890|your_|<|\$\{|os\.|changeme|xxx+|example|dummy|fake|placeholder|replace|todo|none|null)'

# A credential is an opaque blob; a variable name is not. These three rules were derived
# by auditing the detector against this repo's own tracked files, where it fired on
# `api_key=primary_api_key` (kwarg pass-through), `api_key=get_openrouter_api_key(...)`
# (call), and `f"client_id={client_id[:8]}"` (interpolation) — all code, no secrets.
#   identifier_re  a snake_case name is a variable being passed, not a literal
#   codeish_re     brackets/parens mean an expression, not a value
#   entropy_re     a real key mixes letters and digits; `personal-key` does not
#
# identifier_re exempts a variable NAME, and must be narrow enough not to exempt a
# token. Two wrong versions, both live false negatives:
#   ^[A-Za-z_][A-Za-z0-9_]*$        exempted every pure-alphanumeric token (a real
#                                   METACULUS_TOKEN read as "just an identifier")
#   ^[A-Za-z][A-Za-z0-9]*_[...]$    fixed that, but exempted every `prefix_alnum` token:
#                                   npm_, github_pat_, sk_live_, hf_, dckr_pat_, r8_
# A source identifier is lower_snake_case words. A token has a long unbroken alphanumeric
# run, so requiring every underscore-separated segment to be a short, non-digit-bearing
# word separates the two.
identifier_re='^[a-z][a-z]*(_[a-z][a-z]*)+$'
# NOTE the bracket order. The natural-looking `[(){}\[\]]` does NOT reliably match here —
# A/B-verified against tests/test_secret_hook.py, where it let `f"client_id={client_id[:8]}"`
# through while this form catches it. Backslash escaping inside a POSIX bracket expression
# is not portable, so use the portable convention instead: put `]` first (where it is
# literal) and never backslash-escape inside the brackets.
codeish_re='[][(){}]'
entropy_re='[0-9]'
# Minimum length below which a value is not worth flagging. 12 keeps `API_KEY=abc123`
# and the 10-char `1234567890` template placeholder quiet without weakening real cover:
# every credential shape this repo uses is far longer.
min_len=12

# An explicit per-line opt-out, the same convention detect-secrets and gitleaks use.
# Needed because this hook's OWN test file is necessarily full of synthetic credentials,
# and a guard that cannot be committed without --no-verify teaches exactly the habit it
# exists to prevent. Per-LINE and explicit, so it is greppable in review — deliberately
# not a path exclusion, which would silently cover a whole file forever.
allow_marker='pragma: allowlist secret'

# git and the greps are captured SEPARATELY. Folding them into one pipeline with a
# trailing `|| true` swallowed a git failure (not a repo, corrupt index) exactly like
# grep's benign no-match exit 1, so a broken invocation produced an empty diff and the
# hook exited 0. A security guard must fail closed.
if ! raw_diff="$(git diff --cached --no-color -U0 2>&1)"; then
  echo "Commit REFUSED: 'git diff --cached' failed, so the staged diff could not be" >&2
  echo "scanned for credentials. Refusing rather than assuming it is clean." >&2
  echo "${raw_diff}" >&2
  exit 1
fi

filter_added() {
  grep -E '^\+' | grep -Ev '^\+\+\+' | grep -Fv "${allow_marker}" || true
}

staged_added="$(printf '%s\n' "${raw_diff}" | filter_added)"

# The heuristic detector skips tests/data/, which holds captured third-party API payloads
# whose ids are genuinely UUIDs assigned to `token`/`id` keys. Running the heuristic there
# fires on every routine fixture refresh, and a hook that blocks routine work gets
# bypassed. The vendor-prefix detector above still scans these files, so a real key hidden
# in a fixture is still caught.
if ! raw_heuristic_diff="$(git diff --cached --no-color -U0 -- . ':(exclude)tests/data/' 2>&1)"; then
  echo "Commit REFUSED: 'git diff --cached' (heuristic scope) failed." >&2
  exit 1
fi
staged_heuristic="$(printf '%s\n' "${raw_heuristic_diff}" | filter_added)"

if [[ -z "${staged_added}" ]]; then
  exit 0
fi

hits=""

record_hit() {
  # Never echo the secret itself: a blocked commit must not put the key into a terminal
  # scrollback, a CI log, or an agent transcript. Show enough to locate it, no more.
  local found="$1"
  hits="${hits}  - ${found:0:8}… (${#found} chars)"$'\n'
}

for pattern in "${vendor_patterns[@]}"; do
  while IFS= read -r found; do
    [[ -n "${found}" ]] && record_hit "${found}"
  done < <(printf '%s\n' "${staged_added}" | grep -Eo "${pattern}" || true)
done

while IFS= read -r match; do
  [[ -z "${match}" ]] && continue
  # Strip NAME and the separator, then any opening quote, leaving the bare value.
  value="${match#*[:=]}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value#[\"\']}"
  (( ${#value} < min_len )) && continue
  printf '%s' "${value}" | grep -Eiq "${placeholder_re}" && continue
  printf '%s' "${value}" | grep -Eq "${identifier_re}" && continue
  printf '%s' "${value}" | grep -Eq "${codeish_re}" && continue
  printf '%s' "${value}" | grep -Eq "${entropy_re}" || continue
  record_hit "${value}"
done < <(printf '%s\n' "${staged_heuristic}" | grep -Eio "${assignment_re}" || true)

if [[ -z "${hits}" ]]; then
  exit 0
fi

cat >&2 <<EOF
Commit REFUSED: the staged diff contains something shaped like a live credential.

Matched (truncated — the full value is deliberately not printed):
${hits}
Secrets belong in .env (gitignored), never in a tracked file. Templates stay BLANK.

To fix:
    git restore --staged <file>     # unstage it
    # move the value into .env, leave the tracked file with an empty placeholder

If this is genuinely not a credential (a fixture, a documented example), mark that
one line. It stays visible in review, and it leaves every other pre-commit hook
(branch guard, ruff) doing its job:

    SOME_VALUE = "..."  # pragma: allowlist secret
EOF

exit 1
