"""Tests for scripts/hooks/no_staged_secrets.sh.

The hook is the structural half of "secrets never leave local" — .gitignore covers the
paths we thought of, this covers a credential pasted into a tracked file.

It is driven here as a black box against real throwaway git repos, because what matters
is its behaviour on an actual staged diff, not its internals. Each test builds a repo in
tmp_path, stages content, and asserts the exit code.

Two failure directions, weighted differently:
  * a FALSE NEGATIVE leaks a credential  -> the thing the hook exists to prevent
  * a FALSE POSITIVE blocks a good commit -> annoying, and it trains people to reflexively
    pass --no-verify, which quietly converts the hook into nothing

So the false-positive cases below are not politeness; they are what keeps the hook
trusted enough to stay enabled. `.env.template` in particular is tracked and ships
`METACULUS_TOKEN=1234567890`, so a hook that fires on it would break every commit that
touches the template.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / "scripts" / "hooks" / "no_staged_secrets.sh"

# --- Synthetic credentials -------------------------------------------------------
#
# Every fake credential below is ASSEMBLED at import time rather than written as a
# literal. Two reasons, both learned the hard way:
#
#   1. A literal would trip this very hook, making the file un-committable without
#      --no-verify — the exact habit the hook exists to discourage.
#   2. The obvious workaround (a trailing `# pragma: allowlist secret` on each line)
#      does not survive `ruff format`, which reflows the long call lines and leaves the
#      marker on the closing paren while the literal moves to a wrapped line above.
#
# Concatenation sidesteps both: no regex can match a value that never appears whole in
# the source. The allow-marker is still supported by the hook and tested below; it is
# just the wrong tool for a formatter-managed file.
FAKE_ANK = "ank_" + "A" * 28
FAKE_ANK_ALT = "ank_" + "B" * 28
FAKE_ANK_Z = "ank_" + "Z" * 28
FAKE_SK = "sk-proj-" + "A" * 28
FAKE_AIZA = "AIzaSy" + "A" * 35
FAKE_GHP = "ghp_" + "A" * 30
FAKE_XOXB = "xoxb-" + "1111111111-" + "2222222222-" + "a" * 12
FAKE_UUID = "0" * 8 + "-dead" + "-beef-cafe-" + "0" * 12
FAKE_UUID_ALT = "1" * 8 + "-dead" + "-beef-cafe-" + "1" * 12
FAKE_HEX_TOKEN = "d3adb33f" + "01234567" + "d3adb33f" + "01234567" + "d3adb33f"
FAKE_ALNUM_TOKEN = "a1b2c3d4" + "e5f6a1b2" + "c3d4e5f6" + "a1b2c3d4"
FAKE_MIXED_TOKEN = "Xk92LmQp" + "4RtY7wZa" + "3BnC5vDf8"

BLOCKED, ALLOWED = 1, 0


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t.t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    return tmp_path


def _stage(repo: Path, filename: str, content: str) -> None:
    (repo / filename).write_text(content)
    subprocess.run(["git", "-C", str(repo), "add", filename], check=True)


def _run_hook(repo: Path) -> int:
    return subprocess.run([str(HOOK)], cwd=repo, capture_output=True, text=True).returncode


def _check(tmp_path: Path, filename: str, content: str) -> int:
    repo = _repo(tmp_path)
    _stage(repo, filename, content)
    return _run_hook(repo)


class TestCatchesProjectCredentials:
    """The two credentials this repo actually uses. Both were missed by the first cut."""

    def test_asknews_client_id_uuid(self, tmp_path: Path) -> None:
        assert _check(tmp_path, "cfg.py", f'ASKNEWS_CLIENT_ID = "{FAKE_UUID}"') == BLOCKED

    def test_asknews_secret_prefixed(self, tmp_path: Path) -> None:
        assert _check(tmp_path, "cfg.py", f"ASKNEWS_SECRET={FAKE_ANK}") == BLOCKED

    def test_metaculus_token_opaque(self, tmp_path: Path) -> None:
        assert _check(tmp_path, "a.env", f"METACULUS_TOKEN={FAKE_HEX_TOKEN}") == BLOCKED

    def test_google_api_key(self, tmp_path: Path) -> None:
        assert _check(tmp_path, "a.py", f'GOOGLE_API_KEY = "{FAKE_AIZA}"') == BLOCKED


class TestCatchesCommonVendorKeys:
    @pytest.mark.parametrize(
        "content",
        [
            f"OPENAI_KEY={FAKE_SK}",
            f"token = '{FAKE_GHP}'",
            f"SLACK={FAKE_XOXB}",
        ],
    )
    def test_blocked(self, tmp_path: Path, content: str) -> None:
        assert _check(tmp_path, "a.py", content) == BLOCKED

    def test_yaml_colon_assignment(self, tmp_path: Path) -> None:
        """CI configs assign with ':' rather than '='."""
        assert _check(tmp_path, "ci.yml", f"  client_secret: {FAKE_UUID}") == BLOCKED


class TestDoesNotBlockLegitimateContent:
    def test_env_template_placeholders(self, tmp_path: Path) -> None:
        """The real .env.template is TRACKED and ships these. Firing here breaks the repo."""
        content = "METACULUS_TOKEN=1234567890\nOPENROUTER_API_KEY=1234567890\nASKNEWS_CLIENT_ID=1234567890\n"
        assert _check(tmp_path, ".env.template", content) == ALLOWED

    @pytest.mark.parametrize(
        "content",
        [
            "API_KEY=your_free_key_from_fred.stlouisfed.org",
            "TOKEN=<your-token-here>",
            'SECRET = os.getenv("ASKNEWS_SECRET")',
            'API_KEY = os.environ["OPENROUTER_API_KEY"]',
            "CLIENT_SECRET=${SECRET_FROM_CI}",
            "PASSWORD=changeme",
            "API_KEY=",
            'GOOGLE_API_KEY: str = ""',
        ],
    )
    def test_placeholders_and_indirection_allowed(self, tmp_path: Path, content: str) -> None:
        assert _check(tmp_path, "a.py", content) == ALLOWED

    def test_bare_uuid_in_test_data_allowed(self, tmp_path: Path) -> None:
        """Bare UUIDs appear in this repo's tracked fixtures; only ASSIGNED ones are secrets."""
        assert _check(tmp_path, "fixture.json", '{"id": "f5102a39-82bc-44a6-a47a-75749ab28c74"}') == ALLOWED

    def test_git_sha_in_docs_allowed(self, tmp_path: Path) -> None:
        """40-hex SHAs are everywhere in this repo's docs; a bare-hex rule would be unusable."""
        assert _check(tmp_path, "d.md", f"See commit {FAKE_HEX_TOKEN} for context.") == ALLOWED

    def test_ordinary_code_allowed(self, tmp_path: Path) -> None:
        assert _check(tmp_path, "a.py", "def compute(x):\n    return x * 2\n") == ALLOWED

    def test_short_value_allowed(self, tmp_path: Path) -> None:
        """Too short to be a credential; blocking these would be pure noise."""
        assert _check(tmp_path, "a.py", "API_KEY=abc123") == ALLOWED


class TestDiffSemantics:
    def test_removing_a_secret_is_allowed(self, tmp_path: Path) -> None:
        """Deleting a leaked key must never be blocked — that is the remediation commit."""
        repo = _repo(tmp_path)
        _stage(repo, "a.py", f"KEY={FAKE_ANK}")
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base", "--no-verify"], check=True)
        (repo / "a.py").write_text("KEY=\n")
        subprocess.run(["git", "-C", str(repo), "add", "a.py"], check=True)
        assert _run_hook(repo) == ALLOWED

    def test_unstaged_file_is_ignored(self, tmp_path: Path) -> None:
        """A correctly-ignored .env is never in the staged diff, so it costs nothing here."""
        repo = _repo(tmp_path)
        (repo / ".env").write_text(f"ASKNEWS_SECRET={FAKE_ANK}")
        assert _run_hook(repo) == ALLOWED

    def test_empty_staged_diff_is_allowed(self, tmp_path: Path) -> None:
        assert _run_hook(_repo(tmp_path)) == ALLOWED


class TestDoesNotLeakTheSecretItFound:
    def test_full_value_is_not_printed(self, tmp_path: Path) -> None:
        """A blocked commit must not put the key into scrollback, CI logs, or a transcript."""
        secret = FAKE_ANK_Z
        repo = _repo(tmp_path)
        _stage(repo, "a.py", f"ASKNEWS_SECRET={secret}")
        result = subprocess.run([str(HOOK)], cwd=repo, capture_output=True, text=True)
        assert result.returncode == BLOCKED
        assert secret not in result.stderr + result.stdout
        assert "ank_" in result.stderr, "should still say enough to locate the value"


class TestValueMustNotLookLikeCode:
    """Real credentials are opaque blobs. Code-shaped values are variables, not secrets.

    Every case here was found by auditing the detector against this repo's own tracked
    files — the unit tests above all passed while these silently fired, which is exactly
    the false-positive class that gets a hook disabled.
    """

    @pytest.mark.parametrize(
        "content",
        [
            "llm = build(api_key=primary_api_key)",  # kwarg pass-through of a variable
            "llm = build(api_key=secondary_api_key)",
            "llm = build(api_key=get_openrouter_api_key(model_name))",  # function call
            'logger.info(f"client_id={client_id[:8]}...")',  # f-string fragment
            'build(api_key="personal-key")',  # short non-credential literal
            "client_id = os.getenv(ASKNEWS_CLIENT_ID_ENV)",  # env indirection
        ],
    )
    def test_code_shaped_values_allowed(self, tmp_path: Path, content: str) -> None:
        assert _check(tmp_path, "a.py", content) == ALLOWED

    def test_still_blocks_a_real_looking_value_in_code(self, tmp_path: Path) -> None:
        """The relaxations above must not open a hole for an actual pasted credential."""
        assert _check(tmp_path, "a.py", f'build(api_key="{FAKE_UUID}")') == BLOCKED


class TestFixtureDataCarveOut:
    """tests/data/ holds captured API payloads whose ids are genuinely UUIDs.

    Running the assignment heuristic over them fires on every fixture refresh (verified
    against tests/data/manifold_pseudo_numeric_2026_08_05.json, which contains
    `token=<uuid>` rows). A hook that blocks routine fixture updates gets bypassed, so
    the heuristic detector skips this path — but the vendor-prefix detector does NOT,
    so a real `ank_`/`sk-` key hidden in a fixture is still caught.
    """

    def test_uuid_in_fixture_data_allowed(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path)
        (repo / "tests" / "data").mkdir(parents=True)
        _stage(repo, "tests/data/payload.json", f'{{"token": "{FAKE_UUID_ALT}"}}')
        assert _run_hook(repo) == ALLOWED

    def test_vendor_key_in_fixture_data_still_blocked(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path)
        (repo / "tests" / "data").mkdir(parents=True)
        _stage(repo, "tests/data/payload.json", f'{{"token": "{FAKE_ANK}"}}')
        assert _run_hook(repo) == BLOCKED

    def test_uuid_outside_fixture_data_still_blocked(self, tmp_path: Path) -> None:
        assert _check(tmp_path, "config.py", f'CLIENT_ID = "{FAKE_UUID_ALT}"') == BLOCKED


class TestAllowMarker:
    """The per-line opt-out. Needed so this very file can be committed without --no-verify.

    Deliberately per-LINE and explicit rather than a path exclusion: it shows up in review
    diffs, so adding one is a visible decision rather than a silent, permanent carve-out.
    """

    MARKER = "# pragma: allowlist secret"

    def test_marked_line_is_allowed(self, tmp_path: Path) -> None:
        content = f'SECRET = "{FAKE_ANK}"  {self.MARKER}'
        assert _check(tmp_path, "a.py", content) == ALLOWED

    def test_marker_only_exempts_its_own_line(self, tmp_path: Path) -> None:
        """An unmarked secret elsewhere in the same file is still caught.

        The unmarked value is assembled at runtime rather than written literally: a
        literal here could not carry the marker (that is the point of the test), so it
        would make this very file un-committable under its own guard.
        """
        unmarked = FAKE_ANK_ALT
        content = f'A = "{FAKE_ANK}"  {self.MARKER}\nB = "{unmarked}"\n'
        assert _check(tmp_path, "a.py", content) == BLOCKED

    def test_marker_also_exempts_the_assignment_detector(self, tmp_path: Path) -> None:
        content = f'CLIENT_ID = "{FAKE_UUID}"  {self.MARKER}'
        assert _check(tmp_path, "a.py", content) == ALLOWED

    def test_this_repos_own_hook_test_file_is_committable(self, tmp_path: Path) -> None:
        """Regression: the guard must not block the commit that adds its own tests."""
        repo = _repo(tmp_path)
        (repo / "tests" / "unit").mkdir(parents=True)
        _stage(repo, "tests/unit/test_secret_hook.py", Path(__file__).read_text())
        assert _run_hook(repo) == ALLOWED


class TestOpaqueAlphanumericTokensAreCaught:
    """Regression: a pure-alphanumeric token must not be mistaken for a variable name.

    The first version of identifier_re was `^[A-Za-z_][A-Za-z0-9_]*$`, which matches any
    unbroken alphanumeric run — i.e. the exact shape of most opaque API tokens. It made
    the assignment detector blind to the very credential type it was added for.
    """

    @pytest.mark.parametrize(
        "value",
        [
            FAKE_HEX_TOKEN,  # 40-char hex-ish, letters first
            FAKE_ALNUM_TOKEN,  # 32-char alphanumeric
            FAKE_MIXED_TOKEN,  # mixed-case alphanumeric
        ],
    )
    def test_alphanumeric_token_is_blocked(self, tmp_path: Path, value: str) -> None:
        assert _check(tmp_path, "a.env", f"METACULUS_TOKEN={value}") == BLOCKED

    @pytest.mark.parametrize("value", ["primary_api_key", "secondary_api_key", "some_local_variable"])
    def test_snake_case_variables_still_exempt(self, tmp_path: Path, value: str) -> None:
        assert _check(tmp_path, "a.py", f"build(api_key={value})") == ALLOWED


class TestFalseNegativesFoundInReview:
    """Six credential shapes the first cut missed, found by independent review.

    Three causes, all systematic rather than one-off:
      1. `identifier_re` required an underscore to exempt variable names — which also
         exempted every `prefix_alnum` token (npm_, github_pat_, sk_live_, hf_, r8_).
         The fix for one false negative created a whole class of others.
      2. The credential name had to be IMMEDIATELY followed by `=` or `:`, so any
         suffixed name missed — including `SECRET_KEY`, the Django/Flask default.
      3. JSON puts a closing quote between the key and the colon.
    """

    # Built from short fragments for the same reason as the constants at the top of this
    # file: a whole literal here would trip the hook and make this file un-committable.
    @pytest.mark.parametrize(
        "name,prefix",
        [
            ("NPM_TOKEN", "npm" + "_"),
            ("GITHUB_TOKEN", "github" + "_pat_"),
            ("HUGGINGFACE_TOKEN", "hf" + "_"),
            ("DOCKER_TOKEN", "dckr" + "_pat_"),
            ("REPLICATE_API_TOKEN", "r8" + "_"),
            ("STRIPE_KEY", "sk" + "_live_"),
        ],
    )
    def test_prefix_underscore_tokens_are_caught(self, tmp_path: Path, name: str, prefix: str) -> None:
        assert _check(tmp_path, "a.env", f"{name}={prefix}{FAKE_ALNUM_TOKEN}") == BLOCKED

    @pytest.mark.parametrize(
        "name",
        ["SECRET_KEY", "AWS_ACCESS_KEY_ID", "PRIVATE_KEY_PEM", "APIKEY_V2", "TOKEN_B"],
    )
    def test_suffixed_credential_names_are_caught(self, tmp_path: Path, name: str) -> None:
        assert _check(tmp_path, "a.env", f"{name}={FAKE_HEX_TOKEN}") == BLOCKED

    @pytest.mark.parametrize(
        "content",
        [
            '{"api_key": "%s"}',
            "cfg = {'token': '%s'}",
            '  "client_secret" : "%s",',
        ],
    )
    def test_quoted_keys_are_caught(self, tmp_path: Path, content: str) -> None:
        assert _check(tmp_path, "a.json", content % FAKE_HEX_TOKEN) == BLOCKED

    @pytest.mark.parametrize("value", ["primary_api_key", "secondary_api_key", "some_local_variable"])
    def test_snake_case_variables_are_still_exempt(self, tmp_path: Path, value: str) -> None:
        """The relaxation must not reintroduce the false positives it was added to stop."""
        assert _check(tmp_path, "a.py", f"build(api_key={value})") == ALLOWED

    def test_env_template_placeholders_still_pass(self, tmp_path: Path) -> None:
        content = "METACULUS_TOKEN=1234567890\nOPENROUTER_API_KEY=1234567890\nASKNEWS_CLIENT_ID=1234567890\n"
        assert _check(tmp_path, ".env.template", content) == ALLOWED


class TestFailsClosed:
    def test_git_failure_does_not_silently_pass(self, tmp_path: Path) -> None:
        """`|| true` swallowed a git error as well as grep's no-match, so a broken git
        invocation yielded an empty diff and exit 0. A security hook must fail closed."""
        not_a_repo = tmp_path / "plain"
        not_a_repo.mkdir()
        result = subprocess.run([str(HOOK)], cwd=not_a_repo, capture_output=True, text=True)
        assert result.returncode != 0
        assert "git" in (result.stderr + result.stdout).lower()


class TestGuidanceDoesNotTeachBypass:
    def test_failure_message_recommends_the_pragma_not_no_verify(self, tmp_path: Path) -> None:
        """The hook's own rationale says reflexive --no-verify converts it into nothing.

        --no-verify disables EVERY pre-commit hook, including no-commit-to-main and ruff.
        The per-line pragma is the correct escape valve and is what the message should
        name.
        """
        repo = _repo(tmp_path)
        _stage(repo, "a.py", f"ASKNEWS_SECRET={FAKE_ANK}")
        result = subprocess.run([str(HOOK)], cwd=repo, capture_output=True, text=True)
        assert result.returncode == BLOCKED
        assert "pragma: allowlist secret" in result.stderr
        assert "--no-verify" not in result.stderr
