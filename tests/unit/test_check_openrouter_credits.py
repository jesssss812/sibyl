"""Tests for sibyl.telemetry.check_credits.

Mocks the HTTP call so this runs without network access. Asserts that the
full API key never appears in stdout, and that exit codes line up with what
callers / Make targets will see.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from sibyl.telemetry import check_credits as cc

PRIMARY_KEY = "sk-or-v1-PRIMARYsecretAB12"
FALLBACK_KEY = "sk-or-v1-FALLBACKsecretCD34"


def _ok_response(label: str, usage: float = 12.5) -> httpx.Response:
    payload = {
        "data": {
            "label": label,
            "limit": 500.0,
            "limit_remaining": 500.0 - usage,
            "usage": usage,
            "usage_daily": 1.5,
            "usage_weekly": 4.2,
            "usage_monthly": usage,
            "is_free_tier": False,
            "limit_reset": "monthly",
        }
    }
    return httpx.Response(200, content=json.dumps(payload).encode())


def _err_response(status: int) -> httpx.Response:
    return httpx.Response(status, content=b'{"error": "boom"}')


def _patch_env(monkeypatch: pytest.MonkeyPatch, primary: str | None, fallback: str | None) -> None:
    if primary is None:
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
    else:
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", primary)
    if fallback is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", fallback)


def _patch_load_env():
    """Stop the script from re-loading real `.env` during tests."""
    return patch.object(cc, "load_environment", lambda: None)


def test_redact_short_and_long_keys():
    assert cc._redact("abcd") == "****"
    assert cc._redact("abcdef1234") == "****1234"


def test_both_keys_present_both_succeed(monkeypatch, capsys):
    _patch_env(monkeypatch, PRIMARY_KEY, FALLBACK_KEY)
    fake = httpx.MockTransport(lambda _req: _ok_response("Metaculus-Primary"))

    with _patch_load_env(), patch.object(httpx, "get", lambda url, **kw: httpx.Client(transport=fake).get(url, **kw)):
        rc = cc.main(["--key", "both"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Primary key (OAI_ANTH_OPENROUTER_KEY)" in captured.out
    assert "Fallback key (OPENROUTER_API_KEY)" in captured.out
    # Redacted fingerprints visible
    assert "****AB12" in captured.out
    assert "****CD34" in captured.out
    # Full keys NEVER printed
    assert PRIMARY_KEY not in captured.out
    assert FALLBACK_KEY not in captured.out
    assert "PRIMARYsecret" not in captured.out
    assert "FALLBACKsecret" not in captured.out


def test_primary_missing_fallback_present_exits_zero(monkeypatch, capsys):
    _patch_env(monkeypatch, None, FALLBACK_KEY)
    fake = httpx.MockTransport(lambda _req: _ok_response("Fallback"))

    with _patch_load_env(), patch.object(httpx, "get", lambda url, **kw: httpx.Client(transport=fake).get(url, **kw)):
        rc = cc.main(["--key", "both"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "WARN: env var OAI_ANTH_OPENROUTER_KEY not set" in captured.out
    assert "Fallback key (OPENROUTER_API_KEY)" in captured.out
    assert FALLBACK_KEY not in captured.out


def test_both_keys_missing_exits_one(monkeypatch, capsys):
    _patch_env(monkeypatch, None, None)

    with _patch_load_env():
        rc = cc.main(["--key", "both"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "WARN: env var OAI_ANTH_OPENROUTER_KEY not set" in captured.out
    assert "WARN: env var OPENROUTER_API_KEY not set" in captured.out
    assert "ERROR: no keys could be checked." in captured.err


def test_http_401_friendly_error(monkeypatch, capsys):
    _patch_env(monkeypatch, PRIMARY_KEY, None)
    fake = httpx.MockTransport(lambda _req: _err_response(401))

    with _patch_load_env(), patch.object(httpx, "get", lambda url, **kw: httpx.Client(transport=fake).get(url, **kw)):
        rc = cc.main(["--key", "primary"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "401 Unauthorized" in captured.out
    # Even on error, full key must not leak
    assert PRIMARY_KEY not in captured.out
    assert PRIMARY_KEY not in captured.err


def test_only_primary_requested(monkeypatch, capsys):
    _patch_env(monkeypatch, PRIMARY_KEY, FALLBACK_KEY)
    fake = httpx.MockTransport(lambda _req: _ok_response("Metaculus-Primary"))

    with _patch_load_env(), patch.object(httpx, "get", lambda url, **kw: httpx.Client(transport=fake).get(url, **kw)):
        rc = cc.main(["--key", "primary"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Primary key (OAI_ANTH_OPENROUTER_KEY)" in captured.out
    assert "Fallback key (OPENROUTER_API_KEY)" not in captured.out
    assert FALLBACK_KEY not in captured.out
