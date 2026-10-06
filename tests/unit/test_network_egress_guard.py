"""The autouse egress guard in tests/conftest.py must also stop proxied requests.

A cloud agent proxy listens on loopback and adds API credentials to what it forwards,
so allowing loopback alone let golden tests make real, paid OpenRouter calls.
"""

from __future__ import annotations

import os
import socket

import pytest

from tests.conftest import _PROXY_ENV_VARS, _address_is_blocked, _proxy_endpoints

_PROXY = frozenset({("127.0.0.1", 42139)})


def test_proxy_endpoints_are_parsed_from_every_spelling() -> None:
    environ = {"HTTPS_PROXY": "http://127.0.0.1:42139", "http_proxy": "localhost:3128", "ALL_PROXY": ""}
    assert _proxy_endpoints(environ) == {("127.0.0.1", 42139), ("localhost", 3128)}


def test_a_connect_to_the_proxy_is_blocked() -> None:
    assert _address_is_blocked(socket.AF_INET, ("127.0.0.1", 42139), _PROXY)
    assert _address_is_blocked(socket.AF_INET, ("localhost", 42139), _PROXY)


def test_other_loopback_ports_and_unix_sockets_stay_allowed() -> None:
    assert not _address_is_blocked(socket.AF_INET, ("127.0.0.1", 8080), _PROXY)
    assert not _address_is_blocked(socket.AF_UNIX, "/tmp/sock", _PROXY)


def test_a_real_host_is_blocked() -> None:
    assert _address_is_blocked(socket.AF_INET, ("104.18.2.115", 443), frozenset())


@pytest.mark.parametrize("name", _PROXY_ENV_VARS)
def test_proxy_variables_are_removed_during_tests(name: str) -> None:
    assert name not in os.environ
