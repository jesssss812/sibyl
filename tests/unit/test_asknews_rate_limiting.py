"""Test that AskNews integration properly handles rate limiting."""

import asyncio
import time
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sibyl.research.providers as providers
from sibyl.research.providers import _asknews_provider


class _Clock:
    """Virtual time for the provider: its sleeps advance the clock instead of waiting.

    The provider waits 10s before each AskNews call on purpose, so real time cost these
    two tests 40s of the suite. Only the provider module's ``asyncio`` and ``time`` are
    replaced, so the event loop itself keeps real time.
    """

    def __init__(self) -> None:
        self.now = 1_000_000.0

    async def sleep(self, seconds: float, *_args: object) -> None:
        self.now += max(seconds, 0.0)
        await asyncio.sleep(0)

    def monotonic(self) -> float:
        return self.now


class _Forward:
    def __init__(self, module: object, **overrides: object) -> None:
        self._module = module
        self.__dict__.update(overrides)

    def __getattr__(self, name: str) -> object:
        return getattr(self._module, name)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Clock]:
    fake = _Clock()
    monkeypatch.setattr(providers, "asyncio", _Forward(asyncio, sleep=fake.sleep))
    monkeypatch.setattr(providers, "time", _Forward(time, monotonic=fake.monotonic))
    monkeypatch.setattr(providers, "_ASKNEWS_LAST_CALL_TS", 0.0)
    yield fake


def _make_q(text: str) -> MagicMock:
    """Build a minimal MetaculusQuestion-shaped mock for the new ResearchCallable
    contract."""
    q = MagicMock()
    q.question_text = text
    return q


@pytest.mark.asyncio
async def test_asknews_rate_limiting_delay(clock: _Clock):
    """Test that AskNews provider waits between API calls to respect rate limits."""

    # Mock the AsyncAskNewsSDK to track timing
    call_times = []

    async def mock_search_news(*args, **kwargs):
        call_times.append(clock.now)
        # Create a minimal mock response
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        return mock_response

    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: {
            "ASKNEWS_CLIENT_ID": "test_client_id",
            "ASKNEWS_SECRET": "test_secret",
        }.get(key, default)

        with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
            mock_sdk = AsyncMock()
            mock_sdk.news.search_news = mock_search_news
            mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

            provider = _asknews_provider()
            await provider(_make_q("test question"))

            # Verify two calls were made
            assert len(call_times) == 2

            # Verify there was a delay between calls (should be ~1.2 seconds)
            time_diff = call_times[1] - call_times[0]
            assert time_diff >= 10.0, f"Expected delay >= 10.0s, got {time_diff:.2f}s"
            assert time_diff <= 11.1, f"Expected delay <= 11.1s, got {time_diff:.2f}s"


@pytest.mark.asyncio
async def test_asknews_calls_both_endpoints(clock: _Clock):
    """Test that AskNews provider calls both latest and historical news endpoints."""

    search_calls = []

    async def mock_search_news(*args, **kwargs):
        search_calls.append(kwargs.get("strategy", "unknown"))
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        return mock_response

    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: {
            "ASKNEWS_CLIENT_ID": "test_client_id",
            "ASKNEWS_SECRET": "test_secret",
        }.get(key, default)

        with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
            mock_sdk = AsyncMock()
            mock_sdk.news.search_news = mock_search_news
            mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

            provider = _asknews_provider()
            result = await provider(_make_q("test question"))

            # Verify both strategies were called
            assert "latest news" in search_calls
            assert "news knowledge" in search_calls
            assert len(search_calls) == 2

            # Verify result format
            assert "No articles were found" in result
