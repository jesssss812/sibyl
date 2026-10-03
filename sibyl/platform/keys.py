"""Utility functions for API key management."""

import os

from sibyl.config.settings import OAI_ANTH_OPENROUTER_KEY_ENV, OPENROUTER_API_KEY_ENV
from sibyl.platform.key_fallback import should_route_via_primary_key


def get_openrouter_api_key(model: str) -> str | None:
    """Return the primary key for providers it covers (openai/anthropic/google), else the general key.

    See ``OPENROUTER_PRIMARY_KEY_PROVIDERS`` in ``fallback_openrouter`` for the primary-key provider set.
    """
    from sibyl.platform.experiment_funding import active_grant

    grant = active_grant()
    if grant is not None:
        if not model.startswith("openrouter/"):
            raise ValueError("Experiment grant permits OpenRouter routes only")
        return grant.key
    if should_route_via_primary_key(model):
        special_key = os.getenv(OAI_ANTH_OPENROUTER_KEY_ENV)
        if special_key:
            return special_key

    return os.getenv(OPENROUTER_API_KEY_ENV)
