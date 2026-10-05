"""The single gateway to the language model.

Every LLM call in the project goes through `complete()`. The rest of the code
never imports a vendor SDK directly, so switching provider (or model) only
means changing this file and the .env values.

Providers:
  - "anthropic": the Anthropic Claude API (model + key from .env)
  - "mock":      no network, no cost. Echoes the most relevant retrieved
                 document so the app and tests run without an API key.
"""

import re

from app.config import settings


class LLMError(RuntimeError):
    """Raised when the model cannot produce a reply (bad key, network, ...)."""


def complete(system: str, messages: list[dict], max_tokens: int | None = None) -> str:
    """Send a system prompt + chat history and return the reply text.

    `messages` is a list like [{"role": "user", "content": "Hi"}, ...] and must
    end with a user message.
    """
    max_tokens = max_tokens or settings.llm_max_tokens

    if settings.llm_provider == "mock":
        return _mock_complete(system)
    if settings.llm_provider == "anthropic":
        return _anthropic_complete(system, messages, max_tokens)
    raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")


# --- Anthropic -------------------------------------------------------------

_client = None  # created lazily so importing this module never needs a key


def _anthropic_complete(system: str, messages: list[dict], max_tokens: int) -> str:
    global _client
    if not settings.llm_api_key or not settings.llm_model:
        raise LLMError("LLM_API_KEY and LLM_MODEL must be set in .env")

    import anthropic  # imported here so the mock provider needs no SDK

    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.llm_api_key)
    try:
        response = _client.messages.create(
            model=settings.llm_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
    except anthropic.APIError as exc:
        raise LLMError(str(exc)) from exc

    # A reply is a list of content blocks; we only need the text ones.
    return "".join(b.text for b in response.content if b.type == "text").strip()


# --- Mock --------------------------------------------------------------------

def _mock_complete(system: str) -> str:
    """Return the first retrieved document chunk, or a fallback line."""
    match = re.search(r'<document source="[^"]*">\s*(.*?)\s*</document>', system, re.S)
    if match:
        return "(mock reply) " + match.group(1)[:400]
    return "(mock reply) I don't have information about that in the lab's documents."
