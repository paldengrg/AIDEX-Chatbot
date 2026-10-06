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


def complete(system: str, messages: list[dict], max_tokens: int | None = None,
             fast: bool = False) -> str:
    """Send a system prompt + chat history and return the reply text.

    `messages` is a list like [{"role": "user", "content": "Hi"}, ...] and must
    end with a user message. `fast=True` uses LLM_FAST_MODEL (for small
    background jobs such as memory extraction).
    """
    max_tokens = max_tokens or settings.llm_max_tokens
    model = settings.llm_fast_model if fast else settings.llm_model

    if settings.llm_provider == "mock":
        return _mock_complete(system)
    if settings.llm_provider == "anthropic":
        return _anthropic_complete(system, messages, max_tokens, model)
    raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")


# --- Anthropic -------------------------------------------------------------

_client = None  # created lazily so importing this module never needs a key


def _anthropic_complete(system: str, messages: list[dict], max_tokens: int, model: str) -> str:
    global _client
    if not settings.llm_api_key or not model:
        raise LLMError("LLM_API_KEY and LLM_MODEL must be set in .env")

    import anthropic  # imported here so the mock provider needs no SDK

    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.llm_api_key)
    try:
        response = _client.messages.create(
            model=model,
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
    """Return the first retrieved document chunk, or a fallback line.

    If the prompt lists personal memory items, pretend the first one was
    applied (so usage tracking can be tested without an API key).
    """
    match = re.search(r'<document [^>]*>\s*(.*?)\s*</document>', system, re.S)
    if match:
        reply = "(mock reply) " + match.group(1)[:400]
    else:
        reply = "(mock reply) I don't have information about that in the lab's documents."
    memory_ids = re.findall(r"\[(M\d+)\]", system)
    if memory_ids:
        reply += f"\nUSED_MEMORY: {memory_ids[0]}"
    return reply
