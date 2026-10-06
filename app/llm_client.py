"""The single gateway to the language model.

Every LLM call in the project goes through `complete()`. The rest of the code
never imports a vendor SDK directly, so switching provider (or model) only
means changing this file and the .env values.

Providers:
  - "ollama": a model running on this computer through Ollama, such as
              gemma2:2b (free, no API key; the model name comes from .env)
  - "mock":   no model at all. Echoes the most relevant retrieved document so
              the app and tests run without Ollama.
"""

import re

from app.config import settings


class LLMError(RuntimeError):
    """Raised when the model cannot produce a reply (Ollama not running, ...)."""


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
    if settings.llm_provider == "ollama":
        return _ollama_complete(system, messages, max_tokens, model)
    raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")


# --- Ollama (local open models such as Gemma) ---------------------------------

def _ollama_complete(system: str, messages: list[dict], max_tokens: int, model: str) -> str:
    """Call a model served on this machine by Ollama (https://ollama.com), e.g. gemma2:2b."""
    if not model:
        raise LLMError("LLM_MODEL must be set in .env (for example gemma2:2b)")

    import httpx

    try:
        response = httpx.post(f"{settings.ollama_url}/api/chat", timeout=180, json={
            "model": model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            # Ollama's default context window is small and silently cuts long
            # prompts; Gemma 2 can read up to 8k tokens.
            "options": {"num_predict": max_tokens, "num_ctx": 8192},
        })
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMError(f"Ollama: {exc}") from exc
    return response.json()["message"]["content"].strip()


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
