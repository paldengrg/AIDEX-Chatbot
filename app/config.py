"""Central configuration.

Every setting comes from environment variables (loaded from a local .env file
if present), so secrets never live in the code and nothing is hardcoded.
Other modules import `settings` from here instead of reading os.environ.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains app/, static/, data/ ...
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # Branding: the only place the assistant's name is defined.
    assistant_name: str = os.getenv("ASSISTANT_NAME", "AIDX Assistant")

    # LLM provider. "anthropic" uses the real API; "mock" returns canned
    # answers built from the retrieved documents (free, used by the tests).
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_max_tokens: int = _int("LLM_MAX_TOKENS", 800)

    # Retrieval (RAG)
    data_dir: Path = BASE_DIR / "data"
    # Relative paths are resolved from the project root; absolute paths are kept.
    chroma_dir: Path = BASE_DIR / os.getenv("CHROMA_DIR", "chroma_data")
    rag_top_k: int = _int("RAG_TOP_K", 4)

    # Anonymous session memory
    session_max_turns: int = _int("SESSION_MAX_TURNS", 10)
    session_ttl_minutes: int = _int("SESSION_TTL_MINUTES", 60)

    # Basic input limit (full guardrails arrive in Phase 4)
    max_message_chars: int = _int("MAX_MESSAGE_CHARS", 2000)

    # Database (SQLite file in the project root by default)
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'aidx.db').as_posix()}")

    # Accounts. SECRET_KEY signs login cookies: keep it long, random and private.
    secret_key: str = os.getenv("SECRET_KEY", "")
    login_days: int = _int("LOGIN_DAYS", 7)
    # Set to true when served over HTTPS so cookies are never sent unencrypted.
    cookie_secure: bool = os.getenv("COOKIE_SECURE", "false").lower() == "true"

    # Per-user memory
    memory_max_items_in_prompt: int = _int("MEMORY_MAX_ITEMS_IN_PROMPT", 8)
    memory_rule_max_chars: int = _int("MEMORY_RULE_MAX_CHARS", 300)


settings = Settings()
