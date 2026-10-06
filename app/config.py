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
    assistant_name: str = os.getenv("ASSISTANT_NAME", "AIDEX Assistant")

    # LLM provider. "anthropic" uses the real API; "mock" returns canned
    # answers built from the retrieved documents (free, used by the tests).
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    # Optional cheaper/faster model for background jobs like memory extraction.
    # Falls back to LLM_MODEL when not set.
    llm_fast_model: str = os.getenv("LLM_FAST_MODEL", "") or os.getenv("LLM_MODEL", "")
    llm_max_tokens: int = _int("LLM_MAX_TOKENS", 800)

    # Retrieval (RAG)
    data_dir: Path = BASE_DIR / "data"
    # Relative paths are resolved from the project root; absolute paths are kept.
    chroma_dir: Path = BASE_DIR / os.getenv("CHROMA_DIR", "chroma_data")
    rag_top_k: int = _int("RAG_TOP_K", 8)

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
    # Learn memories automatically from conversations (Phase 3)
    auto_learn: bool = os.getenv("AUTO_LEARN", "true").lower() == "true"

    # Guardrails: maximum requests per person
    rate_limit_chat_per_minute: int = _int("RATE_LIMIT_CHAT_PER_MINUTE", 20)
    rate_limit_auth_per_15min: int = _int("RATE_LIMIT_AUTH_PER_15MIN", 10)

    # Research participation. Keep false until the human research ethics
    # application is approved: every log is then marked as synthetic/demo data
    # and the consent screen warns not to enter real personal information.
    ethics_approved: bool = os.getenv("ETHICS_APPROVED", "false").lower() == "true"

    # Admin dashboard (/admin). Leave ADMIN_PASSWORD empty to disable it.
    admin_username: str = os.getenv("ADMIN_USERNAME", "admin")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "")


settings = Settings()
