"""Short-term memory for anonymous visitors.

Each browser gets a random session id (stored in a cookie). We keep the last
few turns of that conversation in server RAM so follow-up questions make sense
("what else has *he* published?").

Privacy by design: nothing here is written to disk. Sessions expire after a
period of inactivity and are lost on server restart. Persistent, per-user
memory (the "Human Layer") is a separate store added in Phase 2.
"""

import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from app.config import settings

SESSION_COOKIE = "aidx_session"  # cookie holding the random session id


@dataclass
class Session:
    # Each turn is {"role": "user"|"assistant", "content": str}.
    turns: deque = field(default_factory=lambda: deque(maxlen=settings.session_max_turns * 2))
    last_seen: float = field(default_factory=time.time)
    # Database id of the previous turn (signed-in users with memory only), so
    # a correction in the next message can be linked back to it.
    last_exchange_id: int | None = None


class SessionStore:
    def __init__(self, ttl_seconds: int):
        self._ttl = ttl_seconds
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()  # FastAPI may serve requests in parallel

    def get_or_create(self, session_id: str | None) -> tuple[str, Session]:
        """Return an existing live session, or start a new one."""
        with self._lock:
            self._purge_expired()
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
            else:
                session_id = secrets.token_urlsafe(16)  # unguessable id
                session = self._sessions[session_id] = Session()
            session.last_seen = time.time()
            return session_id, session

    def add_exchange(self, session: Session, user_text: str, reply: str) -> None:
        with self._lock:
            session.turns.append({"role": "user", "content": user_text})
            session.turns.append({"role": "assistant", "content": reply})

    def history(self, session: Session) -> list[dict]:
        with self._lock:
            return list(session.turns)

    def reset(self, session_id: str | None) -> None:
        with self._lock:
            self._sessions.pop(session_id or "", None)

    def _purge_expired(self) -> None:
        cutoff = time.time() - self._ttl
        for sid in [s for s, sess in self._sessions.items() if sess.last_seen < cutoff]:
            del self._sessions[sid]


sessions = SessionStore(ttl_seconds=settings.session_ttl_minutes * 60)
