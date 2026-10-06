"""Guardrails: checks that run *before* a message reaches the model.

Layers of protection (each simple enough to explain):
  1. sanitize_message  removes invisible control characters.
  2. check_message     catches two cases we answer with a fixed, safe reply
                       instead of calling the model:
                         - attempts to override the assistant's instructions
                         - messages suggesting someone may be in crisis
  3. RateLimiter       stops one person flooding the chat or guessing passwords.

Staying on topic and declining harmful or unrelated requests is handled by the
rules in the system prompt (chat.py), because the model judges those far
better than keyword lists can.
"""

import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import HTTPException, Request

from app.config import settings

# --- 1. Sanitising ------------------------------------------------------------------

def sanitize_message(text: str) -> str:
    """Remove control characters (keeping newlines/tabs) and trim whitespace."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text or "").strip()


# --- 2. Message checks --------------------------------------------------------------

INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above|earlier|your) (instructions|rules|prompt)"
    r"|disregard (the|your|all) (rules|instructions)"
    r"|(reveal|show|print|repeat|tell me) (me )?(your|the) (system |hidden |initial )?(prompt|instructions)"
    r"|you are now (a|an|in)|pretend (you have|there are) no (rules|restrictions)"
    r"|developer mode|jailbreak",
    re.IGNORECASE,
)

CRISIS = re.compile(
    r"\b(kill(ing)? myself|suicid\w*|self[- ]?harm|end(ing)? my life|want to die"
    r"|hurt(ing)? myself)\b",
    re.IGNORECASE,
)

INJECTION_REPLY = (
    "I can't change how I work or share my internal instructions, but I'm happy to "
    "help with questions about the AIDX Lab, its research, people or opportunities."
)

CRISIS_REPLY = (
    "I'm really sorry you're going through this. I'm only the lab's information "
    "assistant, but you don't have to face this alone. In Australia you can call "
    "**Lifeline on 13 11 14** (24/7) or text 0477 13 11 14, and in an emergency call "
    "**000**. ACU students can also contact ACU Counselling through the student portal."
)


@dataclass
class GuardResult:
    blocked: bool
    reason: str = ""   # "injection" | "crisis"
    reply: str = ""    # the fixed reply to show instead of calling the model


def check_message(text: str) -> GuardResult:
    if CRISIS.search(text):
        return GuardResult(True, "crisis", CRISIS_REPLY)
    if INJECTION.search(text):
        return GuardResult(True, "injection", INJECTION_REPLY)
    return GuardResult(False)


# --- 3. Rate limiting -----------------------------------------------------------------

class RateLimiter:
    """Allow at most `limit` events per `window` seconds for each key.

    Keeps a queue of recent timestamps per key ("sliding window"). Stored in
    RAM, which is fine for a single-server prototype.
    """

    def __init__(self, limit: int, window: float):
        self.limit = limit
        self.window = window
        self._events: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, now: float | None = None) -> float:
        """Record an attempt. Returns 0 if allowed, else seconds to wait."""
        now = time.monotonic() if now is None else now
        with self._lock:
            events = self._events[key]
            while events and events[0] <= now - self.window:
                events.popleft()  # forget attempts outside the window
            if len(events) >= self.limit:
                return max(0.1, round(events[0] + self.window - now, 1))
            events.append(now)
            return 0.0


chat_limiter = RateLimiter(settings.rate_limit_chat_per_minute, 60)
auth_limiter = RateLimiter(settings.rate_limit_auth_per_15min, 15 * 60)


def client_key(request: Request, user=None) -> str:
    """Who is making the request: the account if signed in, else the IP address."""
    if user is not None:
        return f"user:{user.id}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def enforce(limiter: RateLimiter, key: str) -> None:
    """Raise HTTP 429 "Too Many Requests" when the key is over its limit."""
    wait = limiter.hit(key)
    if wait:
        raise HTTPException(429, f"Too many requests. Please wait {int(wait) + 1} seconds.",
                            headers={"Retry-After": str(int(wait) + 1)})
