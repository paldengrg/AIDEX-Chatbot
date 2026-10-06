"""The Human Layer: per-user memory items.

Every function takes the *current* user and only ever touches rows whose
user_id is that user's pseudonym. There is no function that accepts another
user's id, so one user can never read or change another user's memory.

Items are added manually (My account panel) or automatically (learning.py).
"""

import difflib
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import CATEGORIES, SOURCES, MemoryItem, User, utcnow

# Very common words that say nothing about relevance.
STOPWORDS = set("""a an and are as at be but by can do does for from has have how i if in
is it me my of on or so tell that the their them they this to us was we what when where
which who why will with you your about please""".split())

# Phrases that try to turn a "memory" into new instructions for the bot.
# (A first line of defence; Phase 4 adds fuller guardrails.)
INJECTION_PATTERNS = re.compile(
    r"ignore (all |any )?(previous|prior|above|earlier) (instructions|rules)"
    r"|system prompt|you are now|disregard (the|your) (rules|instructions)"
    r"|act as (an? )?(admin|developer|system)",
    re.IGNORECASE,
)

DUPLICATE_SIMILARITY = 0.85  # 0..1; above this, two rules count as the same


class MemoryInputError(ValueError):
    """Invalid memory input (shown to the user as a 400 error)."""


# --- Input sanitising -------------------------------------------------------------

def sanitize_rule(text: str) -> str:
    """Clean a rule before storing it, or raise MemoryInputError."""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text or "")   # control characters
    text = text.replace("<", "").replace(">", "")          # can't fake prompt tags
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise MemoryInputError("The rule is empty.")
    if len(text) > settings.memory_rule_max_chars:
        raise MemoryInputError(f"Keep rules under {settings.memory_rule_max_chars} characters.")
    if INJECTION_PATTERNS.search(text):
        raise MemoryInputError("That looks like an instruction to change the assistant's rules, "
                               "so it can't be saved as a memory.")
    return text


def _check_choice(value: str, allowed: tuple, name: str) -> str:
    if value not in allowed:
        raise MemoryInputError(f"{name} must be one of: {', '.join(allowed)}")
    return value


# --- Text helpers (pure functions, unit tested) --------------------------------------

def keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def is_duplicate(a: str, b: str) -> bool:
    norm = lambda s: " ".join(re.findall(r"[a-z0-9]+", s.lower()))  # noqa: E731
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= DUPLICATE_SIMILARITY


def relevance(item_text: str, category: str, query: str) -> float:
    """Score how relevant a memory item is to the current message.

    Intent-calibration rules describe *how* the user wants answers ("keep it
    short"), so they apply to almost every message: they get a base score of 1.
    Domain-knowledge items only count when they share keywords with the message.
    """
    overlap = len(keywords(item_text) & keywords(query))
    base = 1.0 if category == "intent_calibration" else 0.0
    return base + overlap


# --- Switching items on and off ---------------------------------------------------------

def deactivate(item: MemoryItem, reason: str) -> None:
    item.active = False
    item.deactivated_at = utcnow()
    item.deactivation_reason = reason


def reactivate(item: MemoryItem) -> None:
    """Turn an item back on with a fresh start, so the automatic review in
    learning.py doesn't immediately switch it off again for old reasons."""
    if item.active:
        return
    item.active = True
    item.deactivated_at = None
    item.deactivation_reason = None
    item.times_corrected = 0
    item.times_retrieved = item.times_used
    item.last_used_at = utcnow()


# --- CRUD (always scoped to the given user) -----------------------------------------

def list_items(db: Session, user: User) -> list[MemoryItem]:
    stmt = (select(MemoryItem).where(MemoryItem.user_id == user.pseudonym_id)
            .order_by(MemoryItem.created_at.desc()))
    return list(db.scalars(stmt))


def get_owned_item(db: Session, user: User, item_id: int) -> MemoryItem:
    item = db.get(MemoryItem, item_id)
    if item is None or item.user_id != user.pseudonym_id:
        raise LookupError("Memory item not found.")  # same message either way
    return item


def add_item(db: Session, user: User, rule_text: str, category: str,
             source: str = "manual") -> tuple[MemoryItem, bool]:
    """Save a rule. Returns (item, created). Near-duplicates reuse the old item."""
    rule_text = sanitize_rule(rule_text)
    _check_choice(category, CATEGORIES, "category")
    _check_choice(source, SOURCES, "source")

    for existing in list_items(db, user):
        if is_duplicate(existing.rule_text, rule_text):
            reactivate(existing)  # re-stating a rule revives it
            db.commit()
            return existing, False

    item = MemoryItem(user_id=user.pseudonym_id, rule_text=rule_text,
                      category=category, source=source)
    db.add(item)
    db.commit()
    return item, True


def update_item(db: Session, user: User, item_id: int, rule_text: str | None = None,
                category: str | None = None, active: bool | None = None) -> MemoryItem:
    item = get_owned_item(db, user, item_id)
    if rule_text is not None:
        item.rule_text = sanitize_rule(rule_text)
    if category is not None:
        item.category = _check_choice(category, CATEGORIES, "category")
    if active is True:
        reactivate(item)
    elif active is False and item.active:
        deactivate(item, "paused by user")
    db.commit()
    return item


def delete_item(db: Session, user: User, item_id: int) -> None:
    db.delete(get_owned_item(db, user, item_id))
    db.commit()


def delete_all(db: Session, user: User) -> int:
    items = list_items(db, user)
    for item in items:
        db.delete(item)
    db.commit()
    return len(items)


# --- Retrieval for the prompt ---------------------------------------------------------

def select_for_prompt(db: Session, user: User | None, query: str) -> list[MemoryItem]:
    """Pick this user's most relevant active items and log that they were retrieved."""
    if user is None or not user.memory_enabled:
        return []
    scored = [(relevance(i.rule_text, i.category, query), i)
              for i in list_items(db, user) if i.active]
    scored = [(s, i) for s, i in scored if s > 0]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    chosen = [i for _, i in scored[: settings.memory_max_items_in_prompt]]

    for item in chosen:
        item.times_retrieved += 1
    db.commit()
    return chosen
