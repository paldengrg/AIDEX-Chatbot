"""Database tables.

Two stores are deliberately kept apart:
  - Domain knowledge lives in ChromaDB (rag.py), shared by everyone.
  - The "Human Layer" lives here: per-user MemoryItem rows.

Privacy: memory items reference users by a random `pseudonym_id`, never by
username. Research exports (Phase 5) only ever see the pseudonym.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

# Allowed values, kept in one place so the API and UI can reuse them.
TIERS = ("student", "participant")
CATEGORIES = ("intent_calibration", "domain_knowledge")
SOURCES = ("explicit_instruction", "correction", "inferred", "manual")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_pseudonym() -> str:
    return uuid.uuid4().hex


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(30), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    pseudonym_id: Mapped[str] = mapped_column(String(32), unique=True, default=new_pseudonym)
    tier: Mapped[str] = mapped_column(String(20), default="student")
    # Students must opt in before anything is remembered about them.
    memory_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MemoryItem(Base):
    """One learned rule about one user (the paper's unit of 'memory')."""

    __tablename__ = "memory_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.pseudonym_id", ondelete="CASCADE"), index=True
    )
    rule_text: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(30))   # one of CATEGORIES
    source: Mapped[str] = mapped_column(String(30))     # one of SOURCES
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Effectiveness counters (filled in from Phase 3)
    times_retrieved: Mapped[int] = mapped_column(Integer, default=0)  # put in a prompt
    times_used: Mapped[int] = mapped_column(Integer, default=0)       # model applied it
    times_helpful: Mapped[int] = mapped_column(Integer, default=0)    # thumbs up
    times_corrected: Mapped[int] = mapped_column(Integer, default=0)  # user corrected bot
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    active: Mapped[bool] = mapped_column(Boolean, default=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "rule_text": self.rule_text,
            "category": self.category,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "times_retrieved": self.times_retrieved,
            "times_used": self.times_used,
            "times_helpful": self.times_helpful,
            "times_corrected": self.times_corrected,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "active": self.active,
        }
