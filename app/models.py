"""Database tables.

Two stores are deliberately kept apart:
  - Domain knowledge lives in ChromaDB (rag.py), shared by everyone.
  - The "Human Layer" lives here: per-user MemoryItem rows.

Privacy: memory items reference users by a random `pseudonym_id`, never by
username. Research exports (Phase 5) only ever see the pseudonym.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
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
    # Learned per user: 0 = "just answer, don't ask me", 1 = "check with me
    # first when my request is ambiguous". Starts neutral (see learning.py).
    clarify_score: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Research consent (tier "participant"). The version records exactly which
    # information sheet the person agreed to.
    consent_given_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    consent_version: Mapped[str | None] = mapped_column(String(30), nullable=True)
    consent_withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True),
                                                                  nullable=True)

    @property
    def is_participant(self) -> bool:
        return self.tier == "participant" and self.consent_given_at is not None


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
    # Why and when an item was switched off (for the dashboard's "deactivated
    # over time" chart). Empty while the item is active.
    deactivated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deactivation_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)

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
            "deactivated_at": self.deactivated_at.isoformat() if self.deactivated_at else None,
            "deactivation_reason": self.deactivation_reason,
        }


class Exchange(Base):
    """One question/answer turn by a signed-in user who opted in to memory.

    Privacy: no message text is stored here, only which memory items were
    involved and the feedback signals. That is all the effectiveness metrics
    need. (Full transcripts are only for consenting research participants,
    added in Phase 4.)
    """

    __tablename__ = "exchanges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.pseudonym_id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    retrieved_ids: Mapped[str] = mapped_column(Text, default="")  # e.g. "3,7,9"
    used_ids: Mapped[str] = mapped_column(Text, default="")       # subset the model applied
    asked_clarification: Mapped[bool] = mapped_column(Boolean, default=False)
    feedback: Mapped[int | None] = mapped_column(Integer, nullable=True)  # +1 / -1
    was_corrected: Mapped[bool] = mapped_column(Boolean, default=False)   # next msg corrected it

    @staticmethod
    def join_ids(ids) -> str:
        return ",".join(str(i) for i in ids)

    @staticmethod
    def split_ids(text: str) -> list[int]:
        return [int(x) for x in text.split(",") if x]



class ConversationLog(Base):
    """Full transcript of one turn: ONLY for research participants who consented.

    Deleted when the participant withdraws. `synthetic` is true whenever ethics
    approval is not yet confirmed (ETHICS_APPROVED=false), so demo data can
    never be mistaken for real research data.
    """

    __tablename__ = "conversation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.pseudonym_id"), index=True)
    exchange_id: Mapped[int | None] = mapped_column(ForeignKey("exchanges.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    user_message: Mapped[str] = mapped_column(Text)
    assistant_reply: Mapped[str] = mapped_column(Text)
    sources: Mapped[str] = mapped_column(Text, default="")        # "people.md, news.md"
    guardrail: Mapped[str | None] = mapped_column(String(20), nullable=True)  # if blocked
    synthetic: Mapped[bool] = mapped_column(Boolean, default=True)
