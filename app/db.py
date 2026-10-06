"""Database connection (SQLite via SQLAlchemy).

`get_db` is a FastAPI dependency: each request gets its own session, which is
closed automatically when the request finishes.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# check_same_thread=False: FastAPI may use a session from a worker thread.
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Parent class for all table models (see models.py)."""


# Columns added after a table was first created. SQLAlchemy's create_all() makes
# new tables but never changes existing ones, so an older aidx.db would be
# missing these. We add them on startup (a tiny, hand-written "migration").
_ADDED_COLUMNS = {
    "users": {"clarify_score": "FLOAT DEFAULT 0.5", "consent_given_at": "DATETIME",
              "consent_version": "VARCHAR(30)", "consent_withdrawn_at": "DATETIME"},
    "memory_items": {"deactivated_at": "DATETIME", "deactivation_reason": "VARCHAR(100)"},
}


def init_db() -> None:
    """Create missing tables and columns. Safe to call on every startup."""
    from app import models  # noqa: F401  (registers the tables on Base)

    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, columns in _ADDED_COLUMNS.items():
            existing = {c["name"] for c in inspector.get_columns(table)}
            for name, sql_type in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
