"""Database connection (SQLite via SQLAlchemy).

`get_db` is a FastAPI dependency: each request gets its own session, which is
closed automatically when the request finishes.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# check_same_thread=False: FastAPI may use a session from a worker thread.
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Parent class for all table models (see models.py)."""


def init_db() -> None:
    """Create any missing tables. Safe to call on every startup."""
    from app import models  # noqa: F401  (registers the tables on Base)

    Base.metadata.create_all(engine)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
