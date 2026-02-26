"""Engine and session helpers."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from narva_queue.config import DEFAULT_DATABASE_URL


def get_database_url() -> str:
    """Get DB URL from environment."""
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and cache SQLAlchemy engine."""
    return create_engine(get_database_url(), pool_pre_ping=True)


@lru_cache(maxsize=1)
def _get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False)


@contextmanager
def get_session() -> Session:
    """Context manager for DB session lifecycle."""
    session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

