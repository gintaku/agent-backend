"""SQLAlchemy engine / session factories for app-owned Postgres tables.

Mirrors the write/read DSN split ``rag/pg_client.py`` uses for PGVector:
``db_write_url`` points at the primary (all INSERT/UPDATE/DELETE must go
through it) and ``db_read_url`` points at a read replica (SELECT-only).
For a single-instance Postgres setup both settings simply point at the same
place, and this split costs nothing — but it means conversations are ready
to benefit from a read replica the moment RAG's is introduced, without
another migration of this module.
"""
from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import get_settings

_write_engine: Engine | None = None
_read_engine: Engine | None = None
_WriteSession: sessionmaker[Session] | None = None
_ReadSession: sessionmaker[Session] | None = None


def get_write_engine() -> Engine:
    global _write_engine
    if _write_engine is None:
        _write_engine = create_engine(get_settings().db_write_url, pool_pre_ping=True)
    return _write_engine


def get_read_engine() -> Engine:
    global _read_engine
    if _read_engine is None:
        _read_engine = create_engine(get_settings().db_read_url, pool_pre_ping=True)
    return _read_engine


def get_write_session() -> Session:
    """Return a new Session bound to the write engine.

    Caller is responsible for closing it — use as a context manager:
    ``with get_write_session() as session: ...``.
    """
    global _WriteSession
    if _WriteSession is None:
        _WriteSession = sessionmaker(bind=get_write_engine(), expire_on_commit=False)
    return _WriteSession()


def get_read_session() -> Session:
    """Return a new Session bound to the read engine. Caller must close it."""
    global _ReadSession
    if _ReadSession is None:
        _ReadSession = sessionmaker(bind=get_read_engine(), expire_on_commit=False)
    return _ReadSession()


def reset_engines() -> None:
    """Dispose and clear all cached engines/session factories.

    Useful in tests and after config changes (mirrors rag.pg_client.reset_vectorstores).
    """
    global _write_engine, _read_engine, _WriteSession, _ReadSession
    for engine in (_write_engine, _read_engine):
        if engine is not None:
            engine.dispose()
    _write_engine = None
    _read_engine = None
    _WriteSession = None
    _ReadSession = None