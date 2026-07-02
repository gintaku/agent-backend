"""Read/write PGVector vectorstores for the RAG module.

Two singletons are kept — one bound to the write (primary) DSN and one bound
to the read (replica) DSN — both routed through pgbouncer/CNPG. Ingestion
code must always go through :func:`get_write_vectorstore`; retrieval code
must always go through :func:`get_read_vectorstore`.

Only the write instance is allowed to trigger schema creation: PGVector
creates its collection/embedding tables lazily on first use, and that DDL
should never happen against a read replica.

Note: even with ``create_extension=False``, PGVector's constructor still
issues a couple of lightweight statements on first use — a checkfirst
``CREATE TABLE IF NOT EXISTS`` (a no-op once the write store has already
created the tables) and a SELECT-or-INSERT for the collection row (a no-op
once that row exists). Both are harmless once the write store has ingested
at least once, but until then, constructing the read store can still fail
against a genuine read-only replica. Callers must treat any failure from
:func:`get_read_vectorstore` (construction *or* query) as "no context
available" rather than a hard error — see ``rag/retriever.py``.
"""
from __future__ import annotations

from langchain_postgres import PGVector

from config import get_settings
from rag.embeddings import get_embeddings

_write_store: PGVector | None = None
_read_store: PGVector | None = None


def get_write_vectorstore() -> PGVector:
    """Return the lazily-initialised PGVector instance bound to ``db_write_url``.

    Creates the collection/embedding tables on first use if they don't exist
    yet. All ingestion (add/delete) must go through this instance.
    """
    global _write_store
    if _write_store is None:
        s = get_settings()
        _write_store = PGVector(
            embeddings=get_embeddings(),
            collection_name=s.rag_collection_name,
            connection=s.db_write_url,
            use_jsonb=True,
        )
    return _write_store


def get_read_vectorstore() -> PGVector:
    """Return the lazily-initialised PGVector instance bound to ``db_read_url``.

    Used for retrieval only. Assumes the schema already exists — call
    :func:`get_write_vectorstore` at least once (e.g. via startup ingestion)
    before relying on this in a fresh environment. ``create_extension=False``
    is deliberate: a genuine Postgres read replica rejects DDL outright
    (``CREATE EXTENSION`` included), so this instance must never attempt it —
    only the write store is allowed to.
    """
    global _read_store
    if _read_store is None:
        s = get_settings()
        _read_store = PGVector(
            embeddings=get_embeddings(),
            collection_name=s.rag_collection_name,
            connection=s.db_read_url,
            use_jsonb=True,
            create_extension=False,
        )
    return _read_store


def reset_vectorstores() -> None:
    """Clear both cached singletons so the next call creates fresh instances.

    Useful in tests and after config changes.
    """
    global _write_store, _read_store
    _write_store = None
    _read_store = None