"""SQLAlchemy ORM models for application tables stored in Postgres.

These live in the same physical database as the RAG vector store
(``db_write_url`` / ``db_read_url`` in config.py), but in their own tables —
managed by this module's ``Base`` metadata / Alembic migrations, entirely
separate from the tables ``langchain-postgres`` (PGVector) creates for
embeddings. Never mix the two metadata objects: PGVector manages its own
schema lazily and must remain untouched by Alembic.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB on Postgres (production), plain JSON everywhere else. Using the
# Postgres-only JSONB type directly would make this model uncompilable
# against any other dialect — including the in-memory SQLite engine the
# test suite uses to exercise conversation_store.py without a real
# Postgres instance (see tests/test_conversation_store.py). with_variant
# keeps production on JSONB (indexable, binary-stored) while letting tests
# fall back to generic JSON, with zero behavior change in production.
_JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    """Metadata root for app-owned tables (as opposed to PGVector's own)."""


#: Placeholder "logged in" user until a real auth/user module exists.
#: Every row is tagged with a user so the schema doesn't need to change
#: shape when multi-user support lands — only this default goes away.
DEFAULT_USER = "admin"


class Conversation(Base):
    """One row per chat session.

    Mirrors the JSON-file schema conversation_store.py used to write to
    ``conversations/{session_id}.json``: same fields, same semantics —
    just persisted in Postgres instead of on local disk.
    """

    __tablename__ = "conversations"

    # Session IDs are already validated (^[a-zA-Z0-9_-]{1,128}$) before
    # reaching this layer — see conversation_store._validate_session_id.
    id: Mapped[str] = mapped_column(String(128), primary_key=True)

    # Quoted automatically by SQLAlchemy's Postgres dialect since "user" is
    # a reserved word — no special handling needed here.
    user: Mapped[str] = mapped_column(
        String(128), nullable=False, default=DEFAULT_USER, server_default=DEFAULT_USER, index=True
    )

    title: Mapped[str] = mapped_column(String, nullable=False, default="Untitled")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    # Serialized LangChain messages (context restoration) and raw frontend
    # message dicts (UI restoration) — same two blobs the JSON file used to
    # hold, now as native JSONB columns instead of one big file.
    lc_messages: Mapped[list] = mapped_column(_JSON_TYPE, nullable=False, default=list)
    ui_messages: Mapped[list] = mapped_column(_JSON_TYPE, nullable=False, default=list)