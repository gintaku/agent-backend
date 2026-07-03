"""Persistence layer for conversation history — backed by Postgres.

Conversations are stored in the ``conversations`` table (see
``db/models.py``), in the same Postgres database used by the RAG pipeline —
just a different table, with its own Alembic-managed schema (see
``alembic/``). This module assumes the table already exists; it does not
create it (run ``alembic upgrade head`` as part of deployment/startup).

There's no auth/user module yet, so every conversation is tagged with a
``user`` column that defaults to ``"admin"``. Every public function here
accepts an optional ``user`` argument for forward compatibility — once a
real auth module exists, callers just start passing the actual user id and
nothing else about this module needs to change.

Schema (unchanged from the old per-file JSON layout, just relocated):
  id            session_id (primary key)
  user          owner of the conversation (default "admin")
  title         display title
  created_at    tz-aware timestamp, preserved across updates
  updated_at    tz-aware timestamp, bumped on every write
  lc_messages   serialized LangChain messages (context restoration), JSONB
  ui_messages   raw frontend message dicts (display restoration), JSONB
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from db.engine import get_read_session, get_write_session
from db.models import DEFAULT_USER, Conversation

logger = logging.getLogger("app")

# Session IDs must be alphanumeric + hyphens/underscores, max 128 chars.
# Kept even though Postgres parameterizes values (no SQL-injection risk via
# the ORM) — this is the primary key's own format contract, and it's cheap
# insurance against silently accepting garbage session ids from the client.
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(f"Invalid session_id: {session_id!r}")


# ---------------------------------------------------------------------------
# Serialization helpers (unchanged — no storage dependency)
# ---------------------------------------------------------------------------

def serialize_lc_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    result = []
    for m in messages:
        payload: dict[str, Any] = {"type": m.type, "content": m.content}
        name = getattr(m, "name", None)
        if name:
            payload["name"] = name
        additional_kwargs = getattr(m, "additional_kwargs", None)
        if additional_kwargs:
            payload["additional_kwargs"] = additional_kwargs
        result.append(payload)
    return result


def deserialize_lc_messages(serialized: list[dict[str, Any]]) -> list[BaseMessage]:
    result = []
    for d in serialized:
        msg_type = d.get("type", "")
        content = d.get("content", "")
        if msg_type == "human":
            result.append(HumanMessage(content=content))
        elif msg_type == "ai":
            result.append(AIMessage(content=content))
        elif msg_type == "system":
            result.append(SystemMessage(content=content))
        # unknown types are skipped
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_conversation(
    session_id: str,
    title: str,
    lc_messages: list[dict[str, Any]],
    ui_messages: list[dict[str, Any]],
    user: str = DEFAULT_USER,
) -> None:
    """Create or update a conversation row.

    ``created_at`` is preserved across updates (only set when the row is
    first created) — same behavior the old JSON-file version had.
    """
    _validate_session_id(session_id)
    now = datetime.now(timezone.utc)

    with get_write_session() as session:
        try:
            conv = session.get(Conversation, session_id)
            if conv is None:
                conv = Conversation(
                    id=session_id,
                    user=user,
                    title=title,
                    created_at=now,
                    updated_at=now,
                    lc_messages=lc_messages,
                    ui_messages=ui_messages,
                )
                session.add(conv)
            else:
                conv.title = title
                conv.updated_at = now
                conv.lc_messages = lc_messages
                conv.ui_messages = ui_messages
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            logger.exception(
                "Failed to save conversation", extra={"session_id": session_id}
            )
            raise


def load_conversation(session_id: str) -> dict[str, Any] | None:
    try:
        _validate_session_id(session_id)
    except ValueError:
        return None

    with get_read_session() as session:
        try:
            conv = session.get(Conversation, session_id)
        except SQLAlchemyError:
            logger.exception(
                "Failed to load conversation — treating as missing",
                extra={"session_id": session_id},
            )
            return None

    if conv is None:
        return None

    return {
        "id": conv.id,
        "user": conv.user,
        "title": conv.title,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
        "lc_messages": conv.lc_messages,
        "ui_messages": conv.ui_messages,
    }


def list_conversations(user: str = DEFAULT_USER) -> list[dict[str, Any]]:
    """Return conversation metadata for *user*, sorted by updated_at descending."""
    with get_read_session() as session:
        try:
            rows = (
                session.execute(
                    select(Conversation)
                    .where(Conversation.user == user)
                    .order_by(Conversation.updated_at.desc())
                )
                .scalars()
                .all()
            )
        except SQLAlchemyError:
            logger.exception("Failed to list conversations", extra={"user": user})
            return []

    return [
        {
            "id": c.id,
            "title": c.title,
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat(),
        }
        for c in rows
    ]


def delete_conversation(session_id: str, user: str = DEFAULT_USER) -> bool:  # noqa: ARG001
    """Delete a conversation by id.

    ``user`` is accepted for API symmetry with the other functions but not
    yet enforced as an ownership check (no auth module to trust it against
    yet) — once one exists, add a ``.where(Conversation.user == user)``
    clause here to prevent deleting another user's conversation.
    """
    try:
        _validate_session_id(session_id)
    except ValueError:
        return False

    with get_write_session() as session:
        try:
            result = session.execute(
                sa_delete(Conversation).where(Conversation.id == session_id)
            )
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            logger.exception(
                "Failed to delete conversation", extra={"session_id": session_id}
            )
            return False

    return result.rowcount > 0


def update_title(session_id: str, title: str, user: str = DEFAULT_USER) -> bool:  # noqa: ARG001
    """Rename a conversation. See delete_conversation's note on ``user``."""
    try:
        _validate_session_id(session_id)
    except ValueError:
        return False

    with get_write_session() as session:
        try:
            conv = session.get(Conversation, session_id)
            if conv is None:
                return False
            conv.title = title
            conv.updated_at = datetime.now(timezone.utc)
            session.commit()
            return True
        except SQLAlchemyError:
            session.rollback()
            logger.exception(
                "Failed to update conversation title", extra={"session_id": session_id}
            )
            return False