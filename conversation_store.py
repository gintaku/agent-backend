"""Persistence layer for conversation history.

Each conversation is stored as a JSON file:
  {conversations_dir}/{session_id}.json

Schema:
{
  "id": "<session_id>",
  "title": "First user message...",
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "lc_messages": [...],   # serialized LangChain messages for context restoration
  "ui_messages": [...]    # raw frontend message dicts for display restoration
}
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger("app")

# Session IDs must be alphanumeric + hyphens/underscores, max 128 chars.
# This prevents path traversal attacks when building file paths.
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(f"Invalid session_id: {session_id!r}")


def _conversations_dir() -> Path:
    from config import get_settings
    d = Path(get_settings().conversations_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _conv_path(session_id: str) -> Path:
    _validate_session_id(session_id)
    return _conversations_dir() / f"{session_id}.json"


# ---------------------------------------------------------------------------
# Serialization helpers
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
) -> None:
    path = _conv_path(session_id)
    now = datetime.now(timezone.utc).isoformat()

    # Preserve created_at if file already exists
    created_at = now
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            created_at = existing.get("created_at", now)
        except Exception:
            logger.exception(
                "Could not read existing conversation file to preserve created_at "
                "— overwriting with a fresh timestamp",
                extra={"session_id": session_id},
            )

    data = {
        "id": session_id,
        "title": title,
        "created_at": created_at,
        "updated_at": now,
        "lc_messages": lc_messages,
        "ui_messages": ui_messages,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_conversation(session_id: str) -> dict[str, Any] | None:
    try:
        path = _conv_path(session_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception(
            "Failed to load conversation file — treating as missing",
            extra={"session_id": session_id},
        )
        return None


def list_conversations() -> list[dict[str, Any]]:
    """Return conversation metadata sorted by updated_at descending."""
    results = []
    d = _conversations_dir()
    for p in d.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            results.append({
                "id": data.get("id", p.stem),
                "title": data.get("title", "Untitled"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
            })
        except Exception:
            logger.exception(
                "Skipping unreadable conversation file in listing",
                extra={"file": str(p)},
            )
            continue
    results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return results


def delete_conversation(session_id: str) -> bool:
    try:
        path = _conv_path(session_id)
    except ValueError:
        return False
    if path.exists():
        path.unlink()
        return True
    return False


def update_title(session_id: str, title: str) -> bool:
    try:
        path = _conv_path(session_id)
    except ValueError:
        return False
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["title"] = title
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        logger.exception("Failed to update conversation title", extra={"session_id": session_id})
        return False
