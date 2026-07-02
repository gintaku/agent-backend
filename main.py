from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_settings, reset_settings, update_env_file
from logging_setup import configure_logging, session_id_ctx

# Configure structured JSON logging as early as possible, before any other
# module (rag.ingestor, rag.watcher, agent, ...) has a chance to log.
configure_logging(get_settings().log_level)

logger = logging.getLogger("app")

_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------------------------------------------------------------
    # Startup — ingest existing docs, then launch the file watcher
    # ---------------------------------------------------------------
    from rag.ingestor import ingest_file
    from rag.watcher import start_watcher

    settings = get_settings()
    rag_docs_dir = Path(settings.rag_docs_dir)
    rag_docs_dir.mkdir(parents=True, exist_ok=True)

    # Scan and ingest only new/changed files present in rag_docs/
    def _startup_ingest() -> None:
        from rag.ingestor import is_file_ingested
        logger.info("RAG startup: scanning for new or changed files", extra={"dir": str(rag_docs_dir)})
        for p in rag_docs_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES:
                if is_file_ingested(p):
                    logger.debug("RAG startup: already ingested, skipping", extra={"file": str(p)})
                    continue
                try:
                    n = ingest_file(p)
                    logger.info("RAG startup: ingested file", extra={"file": str(p), "chunks": n})
                except Exception:
                    logger.exception("RAG startup: failed to ingest file", extra={"file": str(p)})

    await asyncio.get_event_loop().run_in_executor(None, _startup_ingest)

    observer = await asyncio.get_event_loop().run_in_executor(None, start_watcher)

    yield

    # ---------------------------------------------------------------
    # Shutdown — stop the file watcher
    # ---------------------------------------------------------------
    observer.stop()
    observer.join()
    logger.info("RAG watcher stopped")


app = FastAPI(title="AI Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "PUT", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    model: Optional[str] = None
    cmd_mode: Optional[Literal["bypass", "permission"]] = None
    enabled_tools: Optional[list[str]] = None


class ConversationTitleUpdate(BaseModel):
    title: str


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

def _all_tool_names() -> list[str]:
    from tools import ALL_TOOLS
    return [t.name for t in ALL_TOOLS]


def _enabled_tool_names(s) -> list[str]:
    return list(s.enabled_tools) if s.enabled_tools is not None else _all_tool_names()


@app.get("/api/config")
def read_config() -> dict:
    """Return the current public configuration (API key is never exposed)."""
    s = get_settings()
    return {"model": s.model, "cmd_mode": s.cmd_mode, "enabled_tools": _enabled_tool_names(s)}


@app.get("/api/tools")
def list_tools() -> list[dict]:
    """Return all available tools with their enabled/disabled status."""
    from tools import ALL_TOOLS
    s = get_settings()
    enabled = set(_enabled_tool_names(s))
    return [{"name": t.name, "enabled": t.name in enabled} for t in ALL_TOOLS]


@app.get("/api/conversations")
def get_conversations() -> list[dict]:
    """List all saved conversations, sorted by most recently updated."""
    from conversation_store import list_conversations
    return list_conversations()


@app.delete("/api/conversations/{session_id}")
def remove_conversation(session_id: str) -> dict:
    """Delete a saved conversation and clear its in-memory history."""
    from agent import clear_history
    from conversation_store import delete_conversation
    delete_conversation(session_id)
    clear_history(session_id)
    return {"ok": True}


@app.patch("/api/conversations/{session_id}")
def rename_conversation_rest(session_id: str, body: ConversationTitleUpdate) -> dict:
    """Rename a conversation title."""
    from conversation_store import update_title
    ok = update_title(session_id, body.title.strip() or "Untitled")
    return {"ok": ok}


@app.put("/api/config")
def write_config(update: ConfigUpdate) -> dict:
    """Persist model, cmd_mode, and/or enabled_tools changes to .env, then reload settings."""
    import json as _json
    if update.model is not None:
        update_env_file("MODEL", update.model)
    if update.cmd_mode is not None:
        update_env_file("CMD_MODE", update.cmd_mode)
    if update.enabled_tools is not None:
        update_env_file("ENABLED_TOOLS", _json.dumps(update.enabled_tools))
    reset_settings()
    s = get_settings()
    return {"model": s.model, "cmd_mode": s.cmd_mode, "enabled_tools": _enabled_tool_names(s)}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

async def _run_agent_safe(
    session_id: str,
    content: str,
    ws_send: Any,
    images: list[dict] | None = None,
) -> None:
    """Run the agent, catching any unhandled exception into an error event."""
    from agent import run_agent  # local import avoids circular dependency at startup

    token = session_id_ctx.set(session_id)
    try:
        await run_agent(session_id, content, ws_send, images)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error while running agent turn")
        try:
            await ws_send({"type": "error", "content": str(exc)})
        except Exception:
            logger.warning("Failed to deliver error event — client likely disconnected")
    finally:
        session_id_ctx.reset(token)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "connected", "session_id": session_id})

    # --- Restore conversation history if a saved file exists ---
    from agent import get_history
    from conversation_store import (
        deserialize_lc_messages,
        load_conversation,
        save_conversation,
        serialize_lc_messages,
        update_title,
    )

    saved = load_conversation(session_id)
    if saved and saved.get("lc_messages"):
        history = get_history(session_id)
        if not history:  # only restore when in-memory history is empty
            history.extend(deserialize_lc_messages(saved["lc_messages"]))
        await websocket.send_json({
            "type": "history_restored",
            "messages": saved.get("ui_messages", []),
            "title": saved.get("title", ""),
        })

    async def ws_send(data: dict) -> None:
        await websocket.send_json(data)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "Received malformed WS message, ignoring",
                    extra={"session_id": session_id, "raw_preview": raw[:200]},
                )
                continue
            msg_type = msg.get("type")

            if msg_type in {"message", "token"}:
                content = msg.get("content", "")
                # Support both new `images` (array) and legacy `image` (single object)
                raw_images = msg.get("images")
                if raw_images is None:
                    single = msg.get("image")
                    raw_images = [single] if single else None
                images: list[dict] | None = raw_images if raw_images else None
                # Run agent concurrently so the receive loop stays live for
                # permission_response messages during shell permission gates.
                asyncio.create_task(_run_agent_safe(session_id, content, ws_send, images))

            elif msg_type == "permission_response":
                from tools.shell import resolve_permission
                approved = bool(msg.get("approved", False))
                resolve_permission(session_id, approved)

            elif msg_type == "save_messages":
                ui_messages = msg.get("messages", [])
                title = str(msg.get("title", "")).strip() or "New Conversation"
                lc_serialized = serialize_lc_messages(get_history(session_id))
                save_conversation(session_id, title, lc_serialized, ui_messages)

            elif msg_type == "rename_conversation":
                title = str(msg.get("title", "")).strip()
                if title:
                    update_title(session_id, title)

    except WebSocketDisconnect:
        pass
