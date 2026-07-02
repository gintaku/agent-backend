"""Structured JSON logging for console output (k8s / Loki friendly).

Usage
-----
Call :func:`configure_logging` once, as early as possible in the process
(main.py does this at import time, before the FastAPI app is constructed).

Every log record is emitted as a single-line JSON object to stdout, e.g.::

    {"timestamp": "...", "level": "INFO", "logger": "app", "log_type": "app",
     "message": "RAG watcher started", "session_id": "-", "watch_dir": "..."}

``log_type`` is set per-logger (see :data:`LOG_TYPES`) so a Promtail/Loki
pipeline stage can promote it to a stream label and route each kind of log
to its own bucket without touching this code again, e.g.::

    pipeline_stages:
      - json:
          expressions:
            log_type: log_type
      - labels:
          log_type:

``session_id`` is picked up automatically (when set) from the request-scoped
:data:`session_id_ctx` ContextVar, so call sites don't need to pass it via
``extra`` on every log call.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

#: Request-scoped session id, set by the WebSocket handler for the duration
#: of a turn. Any logger configured via this module will automatically
#: include it (as "-" when unset) without every call site needing extra=.
session_id_ctx: ContextVar[str | None] = ContextVar("session_id_ctx", default=None)

#: Maps logger name -> log_type label, used for Loki stream routing.
#: Extend this when adding a new logger that should land in its own bucket.
LOG_TYPES: dict[str, str] = {
    "app": "app",
    "audit.shell": "audit",
    "llm.traffic": "llm_traffic",
}

# Attributes every stdlib LogRecord already has — anything else on the
# record was passed via logging's `extra=` and should be surfaced verbatim.
_STANDARD_RECORD_ATTRS = frozenset(logging.LogRecord(
    "", 0, "", 0, "", (), None
).__dict__.keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render each LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "log_type": LOG_TYPES.get(record.name, "app"),
            "message": record.getMessage(),
        }

        # Surface any extra= fields the call site attached. These take
        # precedence over the context-var fallback below (e.g. a call site
        # that isn't running inside a request context but still knows its
        # own session_id should be able to supply it explicitly).
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in payload:
                payload[key] = value

        if "session_id" not in payload:
            payload["session_id"] = session_id_ctx.get() or "-"

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger to emit structured JSON to stdout.

    Safe to call more than once (e.g. in tests) — clears any previously
    installed handlers on the root logger first.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    # Keep uvicorn's own loggers (access/error) as-is — they already have
    # sensible defaults. We only own the root/application loggers here.
