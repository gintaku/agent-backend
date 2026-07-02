"""Shell command tool with an optional WebSocket permission gate."""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

from langchain_core.tools import tool

from config import get_settings

#: Dedicated audit trail for every command this tool is asked to run, whether
#: it executes, is blocked, or is denied. Kept on its own logger name (see
#: logging_setup.LOG_TYPES) so it can be routed to a separate, longer-retention
#: Loki bucket independent of general app logs.
audit_logger = logging.getLogger("audit.shell")

# ---------------------------------------------------------------------------
# Context variables — injected by the WebSocket handler before running agent
# ---------------------------------------------------------------------------

#: Identifies the active WebSocket session so the pending-permission table can
#: be keyed correctly.
session_id_var: ContextVar[str | None] = ContextVar("shell_session_id", default=None)

#: Async callable that sends a JSON-serialisable dict to the WebSocket client.
ws_send_var: ContextVar[
    Callable[[dict[str, Any]], Awaitable[None]] | None
] = ContextVar("shell_ws_send", default=None)

# ---------------------------------------------------------------------------
# Pending-permission table  {session_id: (event, result_holder)}
# ---------------------------------------------------------------------------

_pending: dict[str, tuple[asyncio.Event, dict[str, bool]]] = {}

# ---------------------------------------------------------------------------
# Read-only command detection
# ---------------------------------------------------------------------------

_READ_ONLY_SINGLE = frozenset(
    {
        "dir", "ls", "pwd", "whoami", "hostname",
        "where", "which", "date", "time", "ver", "uname",
    }
)

_READ_ONLY_PREFIXES: tuple[str, ...] = (
    "dir ",
    "ls ",
    "cat ",
    "type ",
    "echo ",
    "git status",
    "git log",
    "git diff",
    "git branch",
    "git show",
    "python --version",
    "python3 --version",
    "pip list",
    "pip show ",
    "pip --version",
    "where ",
    "which ",
    "whoami",
    "uname",
)

# Any of these characters let a single string smuggle a second, unrelated
# command past a prefix check (e.g. "echo hi; rm -rf workspace"). If any
# appear, the command is NEVER treated as read-only, regardless of how it
# starts — it always requires permission.
_SHELL_METACHARACTERS = (";", "&&", "||", "|", "`", "$(", ">", "<", "\n", "&")


def _has_shell_metacharacters(command: str) -> bool:
    return any(ch in command for ch in _SHELL_METACHARACTERS)


def _is_read_only(command: str) -> bool:
    """Return True when *command* is considered safe to run without permission.

    Only ever True when the command has no shell metacharacters that could
    chain in a second command AND matches a known-safe single command/prefix.
    """
    if _has_shell_metacharacters(command):
        return False
    cmd_lower = command.strip().lower()
    first_word = cmd_lower.split()[0] if cmd_lower.split() else ""
    if first_word in _READ_ONLY_SINGLE:
        return True
    return any(cmd_lower.startswith(prefix) for prefix in _READ_ONLY_PREFIXES)


# Commands that are destructive enough to block outright, even in
# CMD_MODE=bypass. This is a defense-in-depth backstop, not a substitute for
# sandboxing — bypass mode should generally not be used outside trusted,
# disposable environments.
_HARD_BLOCKED_PATTERNS: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",  # fork bomb
    "> /dev/sd",
    "chmod -r 777 /",
    "shutdown",
    "reboot",
)


def _is_hard_blocked(command: str) -> bool:
    cmd_lower = command.strip().lower()
    return any(pattern in cmd_lower for pattern in _HARD_BLOCKED_PATTERNS)


# ---------------------------------------------------------------------------
# Permission resolution (called by the WebSocket handler)
# ---------------------------------------------------------------------------

def resolve_permission(session_id: str, approved: bool) -> bool:
    """Resolve a pending permission request for *session_id*.

    Called by the WebSocket handler when the frontend sends a
    ``permission_response`` message.

    Returns ``True`` if a pending request existed, ``False`` otherwise.
    """
    entry = _pending.get(session_id)
    if entry is None:
        return False
    event, result_holder = entry
    result_holder["approved"] = approved
    event.set()
    return True


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
async def run_command(command: str) -> str:
    """Execute a shell command and return its stdout/stderr.
    Use this tool to run commands that interact with the file system, query system information, or
    perform other side-effecting operations.

    Read-only commands (dir, ls, cat, pwd, git status, …) run immediately.
    All other commands are gated by CMD_MODE:
      - bypass     → execute immediately
      - permission → ask the user via the UI before executing

    Returns the command output (up to 10 000 characters) or an error string.
    """
    settings = get_settings()
    sid = session_id_var.get() or "-"

    if _is_hard_blocked(command):
        audit_logger.warning(
            "shell_command_blocked",
            extra={"session_id": sid, "command": command},
        )
        return "Error: this command matches a blocked destructive pattern and cannot be executed."

    needs_permission = settings.cmd_mode == "permission" and not _is_read_only(command)

    if needs_permission:
        sid = session_id_var.get()
        ws_send = ws_send_var.get()

        if ws_send is None or sid is None:
            audit_logger.warning(
                "shell_permission_unavailable",
                extra={"session_id": sid or "-", "command": command},
            )
            return (
                "Error: command requires explicit permission but no active WebSocket "
                "session is available. Set CMD_MODE=bypass to run without prompting."
            )

        event: asyncio.Event = asyncio.Event()
        result_holder: dict[str, bool] = {}
        _pending[sid] = (event, result_holder)

        audit_logger.info(
            "shell_permission_requested",
            extra={"session_id": sid, "command": command},
        )

        try:
            await ws_send({"type": "permission_request", "command": command})
            await asyncio.wait_for(event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            audit_logger.warning(
                "shell_permission_timeout",
                extra={"session_id": sid, "command": command},
            )
            return "Error: permission request timed out (60 s)."
        finally:
            _pending.pop(sid, None)

        approved = result_holder.get("approved", False)
        audit_logger.info(
            "shell_permission_granted" if approved else "shell_permission_denied",
            extra={"session_id": sid, "command": command},
        )
        if not approved:
            return "Command denied by user."

    # -----------------------------------------------------------------------
    # Execute
    # -----------------------------------------------------------------------
    import subprocess

    def _run() -> tuple[str, int | None]:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                stdin=subprocess.DEVNULL,
            )
            output = (result.stdout or "") + (result.stderr or "")
            output = output.strip()
            if len(output) > 10_000:
                output = output[:10_000] + "\n…(output truncated)"
            return output or "(no output)", result.returncode
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 30 seconds.", None
        except Exception as exc:  # noqa: BLE001
            return f"Error running command: {exc}", None

    start = time.monotonic()
    try:
        output, returncode = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001
        audit_logger.exception(
            "shell_command_error",
            extra={"session_id": sid, "command": command},
        )
        return f"Error running command: {exc}"

    duration_ms = round((time.monotonic() - start) * 1000, 1)
    # Deliberately omit the raw output from the audit record — it may contain
    # secrets echoed by the command (env vars, file contents, API responses).
    # The command itself, exit status, and output size are enough for an
    # audit trail without duplicating a second copy of sensitive data.
    audit_logger.info(
        "shell_command_executed",
        extra={
            "session_id": sid,
            "command": command,
            "returncode": returncode,
            "output_chars": len(output),
            "duration_ms": duration_ms,
        },
    )
    return output
