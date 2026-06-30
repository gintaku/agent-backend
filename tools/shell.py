"""Shell command tool with an optional WebSocket permission gate."""
from __future__ import annotations

import asyncio
import sys
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

from langchain_core.tools import tool

from config import get_settings

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


def _is_read_only(command: str) -> bool:
    """Return True when *command* is considered safe to run without permission."""
    cmd_lower = command.strip().lower()
    first_word = cmd_lower.split()[0] if cmd_lower.split() else ""
    if first_word in _READ_ONLY_SINGLE:
        return True
    return any(cmd_lower.startswith(prefix) for prefix in _READ_ONLY_PREFIXES)


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
    needs_permission = settings.cmd_mode == "permission" and not _is_read_only(command)

    if needs_permission:
        sid = session_id_var.get()
        ws_send = ws_send_var.get()

        if ws_send is None or sid is None:
            return (
                "Error: command requires explicit permission but no active WebSocket "
                "session is available. Set CMD_MODE=bypass to run without prompting."
            )

        event: asyncio.Event = asyncio.Event()
        result_holder: dict[str, bool] = {}
        _pending[sid] = (event, result_holder)

        try:
            await ws_send({"type": "permission_request", "command": command})
            await asyncio.wait_for(event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            return "Error: permission request timed out (60 s)."
        finally:
            _pending.pop(sid, None)

        if not result_holder.get("approved", False):
            return "Command denied by user."

    # -----------------------------------------------------------------------
    # Execute
    # -----------------------------------------------------------------------
    import subprocess

    def _run() -> str:
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
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 30 seconds."
        except Exception as exc:  # noqa: BLE001
            return f"Error running command: {exc}"

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001
        return f"Error running command: {exc}"
