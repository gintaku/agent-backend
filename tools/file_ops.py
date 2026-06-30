"""File-system tools scoped to WORKSPACE_DIR for safe file operations."""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.tools import tool

from config import get_settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Project root is two levels up from this file (tools/ -> backend/)
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _workspace() -> Path:
    """Return the resolved workspace root from settings.

    Relative paths in WORKSPACE_DIR are resolved against the project root
    (the directory containing .env), not the process working directory.
    """
    raw = get_settings().workspace_dir
    p = Path(raw)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p.resolve()


def _safe_path(raw: str) -> Path:
    """Resolve *raw* relative to the workspace and reject path-traversal attempts.

    Raises ValueError if the resolved path escapes the workspace directory.
    """
    workspace = _workspace()
    # Join then resolve to collapse '..' components
    target = (workspace / raw).resolve()
    try:
        target.relative_to(workspace)
    except ValueError:
        raise ValueError(
            f"Access denied: '{raw}' resolves outside the workspace directory."
        )
    return target


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def read_file(path: str) -> str:
    """Read and return the text content of a file inside the workspace.

    *path* is relative to the workspace directory (e.g. 'README.md' or 'src/app.py').
    Returns an error string if the file does not exist or cannot be decoded.
    """
    try:
        target = _safe_path(path)
        return target.read_text(encoding="utf-8")
    except ValueError as exc:
        return f"Error: {exc}"
    except FileNotFoundError:
        return f"Error: file '{path}' not found."
    except IsADirectoryError:
        return f"Error: '{path}' is a directory, not a file."
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a text file (binary content)."
    except OSError as exc:
        return f"Error reading '{path}': {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """Write *content* to a file inside the workspace, creating it if needed.

    *path* is relative to the workspace directory.
    Parent directories are created automatically.
    Returns a success message or an error string.
    """
    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{path}'."
    except ValueError as exc:
        return f"Error: {exc}"
    except OSError as exc:
        return f"Error writing '{path}': {exc}"


@tool
def list_directory(path: str = ".") -> str:
    """List the files and subdirectories inside a workspace directory.

    *path* is relative to the workspace directory; defaults to the workspace root.
    Returns a newline-separated list or an error string.
    """
    try:
        target = _safe_path(path)
        if not target.exists():
            return f"Error: directory '{path}' does not exist."
        if not target.is_dir():
            return f"Error: '{path}' is a file, not a directory."
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        if not entries:
            return f"Directory '{path}' is empty."
        lines = []
        for entry in entries:
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")
        return "\n".join(lines)
    except ValueError as exc:
        return f"Error: {exc}"
    except OSError as exc:
        return f"Error listing '{path}': {exc}"


@tool
def delete_file(path: str) -> str:
    """Delete a file inside the workspace.

    *path* is relative to the workspace directory.
    Only files can be deleted; use with caution.
    Returns a success message or an error string.
    """
    try:
        target = _safe_path(path)
        if not target.exists():
            return f"Error: '{path}' does not exist."
        if target.is_dir():
            return f"Error: '{path}' is a directory. Only files can be deleted."
        target.unlink()
        return f"Successfully deleted '{path}'."
    except ValueError as exc:
        return f"Error: {exc}"
    except OSError as exc:
        return f"Error deleting '{path}': {exc}"
