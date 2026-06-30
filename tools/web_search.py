"""Web search tool backed by DuckDuckGo (no API key required)."""
from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

_ddg: DuckDuckGoSearchRun | None = None


def _get_ddg() -> DuckDuckGoSearchRun:
    """Return (and lazily create) the DuckDuckGoSearchRun instance."""
    global _ddg
    if _ddg is None:
        _ddg = DuckDuckGoSearchRun()
    return _ddg


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return a text summary of results.

    Does not require an API key. Returns up to ~2 000 characters of search results.
    """
    try:
        return _get_ddg().run(query)
    except Exception as exc:  # noqa: BLE001
        return f"Error performing web search: {exc}"
