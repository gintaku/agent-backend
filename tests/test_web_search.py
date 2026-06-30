"""Tests for tools/web_search.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_web_search_returns_ddg_results():
    from tools.web_search import web_search

    mock_ddg = MagicMock()
    mock_ddg.run.return_value = "Python 3.13 released in October 2024."
    with patch("tools.web_search._get_ddg", return_value=mock_ddg):
        result = web_search.invoke({"query": "latest Python version"})

    assert result == "Python 3.13 released in October 2024."
    mock_ddg.run.assert_called_once_with("latest Python version")


def test_web_search_passes_query_unchanged():
    from tools.web_search import web_search

    mock_ddg = MagicMock()
    mock_ddg.run.return_value = "some result"
    with patch("tools.web_search._get_ddg", return_value=mock_ddg):
        web_search.invoke({"query": "  FastAPI vs Flask  "})

    mock_ddg.run.assert_called_once_with("  FastAPI vs Flask  ")


def test_web_search_returns_error_on_exception():
    from tools.web_search import web_search

    mock_ddg = MagicMock()
    mock_ddg.run.side_effect = RuntimeError("network failure")
    with patch("tools.web_search._get_ddg", return_value=mock_ddg):
        result = web_search.invoke({"query": "anything"})

    assert result.startswith("Error")
    assert "network failure" in result


def test_web_search_tool_name():
    from tools.web_search import web_search
    assert web_search.name == "web_search"
