"""Tests for tools/web_scrape.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(html: str, status_code: int = 200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = html
    mock.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests
        http_err = requests.exceptions.HTTPError(response=mock)
        mock.raise_for_status.side_effect = http_err

    # scrape_url() walks redirects manually:
    #   if resp.is_redirect or resp.is_permanent_redirect: ...
    # A bare MagicMock auto-creates any unset attribute on access, and that
    # auto-created attribute is truthy — so without pinning these explicitly
    # to False, every mocked response looks like a redirect, the code follows
    # a fake "Location" header (itself just another auto-created MagicMock,
    # not a real URL string), and blows up deep inside URL parsing instead of
    # failing the test with a clear assertion.
    mock.is_redirect = False
    mock.is_permanent_redirect = False

    return mock


def _mock_public_dns():
    """Patch socket.getaddrinfo to resolve to a public (non-blocked) IP.

    scrape_url()'s SSRF guard (_validate_url) resolves the hostname via a
    real socket.getaddrinfo() call *before* requests.get() is ever invoked.
    Patching only requests.get therefore doesn't make these tests hermetic —
    resolution for a fictitious host like "unreachable.example.com" hits the
    real network/DNS resolver and fails there, with an error message that
    has nothing to do with what the test is actually trying to exercise.
    This patches DNS resolution itself so _validate_url passes and control
    reaches the mocked requests.get() call, regardless of network access or
    which specific hostname is used.
    """
    return patch(
        "tools.web_scrape.socket.getaddrinfo",
        return_value=[(2, 1, 6, "", ("93.184.216.34", 0))],  # public IP
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_scrape_returns_plain_text():
    from tools.web_scrape import scrape_url

    html = "<html><body><p>Hello, world!</p></body></html>"
    with (
        _mock_public_dns(),
        patch("tools.web_scrape.requests.get", return_value=_make_response(html)),
    ):
        result = scrape_url.invoke({"url": "https://example.com"})

    assert "Hello, world!" in result


def test_scrape_strips_script_tags():
    from tools.web_scrape import scrape_url

    html = (
        "<html><body>"
        "<script>alert('xss')</script>"
        "<p>Clean content</p>"
        "</body></html>"
    )
    with (
        _mock_public_dns(),
        patch("tools.web_scrape.requests.get", return_value=_make_response(html)),
    ):
        result = scrape_url.invoke({"url": "https://example.com"})

    assert "xss" not in result
    assert "Clean content" in result


def test_scrape_strips_style_tags():
    from tools.web_scrape import scrape_url

    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><p>Visible</p></body></html>"
    )
    with (
        _mock_public_dns(),
        patch("tools.web_scrape.requests.get", return_value=_make_response(html)),
    ):
        result = scrape_url.invoke({"url": "https://example.com"})

    assert "color:red" not in result
    assert "Visible" in result


def test_scrape_truncates_long_content():
    from tools.web_scrape import scrape_url, _MAX_CHARS

    long_text = "A" * (_MAX_CHARS + 500)
    html = f"<html><body><p>{long_text}</p></body></html>"
    with (
        _mock_public_dns(),
        patch("tools.web_scrape.requests.get", return_value=_make_response(html)),
    ):
        result = scrape_url.invoke({"url": "https://example.com"})

    assert len(result) <= _MAX_CHARS + 100  # small buffer for truncation marker
    assert "truncated" in result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_scrape_timeout_returns_error():
    from tools.web_scrape import scrape_url
    import requests

    with (
        _mock_public_dns(),
        patch(
            "tools.web_scrape.requests.get",
            side_effect=requests.exceptions.Timeout,
        ),
    ):
        result = scrape_url.invoke({"url": "https://slow.example.com"})

    assert result.startswith("Error")
    assert "timed out" in result


def test_scrape_connection_error_returns_error():
    from tools.web_scrape import scrape_url
    import requests

    with (
        _mock_public_dns(),
        patch(
            "tools.web_scrape.requests.get",
            side_effect=requests.exceptions.ConnectionError("no route"),
        ),
    ):
        result = scrape_url.invoke({"url": "https://unreachable.example.com"})

    assert result.startswith("Error")
    assert "connect" in result.lower()


def test_scrape_http_error_returns_error():
    from tools.web_scrape import scrape_url

    bad_response = _make_response("<html/>", status_code=404)
    with (
        _mock_public_dns(),
        patch("tools.web_scrape.requests.get", return_value=bad_response),
    ):
        result = scrape_url.invoke({"url": "https://example.com/missing"})

    assert result.startswith("Error")
    assert "404" in result


def test_scrape_tool_name():
    from tools.web_scrape import scrape_url
    assert scrape_url.name == "scrape_url"