"""Web scraping tool — fetches a URL and returns clean plain text."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import html2text
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

_REQUEST_TIMEOUT = 15  # seconds
_MAX_CHARS = 100_000
_ALLOWED_SCHEMES = {"http", "https"}
_MAX_REDIRECTS = 5


def _is_blocked_ip(ip_str: str) -> bool:
    """Return True if *ip_str* is loopback, private, link-local, or otherwise
    not a legitimate public destination (blocks SSRF to internal services
    and the cloud metadata endpoint 169.254.169.254)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparsable -> treat as unsafe
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_url(url: str) -> str | None:
    """Return an error string if *url* is unsafe to fetch, else None."""
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return f"Error: only http/https URLs are allowed (got scheme '{parsed.scheme}')."
    hostname = parsed.hostname
    if not hostname:
        return "Error: URL has no hostname."
    try:
        resolved = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        return f"Error: could not resolve host '{hostname}': {exc}"
    for family, _, _, _, sockaddr in resolved:
        ip_str = sockaddr[0]
        if _is_blocked_ip(ip_str):
            return f"Error: '{hostname}' resolves to a non-public address and cannot be fetched."
    return None


@tool
def scrape_url(url: str) -> str:
    """Fetch a web page and return its main text content.

    Strips HTML tags and scripts; converts the page to readable plain text.
    Only public http/https URLs are allowed — internal/private network
    addresses are blocked. Returns up to 100 000 characters or an error string.
    """
    block_reason = _validate_url(url)
    if block_reason:
        return block_reason

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; AIAgent/1.0; +https://github.com/example)"
            )
        }
        # Walk redirects manually so each hop is re-validated; requests'
        # built-in redirect following would otherwise let a public URL
        # redirect to an internal address (a classic SSRF bypass).
        current_url = url
        response = None
        for _ in range(_MAX_REDIRECTS):
            resp = requests.get(
                current_url,
                headers=headers,
                timeout=_REQUEST_TIMEOUT,
                allow_redirects=False,
            )
            if resp.is_redirect or resp.is_permanent_redirect:
                next_url = resp.headers.get("Location")
                if not next_url:
                    break
                next_block_reason = _validate_url(next_url)
                if next_block_reason:
                    return f"Error: redirect target blocked — {next_block_reason}"
                current_url = next_url
                continue
            response = resp
            break
        if response is None:
            return "Error: too many redirects."

        response.raise_for_status()

        # Strip <script> and <style> blocks before conversion
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe"]):
            tag.decompose()

        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        converter.body_width = 0  # no line wrapping

        text = converter.handle(str(soup)).strip()

        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + "\n…(content truncated)"

        return text or "(no readable content found)"

    except requests.exceptions.Timeout:
        return f"Error: request to '{url}' timed out after {_REQUEST_TIMEOUT} seconds."
    except requests.exceptions.SSLError as exc:
        return f"Error: SSL certificate verification failed for '{url}': {exc}"
    except requests.exceptions.ConnectionError as exc:
        return f"Error: could not connect to '{url}': {exc}"
    except requests.exceptions.HTTPError as exc:
        return f"Error: HTTP {exc.response.status_code} from '{url}'."
    except Exception as exc:  # noqa: BLE001
        return f"Error scraping '{url}': {exc}"
