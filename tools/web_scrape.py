"""Web scraping tool — fetches a URL and returns clean plain text."""
from __future__ import annotations

import html2text
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

_REQUEST_TIMEOUT = 15  # seconds
_MAX_CHARS = 100_000


@tool
def scrape_url(url: str) -> str:
    """Fetch a web page and return its main text content.

    Strips HTML tags and scripts; converts the page to readable plain text.
    Returns up to 8 000 characters or an error string.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; AIAgent/1.0; +https://github.com/example)"
            )
        }
        response = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
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
