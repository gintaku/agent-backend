"""Document ingestion pipeline for the RAG knowledge base.

Supported sources
-----------------
* ``.txt`` / ``.md``  — plain text, read with UTF-8 encoding
* ``.pdf``            — text extracted via pypdf
* URL string          — fetched and converted to plain text (requests + html2text)

All text is split with a :class:`RecursiveCharacterTextSplitter` before being
upserted into ChromaDB with stable, deterministic chunk IDs so that
re-ingesting the same source replaces (rather than duplicates) existing chunks.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import html2text
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
# ---------------------------------------------------------------------------
# Markdown semantic splitter
# ---------------------------------------------------------------------------
def split_markdown_semantically(markdown_text):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )
    final_chunks = text_splitter.split_documents(md_header_splits)
    return final_chunks

from rag.chroma_client import get_vectorstore

# ---------------------------------------------------------------------------
# Ingestion manifest — tracks which files have already been ingested so the
# startup scan can skip them on subsequent runs.
# ---------------------------------------------------------------------------
_MANIFEST_PATH = Path(__file__).parent / ".ingested_manifest.json"


def _load_manifest() -> dict[str, float]:
    """Return the manifest dict mapping resolved path -> mtime."""
    try:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_manifest(manifest: dict[str, float]) -> None:
    _MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def is_file_ingested(path: Path) -> bool:
    """Return True if *path* is already ingested and has not changed since."""
    source = str(path.resolve())
    manifest = _load_manifest()
    if source not in manifest:
        return False
    try:
        return os.path.getmtime(path) == manifest[source]
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------
_CHUNK_SIZE = 1_000
_CHUNK_OVERLAP = 200
_REQUEST_TIMEOUT = 15  # seconds

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _doc_id(source: str, chunk_index: int) -> str:
    """Return a stable hex ID for a (source, chunk_index) pair."""
    return hashlib.md5(f"{source}::{chunk_index}".encode()).hexdigest()


def _delete_source(source: str) -> None:
    """Remove every existing chunk whose ``source`` metadata matches *source*."""
    vs = get_vectorstore()
    try:
        vs._collection.delete(where={"source": source})
    except Exception:
        pass  # collection may be empty — not an error


def _split_and_tag(text: str, source: str, doc_type: str) -> list[Document]:
    """Split *text* into chunks and wrap each in a :class:`Document` with metadata."""
    chunks = _splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={"source": source, "type": doc_type, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    import pypdf  # lazy import — only needed when processing PDFs

    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _load_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AIAgent-RAG/1.0)"}
    response = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.body_width = 0  # no line-wrapping

    return converter.handle(str(soup)).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_file(path: Path) -> int:
    """Ingest a file into the knowledge base.

    Parameters
    ----------
    path:
        Absolute or relative path to a ``.txt``, ``.md``, or ``.pdf`` file.

    Returns
    -------
    int
        Number of chunks added to the vectorstore.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    suffix = path.suffix.lower()
    source = str(path.resolve())

    if suffix == ".txt":
        text = _load_text_file(path)
        doc_type = "text"
        docs = _split_and_tag(text, source, doc_type)
    elif suffix == ".md":
        text = _load_text_file(path)
        doc_type = "markdown"
        # Use semantic markdown splitter
        md_chunks = split_markdown_semantically(text)
        docs = [
            Document(
                page_content=chunk.page_content,
                metadata={"source": source, "type": doc_type, "chunk_index": i, **chunk.metadata},
            )
            for i, chunk in enumerate(md_chunks)
        ]
    elif suffix == ".pdf":
        text = _load_pdf(path)
        doc_type = "pdf"
        docs = _split_and_tag(text, source, doc_type)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Supported: .txt, .md, .pdf")

    if not docs:
        return 0

    _delete_source(source)
    ids = [_doc_id(source, i) for i in range(len(docs))]
    get_vectorstore().add_documents(docs, ids=ids)

    # Update manifest with the file's current mtime
    manifest = _load_manifest()
    manifest[source] = os.path.getmtime(path)
    _save_manifest(manifest)

    return len(docs)


def ingest_url(url: str) -> int:
    """Fetch a URL and ingest its text content into the knowledge base.

    Parameters
    ----------
    url:
        Fully-qualified URL to fetch (``http://`` or ``https://``).

    Returns
    -------
    int
        Number of chunks added to the vectorstore.
    """
    text = _load_url(url)
    docs = _split_and_tag(text, url, "url")
    if not docs:
        return 0

    _delete_source(url)
    ids = [_doc_id(url, i) for i in range(len(docs))]
    get_vectorstore().add_documents(docs, ids=ids)
    return len(docs)


def delete_source(source: str) -> None:
    """Remove all chunks for *source* from the knowledge base.

    Parameters
    ----------
    source:
        The ``source`` metadata value used when the document was ingested
        (resolved file path or URL string).
    """
    _delete_source(source)
    # Remove from manifest so the file will be re-ingested if it reappears
    manifest = _load_manifest()
    manifest.pop(source, None)
    _save_manifest(manifest)
