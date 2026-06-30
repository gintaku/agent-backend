"""Tests for backend/rag/ingestor.py."""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Shared fake embeddings (no API call — returns fixed-length vectors)
# ---------------------------------------------------------------------------

class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 128 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 128


@pytest.fixture()
def vs():
    """Isolated in-memory Chroma vectorstore per test."""
    return Chroma(
        collection_name=f"test_{uuid.uuid4().hex}",
        embedding_function=_FakeEmbeddings(),
    )


# ---------------------------------------------------------------------------
# _doc_id
# ---------------------------------------------------------------------------

def test_doc_id_is_deterministic():
    from rag.ingestor import _doc_id

    assert _doc_id("file.txt", 0) == _doc_id("file.txt", 0)


def test_doc_id_differs_by_index():
    from rag.ingestor import _doc_id

    assert _doc_id("file.txt", 0) != _doc_id("file.txt", 1)


def test_doc_id_differs_by_source():
    from rag.ingestor import _doc_id

    assert _doc_id("a.txt", 0) != _doc_id("b.txt", 0)


def test_doc_id_is_valid_hex():
    from rag.ingestor import _doc_id

    result = _doc_id("source", 0)
    int(result, 16)  # raises ValueError if not valid hex


# ---------------------------------------------------------------------------
# _load_text_file
# ---------------------------------------------------------------------------

def test_load_text_file(tmp_path: Path):
    from rag.ingestor import _load_text_file

    f = tmp_path / "hello.txt"
    f.write_text("hello world", encoding="utf-8")
    assert _load_text_file(f) == "hello world"


def test_load_text_file_handles_non_utf8(tmp_path: Path):
    from rag.ingestor import _load_text_file

    f = tmp_path / "latin.txt"
    f.write_bytes(b"\xff\xfe hello")  # invalid UTF-8 bytes
    result = _load_text_file(f)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _split_and_tag
# ---------------------------------------------------------------------------

def test_split_and_tag_returns_documents():
    from rag.ingestor import _split_and_tag

    docs = _split_and_tag("word " * 300, "src.txt", "text")
    assert len(docs) >= 1
    assert all(isinstance(d, Document) for d in docs)


def test_split_and_tag_metadata():
    from rag.ingestor import _split_and_tag

    docs = _split_and_tag("word " * 300, "my_source.txt", "text")
    for i, doc in enumerate(docs):
        assert doc.metadata["source"] == "my_source.txt"
        assert doc.metadata["type"] == "text"
        assert doc.metadata["chunk_index"] == i


def test_split_and_tag_empty_text():
    from rag.ingestor import _split_and_tag

    docs = _split_and_tag("", "empty.txt", "text")
    assert docs == []


# ---------------------------------------------------------------------------
# ingest_file — .txt
# ---------------------------------------------------------------------------

def test_ingest_txt_file(tmp_path: Path, vs: Chroma):
    from rag import ingestor

    f = tmp_path / "sample.txt"
    f.write_text("The quick brown fox. " * 60, encoding="utf-8")

    with patch.object(ingestor, "get_vectorstore", return_value=vs):
        count = ingestor.ingest_file(f)

    assert count >= 1
    results = vs.similarity_search("quick brown fox", k=3)
    assert len(results) >= 1


def test_ingest_md_file(tmp_path: Path, vs: Chroma):
    from rag import ingestor

    f = tmp_path / "readme.md"
    f.write_text("# Title\n\nSome markdown content.\n" * 30, encoding="utf-8")

    with patch.object(ingestor, "get_vectorstore", return_value=vs):
        count = ingestor.ingest_file(f)

    assert count >= 1


def test_ingest_unsupported_extension_raises(tmp_path: Path):
    from rag.ingestor import ingest_file

    f = tmp_path / "data.csv"
    f.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        ingest_file(f)


# ---------------------------------------------------------------------------
# ingest_file — deduplication
# ---------------------------------------------------------------------------

def test_ingest_deduplication(tmp_path: Path, vs: Chroma):
    """Re-ingesting the same file replaces existing chunks, not duplicates them."""
    from rag import ingestor

    f = tmp_path / "dup.txt"
    f.write_text("Content version 1. " * 60, encoding="utf-8")

    with patch.object(ingestor, "get_vectorstore", return_value=vs):
        count1 = ingestor.ingest_file(f)
        count2 = ingestor.ingest_file(f)

    source = str(f.resolve())
    stored = vs._collection.get(where={"source": source})
    # Should have count2 chunks, not count1 + count2
    assert len(stored["ids"]) == count2


# ---------------------------------------------------------------------------
# ingest_url
# ---------------------------------------------------------------------------

def test_ingest_url(vs: Chroma):
    from rag import ingestor

    fake_html = (
        "<html><body><p>"
        + "word " * 300
        + "</p></body></html>"
    )
    mock_resp = MagicMock()
    mock_resp.text = fake_html
    mock_resp.raise_for_status = MagicMock()

    with (
        patch("rag.ingestor.requests.get", return_value=mock_resp),
        patch.object(ingestor, "get_vectorstore", return_value=vs),
    ):
        count = ingestor.ingest_url("https://example.com/article")

    assert count >= 1
    stored = vs._collection.get(where={"source": "https://example.com/article"})
    assert len(stored["ids"]) >= 1


def test_ingest_url_metadata_type(vs: Chroma):
    from rag import ingestor

    fake_html = "<html><body><p>" + "word " * 300 + "</p></body></html>"
    mock_resp = MagicMock()
    mock_resp.text = fake_html
    mock_resp.raise_for_status = MagicMock()

    with (
        patch("rag.ingestor.requests.get", return_value=mock_resp),
        patch.object(ingestor, "get_vectorstore", return_value=vs),
    ):
        ingestor.ingest_url("https://example.com/page")

    stored = vs._collection.get(
        where={"source": "https://example.com/page"},
        include=["metadatas"],
    )
    for meta in stored["metadatas"]:
        assert meta["type"] == "url"


# ---------------------------------------------------------------------------
# delete_source
# ---------------------------------------------------------------------------

def test_delete_source_removes_chunks(tmp_path: Path, vs: Chroma):
    from rag import ingestor

    f = tmp_path / "to_delete.txt"
    f.write_text("Some content to delete. " * 50, encoding="utf-8")
    source = str(f.resolve())

    with patch.object(ingestor, "get_vectorstore", return_value=vs):
        ingestor.ingest_file(f)
        ingestor.delete_source(source)

    remaining = vs._collection.get(where={"source": source})
    assert len(remaining["ids"]) == 0


def test_delete_source_no_error_when_empty(vs: Chroma):
    """Deleting a non-existent source should not raise."""
    from rag import ingestor

    with patch.object(ingestor, "get_vectorstore", return_value=vs):
        ingestor.delete_source("does_not_exist.txt")  # must not raise
