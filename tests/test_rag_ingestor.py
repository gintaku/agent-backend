"""Tests for backend/rag/ingestor.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def _isolated_manifest(tmp_path, monkeypatch):
    """Point the ingestion manifest at a throwaway file for every test.

    ``rag.ingestor`` persists ``.ingested_manifest.json`` next to the module
    on real disk. Now that source deletion is ID-based (reconstructed from
    the manifest's recorded chunk_count — see ``_delete_source``), tests
    depend on that manifest being clean and isolated per test, not just
    incidentally reading/writing it.
    """
    from rag import ingestor

    monkeypatch.setattr(ingestor, "_MANIFEST_PATH", tmp_path / "manifest.json")


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
# manifest — load/save round-trip and back-compat
# ---------------------------------------------------------------------------

def test_manifest_round_trip(tmp_path: Path):
    from rag import ingestor

    manifest = {"a.txt": {"mtime": 123.0, "chunk_count": 4}}
    ingestor._save_manifest(manifest)
    assert ingestor._load_manifest() == manifest


def test_manifest_upgrades_legacy_mtime_only_entries(tmp_path: Path):
    """Manifests written before chunk_count existed stored a bare float."""
    from rag import ingestor

    ingestor._MANIFEST_PATH.write_text('{"old.txt": 123.0}', encoding="utf-8")
    manifest = ingestor._load_manifest()
    assert manifest == {"old.txt": {"mtime": 123.0, "chunk_count": 0}}


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

def test_ingest_txt_file(tmp_path: Path, vs):
    from rag import ingestor

    f = tmp_path / "sample.txt"
    f.write_text("The quick brown fox. " * 60, encoding="utf-8")

    with patch.object(ingestor, "get_write_vectorstore", return_value=vs):
        count = ingestor.ingest_file(f)

    assert count >= 1
    results = vs.similarity_search("quick brown fox", k=3)
    assert len(results) >= 1


def test_ingest_md_file(tmp_path: Path, vs):
    from rag import ingestor

    f = tmp_path / "readme.md"
    f.write_text("# Title\n\nSome markdown content.\n" * 30, encoding="utf-8")

    with patch.object(ingestor, "get_write_vectorstore", return_value=vs):
        count = ingestor.ingest_file(f)

    assert count >= 1


def test_ingest_unsupported_extension_raises(tmp_path: Path):
    from rag.ingestor import ingest_file

    f = tmp_path / "data.csv"
    f.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        ingest_file(f)


# ---------------------------------------------------------------------------
# ingest_file — manifest bookkeeping
# ---------------------------------------------------------------------------

def test_ingest_file_records_chunk_count_in_manifest(tmp_path: Path, vs):
    from rag import ingestor

    f = tmp_path / "sample.txt"
    f.write_text("The quick brown fox. " * 60, encoding="utf-8")

    with patch.object(ingestor, "get_write_vectorstore", return_value=vs):
        count = ingestor.ingest_file(f)

    manifest = ingestor._load_manifest()
    source = str(f.resolve())
    assert manifest[source]["chunk_count"] == count
    assert manifest[source]["mtime"] == pytest.approx(f.stat().st_mtime)


# ---------------------------------------------------------------------------
# ingest_file — deduplication
# ---------------------------------------------------------------------------

def test_ingest_deduplication(tmp_path: Path, vs):
    """Re-ingesting the same file replaces existing chunks, not duplicates them."""
    from rag import ingestor

    f = tmp_path / "dup.txt"
    f.write_text("Content version 1. " * 60, encoding="utf-8")

    with patch.object(ingestor, "get_write_vectorstore", return_value=vs):
        count1 = ingestor.ingest_file(f)
        count2 = ingestor.ingest_file(f)

    # Same content -> same chunk count both times, and no leftover duplicates
    # in the underlying store (same deterministic IDs get overwritten).
    assert count1 == count2
    assert len(vs._docs) == count2


def test_ingest_deduplication_removes_stale_trailing_chunks(tmp_path: Path, vs):
    """Re-ingesting with *fewer* chunks must not leave old chunks behind.

    This is exactly the case the manifest's chunk_count exists to handle:
    without it, ID-based deletion has no way to know how many old chunk IDs
    to remove when a source shrinks.
    """
    from rag import ingestor

    f = tmp_path / "shrink.txt"
    f.write_text("Content. " * 400, encoding="utf-8")  # many chunks

    with patch.object(ingestor, "get_write_vectorstore", return_value=vs):
        count1 = ingestor.ingest_file(f)
        f.write_text("Short content.", encoding="utf-8")  # one chunk
        count2 = ingestor.ingest_file(f)

    assert count2 < count1
    assert len(vs._docs) == count2


# ---------------------------------------------------------------------------
# ingest_url
# ---------------------------------------------------------------------------

def test_ingest_url(vs):
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
        patch.object(ingestor, "get_write_vectorstore", return_value=vs),
    ):
        count = ingestor.ingest_url("https://example.com/article")

    assert count >= 1
    assert len(vs._docs) == count


def test_ingest_url_metadata_type(vs):
    from rag import ingestor

    fake_html = "<html><body><p>" + "word " * 300 + "</p></body></html>"
    mock_resp = MagicMock()
    mock_resp.text = fake_html
    mock_resp.raise_for_status = MagicMock()

    with (
        patch("rag.ingestor.requests.get", return_value=mock_resp),
        patch.object(ingestor, "get_write_vectorstore", return_value=vs),
    ):
        ingestor.ingest_url("https://example.com/page")

    for doc in vs._docs.values():
        assert doc.metadata["type"] == "url"


# ---------------------------------------------------------------------------
# delete_source
# ---------------------------------------------------------------------------

def test_delete_source_removes_chunks(tmp_path: Path, vs):
    from rag import ingestor

    f = tmp_path / "to_delete.txt"
    f.write_text("Some content to delete. " * 50, encoding="utf-8")
    source = str(f.resolve())

    with patch.object(ingestor, "get_write_vectorstore", return_value=vs):
        ingestor.ingest_file(f)
        ingestor.delete_source(source)

    assert len(vs._docs) == 0
    assert source not in ingestor._load_manifest()


def test_delete_source_no_error_when_empty(vs):
    """Deleting a non-existent source should not raise."""
    from rag import ingestor

    with patch.object(ingestor, "get_write_vectorstore", return_value=vs):
        ingestor.delete_source("does_not_exist.txt")  # must not raise
