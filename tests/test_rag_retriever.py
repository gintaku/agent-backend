"""Tests for backend/rag/retriever.py."""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Shared fake embeddings (no API call)
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
# retrieve — empty knowledge base
# ---------------------------------------------------------------------------

def test_retrieve_empty_kb_returns_empty_string(vs: Chroma):
    from rag import retriever

    with patch.object(retriever, "get_vectorstore", return_value=vs):
        result = retriever.retrieve("any query")

    assert result == ""


# ---------------------------------------------------------------------------
# retrieve — single document
# ---------------------------------------------------------------------------

def test_retrieve_includes_chunk_text(vs: Chroma):
    from rag import retriever

    vs.add_documents([
        Document(
            page_content="Paris is the capital of France.",
            metadata={"source": "geo.txt"},
        )
    ])

    with patch.object(retriever, "get_vectorstore", return_value=vs):
        result = retriever.retrieve("capital of France")

    assert "Paris is the capital of France." in result


def test_retrieve_includes_source_label(vs: Chroma):
    from rag import retriever

    vs.add_documents([
        Document(
            page_content="Some content.",
            metadata={"source": "knowledge/geo.txt"},
        )
    ])

    with patch.object(retriever, "get_vectorstore", return_value=vs):
        result = retriever.retrieve("content")

    assert "[Source: knowledge/geo.txt]" in result


# ---------------------------------------------------------------------------
# retrieve — multiple chunks
# ---------------------------------------------------------------------------

def test_retrieve_respects_top_k(vs: Chroma):
    from rag import retriever

    docs = [
        Document(page_content=f"Fact {i}.", metadata={"source": "facts.txt"})
        for i in range(10)
    ]
    vs.add_documents(docs)

    with (
        patch.object(retriever, "get_vectorstore", return_value=vs),
        patch("rag.retriever.get_settings") as mock_settings,
    ):
        mock_settings.return_value.rag_top_k = 3
        result = retriever.retrieve("fact")

    # At most 3 separators means at most 3 chunks
    assert result.count("---") <= 2


def test_retrieve_separates_chunks(vs: Chroma):
    from rag import retriever

    vs.add_documents([
        Document(page_content="First chunk.", metadata={"source": "a.txt"}),
        Document(page_content="Second chunk.", metadata={"source": "b.txt"}),
    ])

    with (
        patch.object(retriever, "get_vectorstore", return_value=vs),
        patch("rag.retriever.get_settings") as mock_settings,
    ):
        mock_settings.return_value.rag_top_k = 2
        result = retriever.retrieve("chunk")

    assert "---" in result


# ---------------------------------------------------------------------------
# retrieve — error resilience
# ---------------------------------------------------------------------------

def test_retrieve_returns_empty_string_on_exception():
    from rag import retriever

    broken_vs = Chroma.__new__(Chroma)
    broken_vs.similarity_search_with_score = lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("DB offline")
    )

    with patch.object(retriever, "get_vectorstore", return_value=broken_vs):
        result = retriever.retrieve("query")

    assert result == ""


def test_retrieve_missing_source_metadata(vs: Chroma):
    from rag import retriever

    # Document with no "source" key in metadata
    vs.add_documents([
        Document(page_content="Orphan chunk.", metadata={})
    ])

    with patch.object(retriever, "get_vectorstore", return_value=vs):
        result = retriever.retrieve("orphan")

    # Should still return content with a fallback source label
    assert "Orphan chunk." in result
    assert "[Source: unknown]" in result
