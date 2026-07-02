"""Tests for backend/rag/retriever.py."""
from __future__ import annotations

from unittest.mock import patch

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# retrieve — empty knowledge base
# ---------------------------------------------------------------------------

def test_retrieve_empty_kb_returns_empty_string(vs):
    from rag import retriever

    with patch.object(retriever, "get_read_vectorstore", return_value=vs):
        result = retriever.retrieve("any query")

    assert result == ""


# ---------------------------------------------------------------------------
# retrieve — single document
# ---------------------------------------------------------------------------

def test_retrieve_includes_chunk_text(vs):
    from rag import retriever

    vs.add_documents([
        Document(
            page_content="Paris is the capital of France.",
            metadata={"source": "geo.txt"},
        )
    ])

    with patch.object(retriever, "get_read_vectorstore", return_value=vs):
        result = retriever.retrieve("capital of France")

    assert "Paris is the capital of France." in result


def test_retrieve_includes_source_label(vs):
    from rag import retriever

    vs.add_documents([
        Document(
            page_content="Some content.",
            metadata={"source": "knowledge/geo.txt"},
        )
    ])

    with patch.object(retriever, "get_read_vectorstore", return_value=vs):
        result = retriever.retrieve("content")

    assert "[Source: knowledge/geo.txt]" in result


# ---------------------------------------------------------------------------
# retrieve — multiple chunks
# ---------------------------------------------------------------------------

def test_retrieve_respects_top_k(vs):
    from rag import retriever

    docs = [
        Document(page_content=f"Fact {i}.", metadata={"source": "facts.txt"})
        for i in range(10)
    ]
    vs.add_documents(docs)

    with (
        patch.object(retriever, "get_read_vectorstore", return_value=vs),
        patch("rag.retriever.get_settings") as mock_settings,
    ):
        mock_settings.return_value.rag_top_k = 3
        mock_settings.return_value.rag_score_threshold = 1.0
        result = retriever.retrieve("fact")

    # At most 3 chunks means at most 2 "---" separators
    assert result.count("---") <= 2


def test_retrieve_separates_chunks(vs):
    from rag import retriever

    vs.add_documents([
        Document(page_content="First chunk.", metadata={"source": "a.txt"}),
        Document(page_content="Second chunk.", metadata={"source": "b.txt"}),
    ])

    with (
        patch.object(retriever, "get_read_vectorstore", return_value=vs),
        patch("rag.retriever.get_settings") as mock_settings,
    ):
        mock_settings.return_value.rag_top_k = 2
        mock_settings.return_value.rag_score_threshold = 1.0
        result = retriever.retrieve("chunk")

    assert "---" in result


# ---------------------------------------------------------------------------
# retrieve — score threshold filtering
# ---------------------------------------------------------------------------

def test_retrieve_filters_by_score_threshold(vs):
    """Chunks scoring above the configured threshold must be dropped."""
    from rag import retriever

    vs.add_documents([
        Document(page_content="Relevant chunk.", metadata={"source": "a.txt"}),
    ])

    with (
        patch.object(retriever, "get_read_vectorstore", return_value=vs),
        patch("rag.retriever.get_settings") as mock_settings,
    ):
        # The fake vectorstore always scores 0.0 — a negative threshold means
        # nothing can pass the "<= threshold" filter.
        mock_settings.return_value.rag_top_k = 5
        mock_settings.return_value.rag_score_threshold = -1.0
        result = retriever.retrieve("chunk")

    assert result == ""


# ---------------------------------------------------------------------------
# retrieve — error resilience
# ---------------------------------------------------------------------------

def test_retrieve_returns_empty_string_on_exception():
    from rag import retriever

    class _BrokenVectorStore:
        def similarity_search_with_score(self, *a, **kw):
            raise RuntimeError("DB offline")

    with patch.object(retriever, "get_read_vectorstore", return_value=_BrokenVectorStore()):
        result = retriever.retrieve("query")

    assert result == ""


def test_retrieve_missing_source_metadata(vs):
    from rag import retriever

    # Document with no "source" key in metadata
    vs.add_documents([
        Document(page_content="Orphan chunk.", metadata={})
    ])

    with patch.object(retriever, "get_read_vectorstore", return_value=vs):
        result = retriever.retrieve("orphan")

    # Should still return content with a fallback source label
    assert "Orphan chunk." in result
    assert "[Source: unknown]" in result
