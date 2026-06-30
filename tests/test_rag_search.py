"""Tests for backend/tools/rag_search.py."""
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
# search_knowledge_base — empty KB
# ---------------------------------------------------------------------------

def test_search_kb_empty_returns_no_results_message(vs: Chroma):
    from rag import retriever
    from tools.rag_search import search_knowledge_base

    with patch.object(retriever, "get_vectorstore", return_value=vs):
        result = search_knowledge_base.invoke({"query": "anything"})

    assert "No relevant information" in result


# ---------------------------------------------------------------------------
# search_knowledge_base — with documents
# ---------------------------------------------------------------------------

def test_search_kb_returns_matching_content(vs: Chroma):
    from rag import retriever
    from tools.rag_search import search_knowledge_base

    vs.add_documents([
        Document(
            page_content="The speed of light is approximately 300,000 km/s.",
            metadata={"source": "physics.txt"},
        )
    ])

    with patch.object(retriever, "get_vectorstore", return_value=vs):
        result = search_knowledge_base.invoke({"query": "speed of light"})

    assert "300,000 km/s" in result
    assert "[Source: physics.txt]" in result


def test_search_kb_returns_multiple_chunks(vs: Chroma):
    from rag import retriever
    from tools.rag_search import search_knowledge_base

    vs.add_documents([
        Document(page_content="Chunk one content.", metadata={"source": "a.txt"}),
        Document(page_content="Chunk two content.", metadata={"source": "b.txt"}),
    ])

    with patch.object(retriever, "get_vectorstore", return_value=vs):
        result = search_knowledge_base.invoke({"query": "content"})

    assert "Chunk one content." in result
    assert "Chunk two content." in result


# ---------------------------------------------------------------------------
# search_knowledge_base — is a valid LangChain tool
# ---------------------------------------------------------------------------

def test_search_kb_is_langchain_tool():
    from tools.rag_search import search_knowledge_base

    assert search_knowledge_base.name == "search_knowledge_base"
    assert "knowledge base" in search_knowledge_base.description.lower()


# ---------------------------------------------------------------------------
# search_knowledge_base registered in ALL_TOOLS
# ---------------------------------------------------------------------------

def test_search_kb_in_all_tools():
    from tools import ALL_TOOLS
    from tools.rag_search import search_knowledge_base

    assert search_knowledge_base in ALL_TOOLS
