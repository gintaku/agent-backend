"""Shared pytest fixtures for the backend test suite."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure `backend/` is importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Clear the Settings singleton before and after every test for isolation."""
    import config  # noqa: PLC0415
    config.reset_settings()
    yield
    config.reset_settings()


@pytest.fixture()
def client() -> TestClient:
    """Return a synchronous FastAPI TestClient."""
    from main import app  # noqa: PLC0415
    return TestClient(app)


class FakeVectorStore:
    """Minimal in-memory stand-in for PGVector's public interface.

    PGVector always talks to a real Postgres+pgvector instance — unlike
    Chroma, there's no embedded/in-memory mode to instantiate directly in a
    unit test. So instead of standing up a real database for every test,
    ``rag.retriever`` / ``rag.ingestor`` are unit-tested against this fake,
    which implements only the methods they actually call:
    ``add_documents``, ``similarity_search``, ``similarity_search_with_score``,
    and ``delete``.

    Real Postgres-specific behavior (schema creation, upsert semantics,
    distance-metric scoring) is exercised separately in
    ``test_rag_pg_integration.py``, which is skipped unless a real database
    is available.
    """

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}

    def add_documents(self, docs: list[Document], ids: list[str] | None = None) -> list[str]:
        if ids is None:
            ids = [str(i) for i in range(len(self._docs), len(self._docs) + len(docs))]
        for doc_id, doc in zip(ids, docs):
            self._docs[doc_id] = doc
        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> list[Document]:
        return list(self._docs.values())[:k]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs
    ) -> list[tuple[Document, float]]:
        # Every test uses fixed-value fake embeddings, so every stored doc is
        # equidistant from the query — a constant 0.0 score reproduces the
        # "everything matches" behavior the old fake-embeddings-in-real-Chroma
        # tests relied on, without needing real vector math.
        return [(doc, 0.0) for doc in list(self._docs.values())[:k]]

    def delete(self, ids: list[str] | None = None, **kwargs) -> None:
        if ids is None:
            self._docs.clear()
            return
        for doc_id in ids:
            self._docs.pop(doc_id, None)


@pytest.fixture()
def vs() -> FakeVectorStore:
    """Isolated fake vectorstore per test — see :class:`FakeVectorStore`."""
    return FakeVectorStore()
