"""Integration tests for the Postgres/pgvector RAG backend.

The unit tests in test_rag_ingestor.py / test_rag_retriever.py / test_rag_search.py
run against ``conftest.FakeVectorStore`` and never touch a real database —
that's deliberate, so the everyday suite stays fast and hermetic. But it
means none of them actually verify that ``langchain-postgres`` behaves the
way ``rag/pg_client.py`` assumes: that the collection/embedding tables get
created correctly, that ``add_documents(..., ids=...)`` truly upserts rather
than duplicates, that ``delete(ids=...)`` works, and that the read DSN can
see rows written through the write DSN.

These tests cover exactly that gap, against a **real** Postgres with the
``pgvector`` extension. They are skipped by default and only run when a
database is actually reachable, via one of:

* Setting ``TEST_DATABASE_URL`` to a Postgres DSN that already has
  ``pgvector`` available (e.g. a CNPG dev cluster, or a local
  ``pgvector/pgvector`` Docker image).
* Having the optional ``testcontainers`` package installed, in which case a
  disposable ``pgvector/pgvector:pg16`` container is spun up automatically
  for the test session.

Run explicitly with::

    TEST_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/testdb \\
        pytest tests/test_rag_pg_integration.py -v

or, with testcontainers installed and Docker available::

    pytest tests/test_rag_pg_integration.py -v
"""
from __future__ import annotations

import os
import uuid

import pytest
from langchain_core.documents import Document

pytestmark = pytest.mark.integration


def _external_database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def _pg_container():
    """Start a disposable pgvector-enabled Postgres container, if possible.

    Returns None (rather than failing collection) when neither
    TEST_DATABASE_URL nor testcontainers+Docker are available, so this whole
    file cleanly skips instead of erroring out in environments without
    Docker access (e.g. this sandboxed environment).
    """
    if _external_database_url():
        yield None  # external DB in use — no container needed
        return

    try:
        from testcontainers.postgres import PostgresContainer  # noqa: PLC0415
    except ImportError:
        yield None
        return

    try:
        container = PostgresContainer("pgvector/pgvector:pg16")
        container.start()
    except Exception:
        # Docker not available/reachable in this environment.
        yield None
        return

    try:
        yield container
    finally:
        container.stop()


@pytest.fixture()
def db_url(_pg_container) -> str:
    external = _external_database_url()
    if external:
        return external
    if _pg_container is None:
        pytest.skip(
            "No TEST_DATABASE_URL set and testcontainers/Docker unavailable — "
            "skipping Postgres/pgvector integration tests."
        )
    url = _pg_container.get_connection_url()  # postgresql+psycopg2://...
    return url.replace("postgresql+psycopg2://", "postgresql+psycopg://")


@pytest.fixture()
def _enable_pgvector(db_url: str) -> None:
    import psycopg

    with psycopg.connect(db_url.replace("postgresql+psycopg://", "postgresql://")) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()


class _FakeEmbeddings:
    """Deterministic, dependency-free embeddings for integration tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 128 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 128


@pytest.fixture()
def pg_store(db_url, _enable_pgvector):
    """A real PGVector instance against an isolated, throwaway collection."""
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=_FakeEmbeddings(),
        collection_name=f"test_{uuid.uuid4().hex}",
        connection=db_url,
        use_jsonb=True,
    )


# ---------------------------------------------------------------------------
# Schema / connection sanity
# ---------------------------------------------------------------------------

def test_pgvector_creates_schema_and_accepts_documents(pg_store):
    pg_store.add_documents(
        [Document(page_content="hello world", metadata={"source": "a.txt"})],
        ids=["id-1"],
    )
    results = pg_store.similarity_search("hello", k=1)
    assert len(results) == 1
    assert results[0].page_content == "hello world"


# ---------------------------------------------------------------------------
# Upsert semantics — what rag/ingestor.py's dedup logic relies on
# ---------------------------------------------------------------------------

def test_pgvector_add_documents_upserts_by_id(pg_store):
    pg_store.add_documents(
        [Document(page_content="version 1", metadata={"source": "a.txt"})],
        ids=["stable-id"],
    )
    pg_store.add_documents(
        [Document(page_content="version 2", metadata={"source": "a.txt"})],
        ids=["stable-id"],
    )
    results = pg_store.similarity_search("version", k=10)
    assert len(results) == 1
    assert results[0].page_content == "version 2"


# ---------------------------------------------------------------------------
# ID-based delete — what rag/ingestor.py's _delete_source relies on
# ---------------------------------------------------------------------------

def test_pgvector_delete_by_ids(pg_store):
    pg_store.add_documents(
        [
            Document(page_content="keep me", metadata={"source": "a.txt"}),
            Document(page_content="remove me", metadata={"source": "a.txt"}),
        ],
        ids=["keep-id", "remove-id"],
    )
    pg_store.delete(ids=["remove-id"])
    results = pg_store.similarity_search("me", k=10)
    contents = {doc.page_content for doc in results}
    assert contents == {"keep me"}


# ---------------------------------------------------------------------------
# Read DSN sees writes made through the write DSN
# ---------------------------------------------------------------------------

def test_write_then_read_via_separate_connections(db_url, _enable_pgvector):
    from langchain_postgres import PGVector

    collection = f"test_{uuid.uuid4().hex}"
    write_store = PGVector(
        embeddings=_FakeEmbeddings(),
        collection_name=collection,
        connection=db_url,
        use_jsonb=True,
    )
    read_store = PGVector(
        embeddings=_FakeEmbeddings(),
        collection_name=collection,
        connection=db_url,
        use_jsonb=True,
    )

    write_store.add_documents(
        [Document(page_content="written via primary", metadata={"source": "a.txt"})],
        ids=["id-1"],
    )
    results = read_store.similarity_search("written", k=1)
    assert len(results) == 1
    assert results[0].page_content == "written via primary"
