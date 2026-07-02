"""Knowledge base retrieval for the RAG module.

Embeds the incoming query and returns the top-K most relevant chunks from
Postgres/pgvector (via the read-replica connection), formatted as a single
string ready to be injected into a prompt.
"""
from __future__ import annotations

import logging

from config import get_settings
from rag.pg_client import get_read_vectorstore

logger = logging.getLogger("rag.retriever")


def retrieve(query: str) -> str:
    """Return top-K relevant knowledge-base chunks for *query*.

    Results are formatted as::

        [Source: /path/to/doc.txt]
        <chunk text>

        ---

        [Source: https://example.com]
        <chunk text>

    Returns an empty string when the knowledge base is empty or an error
    occurs so callers can safely skip prompt injection.
    """
    s = get_settings()
    try:
        vs = get_read_vectorstore()
        results = vs.similarity_search_with_score(query, k=s.rag_top_k)
    except Exception:
        logger.exception("RAG retrieval failed — continuing without context")
        return ""

    if not results:
        return ""

    # Filter by score threshold — pgvector distance (lower = more similar,
    # exact metric depends on the configured distance strategy).
    threshold = s.rag_score_threshold
    results = [(doc, score) for doc, score in results if score <= threshold]

    if not results:
        return ""

    chunks = []
    for doc, _score in results:
        source = doc.metadata.get("source", "unknown")
        chunks.append(f"[Source: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)