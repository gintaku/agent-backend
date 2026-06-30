"""Knowledge base retrieval for the RAG module.

Embeds the incoming query and returns the top-K most relevant chunks from
ChromaDB, formatted as a single string ready to be injected into a prompt.
"""
from __future__ import annotations

from config import get_settings
from rag.chroma_client import get_vectorstore


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
    vs = get_vectorstore()
    s = get_settings()
    try:
        results = vs.similarity_search_with_score(query, k=s.rag_top_k)
    except Exception:
        return ""

    if not results:
        return ""

    # Filter by score threshold — Chroma returns L2 distance (lower = more similar).
    # Only inject chunks whose distance is at or below the configured threshold.
    threshold = s.rag_score_threshold
    results = [(doc, score) for doc, score in results if score <= threshold]

    if not results:
        return ""

    chunks = []
    for doc, _score in results:
        chunks.append(f"[Score: {_score}]\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)
