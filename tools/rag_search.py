"""RAG knowledge-base search tool for the LangChain agent."""
from __future__ import annotations

from langchain_core.tools import tool

from rag.retriever import retrieve


@tool
def search_knowledge_base(query: str) -> str:
    """Search the local knowledge base for information relevant to the query.
    This is the Github Copilot knowledge base.

    Uses vector similarity to find the most relevant document chunks
    previously ingested from the rag_docs/ folder. Returns formatted
    context passages with source labels, or a message when no results
    are found.
    """
    result = retrieve(query)
    if not result:
        return "No relevant information found in the knowledge base."
    return result
