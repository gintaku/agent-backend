"""Singleton Chroma vectorstore for the RAG module."""
from __future__ import annotations

from langchain_chroma import Chroma

from config import get_settings
from rag.embeddings import get_embeddings

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    """Return the lazily-initialised persistent Chroma vectorstore singleton."""
    global _vectorstore
    if _vectorstore is None:
        s = get_settings()
        _vectorstore = Chroma(
            collection_name=s.rag_collection_name,
            embedding_function=get_embeddings(),
            persist_directory=s.chroma_persist_dir,
        )
    return _vectorstore


def reset_vectorstore() -> None:
    """Clear the cached singleton so the next call creates a fresh instance.

    Useful in tests and after config changes.
    """
    global _vectorstore
    _vectorstore = None
