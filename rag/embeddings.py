"""Embedding model factory for the RAG module."""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from config import get_settings


def get_embeddings() -> OpenAIEmbeddings:
    """Return an OpenAI-compatible embeddings model.

    Uses a direct OpenAI key when ``OPENAI_API_KEY`` is set in config,
    otherwise falls back to the OpenRouter-compatible embeddings endpoint
    (which accepts the same ``OPENROUTER_API_KEY`` used by the chat LLM).
    """
    s = get_settings()
    if s.openai_api_key:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=s.openai_api_key,
        )
    # OpenRouter exposes an OpenAI-compatible embeddings endpoint.
    # The model name must be prefixed with the provider namespace.
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=s.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )
