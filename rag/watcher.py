"""Filesystem watcher for the RAG knowledge base.

Uses :mod:`watchdog` to monitor ``RAG_DOCS_DIR`` for changes:

* **Created / modified** — the file is (re-)ingested via :func:`~rag.ingestor.ingest_file`.
* **Deleted** — all chunks previously stored for that source are removed via
  :func:`~rag.ingestor.delete_source`.

Usage::

    from rag.watcher import start_watcher
    observer = start_watcher()          # returns a running watchdog Observer
    ...
    observer.stop(); observer.join()    # clean shutdown
"""
from __future__ import annotations

import logging
from pathlib import Path

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from config import get_settings
from rag.ingestor import delete_source, ingest_file

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


class _RAGEventHandler(FileSystemEventHandler):
    """Watchdog event handler that syncs the knowledge base with filesystem changes."""

    def _should_handle(self, path: str) -> bool:
        return Path(path).suffix.lower() in _SUPPORTED_SUFFIXES

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ingest(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            return
        try:
            n = ingest_file(p)
            logger.info("RAG: ingested %d chunk(s) from %s", n, path)
        except Exception as exc:
            logger.error("RAG: failed to ingest %s — %s", path, exc)

    def _delete(self, path: str) -> None:
        source = str(Path(path).resolve())
        try:
            delete_source(source)
            logger.info("RAG: removed chunks for deleted file %s", path)
        except Exception as exc:
            logger.error("RAG: failed to remove chunks for %s — %s", path, exc)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if not event.is_directory and self._should_handle(event.src_path):
            self._ingest(event.src_path)

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        if not event.is_directory and self._should_handle(event.src_path):
            self._ingest(event.src_path)

    def on_deleted(self, event: FileDeletedEvent) -> None:  # type: ignore[override]
        if not event.is_directory and self._should_handle(event.src_path):
            self._delete(event.src_path)

    def on_moved(self, event: FileMovedEvent) -> None:  # type: ignore[override]
        # Treat a move as delete-old + create-new
        if not event.is_directory:
            if self._should_handle(event.src_path):
                self._delete(event.src_path)
            if self._should_handle(event.dest_path):
                self._ingest(event.dest_path)


def start_watcher() -> Observer:
    """Create, start and return a :class:`watchdog.observers.Observer`.

    The observer watches ``RAG_DOCS_DIR`` recursively and fires
    :class:`_RAGEventHandler` events.  The caller is responsible for calling
    ``observer.stop()`` and ``observer.join()`` on shutdown.
    """
    settings = get_settings()
    watch_dir = Path(settings.rag_docs_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)

    handler = _RAGEventHandler()
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=True)
    observer.start()
    logger.info("RAG watcher started on %s", watch_dir)
    return observer
