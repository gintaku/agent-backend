"""Tests for backend/rag/watcher.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler():
    """Return a fresh _RAGEventHandler with ingest_file / delete_source mocked."""
    from rag.watcher import _RAGEventHandler

    handler = _RAGEventHandler()
    return handler


# ---------------------------------------------------------------------------
# _should_handle
# ---------------------------------------------------------------------------

def test_should_handle_txt():
    from rag.watcher import _RAGEventHandler
    h = _RAGEventHandler()
    assert h._should_handle("docs/file.txt") is True


def test_should_handle_md():
    from rag.watcher import _RAGEventHandler
    h = _RAGEventHandler()
    assert h._should_handle("README.md") is True


def test_should_handle_pdf():
    from rag.watcher import _RAGEventHandler
    h = _RAGEventHandler()
    assert h._should_handle("report.pdf") is True


def test_should_handle_ignores_py():
    from rag.watcher import _RAGEventHandler
    h = _RAGEventHandler()
    assert h._should_handle("script.py") is False


def test_should_handle_ignores_no_extension():
    from rag.watcher import _RAGEventHandler
    h = _RAGEventHandler()
    assert h._should_handle("Makefile") is False


def test_should_handle_case_insensitive():
    from rag.watcher import _RAGEventHandler
    h = _RAGEventHandler()
    assert h._should_handle("FILE.TXT") is True
    assert h._should_handle("REPORT.PDF") is True


# ---------------------------------------------------------------------------
# on_created
# ---------------------------------------------------------------------------

def test_on_created_ingests_supported_file(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("hello", encoding="utf-8")

    with patch("rag.watcher.ingest_file", return_value=3) as mock_ingest:
        handler = _make_handler()
        handler.on_created(FileCreatedEvent(str(f)))

    mock_ingest.assert_called_once_with(f)


def test_on_created_skips_unsupported_extension(tmp_path: Path):
    f = tmp_path / "script.py"
    f.write_text("print()", encoding="utf-8")

    with patch("rag.watcher.ingest_file") as mock_ingest:
        handler = _make_handler()
        handler.on_created(FileCreatedEvent(str(f)))

    mock_ingest.assert_not_called()


def test_on_created_skips_directories():
    with patch("rag.watcher.ingest_file") as mock_ingest:
        handler = _make_handler()
        event = FileCreatedEvent("/some/dir/")
        event.is_directory = True
        handler.on_created(event)

    mock_ingest.assert_not_called()


def test_on_created_skips_nonexistent_file(tmp_path: Path):
    """File listed in event but not on disk (e.g., temp file race) — silently skip."""
    with patch("rag.watcher.ingest_file") as mock_ingest:
        handler = _make_handler()
        handler.on_created(FileCreatedEvent(str(tmp_path / "ghost.txt")))

    mock_ingest.assert_not_called()


def test_on_created_handles_ingest_exception(tmp_path: Path, caplog):
    f = tmp_path / "bad.txt"
    f.write_text("data", encoding="utf-8")

    with patch("rag.watcher.ingest_file", side_effect=RuntimeError("oops")):
        handler = _make_handler()
        # Should NOT raise
        handler.on_created(FileCreatedEvent(str(f)))


# ---------------------------------------------------------------------------
# on_modified
# ---------------------------------------------------------------------------

def test_on_modified_ingests_supported_file(tmp_path: Path):
    f = tmp_path / "notes.md"
    f.write_text("updated", encoding="utf-8")

    with patch("rag.watcher.ingest_file", return_value=2) as mock_ingest:
        handler = _make_handler()
        handler.on_modified(FileModifiedEvent(str(f)))

    mock_ingest.assert_called_once_with(f)


def test_on_modified_skips_directories():
    with patch("rag.watcher.ingest_file") as mock_ingest:
        handler = _make_handler()
        event = FileModifiedEvent("/some/dir/")
        event.is_directory = True
        handler.on_modified(event)

    mock_ingest.assert_not_called()


# ---------------------------------------------------------------------------
# on_deleted
# ---------------------------------------------------------------------------

def test_on_deleted_removes_chunks():
    src = "/rag_docs/old.txt"

    with patch("rag.watcher.delete_source") as mock_delete:
        handler = _make_handler()
        handler.on_deleted(FileDeletedEvent(src))

    expected_source = str(Path(src).resolve())
    mock_delete.assert_called_once_with(expected_source)


def test_on_deleted_skips_directories():
    with patch("rag.watcher.delete_source") as mock_delete:
        handler = _make_handler()
        event = FileDeletedEvent("/some/dir/")
        event.is_directory = True
        handler.on_deleted(event)

    mock_delete.assert_not_called()


def test_on_deleted_skips_unsupported_extension():
    with patch("rag.watcher.delete_source") as mock_delete:
        handler = _make_handler()
        handler.on_deleted(FileDeletedEvent("/rag_docs/script.py"))

    mock_delete.assert_not_called()


def test_on_deleted_handles_exception(caplog):
    with patch("rag.watcher.delete_source", side_effect=RuntimeError("db error")):
        handler = _make_handler()
        # Should NOT raise
        handler.on_deleted(FileDeletedEvent("/rag_docs/doc.txt"))


# ---------------------------------------------------------------------------
# on_moved
# ---------------------------------------------------------------------------

def test_on_moved_deletes_old_and_ingests_new(tmp_path: Path):
    src = "/rag_docs/old.txt"
    dest = tmp_path / "new.md"
    dest.write_text("moved", encoding="utf-8")

    with (
        patch("rag.watcher.delete_source") as mock_delete,
        patch("rag.watcher.ingest_file", return_value=1) as mock_ingest,
    ):
        handler = _make_handler()
        handler.on_moved(FileMovedEvent(src, str(dest)))

    mock_delete.assert_called_once_with(str(Path(src).resolve()))
    mock_ingest.assert_called_once_with(dest)


def test_on_moved_only_deletes_when_dest_unsupported(tmp_path: Path):
    src = "/rag_docs/old.txt"
    dest = "/somewhere/file.py"

    with (
        patch("rag.watcher.delete_source") as mock_delete,
        patch("rag.watcher.ingest_file") as mock_ingest,
    ):
        handler = _make_handler()
        handler.on_moved(FileMovedEvent(src, dest))

    mock_delete.assert_called_once()
    mock_ingest.assert_not_called()


def test_on_moved_skips_when_both_unsupported():
    with (
        patch("rag.watcher.delete_source") as mock_delete,
        patch("rag.watcher.ingest_file") as mock_ingest,
    ):
        handler = _make_handler()
        handler.on_moved(FileMovedEvent("/docs/a.py", "/docs/b.py"))

    mock_delete.assert_not_called()
    mock_ingest.assert_not_called()


# ---------------------------------------------------------------------------
# start_watcher
# ---------------------------------------------------------------------------

def test_start_watcher_creates_dir_and_starts_observer(tmp_path: Path, monkeypatch):
    watch_dir = tmp_path / "rag_docs_watch"

    import config as cfg_module
    fake_settings = MagicMock()
    fake_settings.rag_docs_dir = str(watch_dir)
    monkeypatch.setattr(cfg_module, "_settings", fake_settings)

    mock_observer = MagicMock()

    with patch("rag.watcher.Observer", return_value=mock_observer):
        from rag.watcher import start_watcher
        obs = start_watcher()

    assert watch_dir.exists()
    mock_observer.schedule.assert_called_once()
    mock_observer.start.assert_called_once()
    assert obs is mock_observer


def test_start_watcher_watches_correct_directory(tmp_path: Path, monkeypatch):
    watch_dir = tmp_path / "my_rag"

    import config as cfg_module
    fake_settings = MagicMock()
    fake_settings.rag_docs_dir = str(watch_dir)
    monkeypatch.setattr(cfg_module, "_settings", fake_settings)

    mock_observer = MagicMock()

    with patch("rag.watcher.Observer", return_value=mock_observer):
        from rag.watcher import start_watcher
        start_watcher()

    schedule_call = mock_observer.schedule.call_args
    assert schedule_call.args[1] == str(watch_dir)
    assert schedule_call.kwargs.get("recursive") is True
