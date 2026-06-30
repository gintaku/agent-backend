"""Tests for tools/file_ops.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_workspace(tmp_path, monkeypatch):
    """Point WORKSPACE_DIR at a fresh tmp directory for every test."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import config
    config.reset_settings()
    yield
    config.reset_settings()


def _invoke(tool_name: str, **kwargs) -> str:
    import importlib
    mod = importlib.import_module("tools.file_ops")
    return getattr(mod, tool_name).invoke(kwargs)


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

def test_write_file_creates_file(tmp_path):
    result = _invoke("write_file", path="hello.txt", content="hello world")
    assert "Successfully wrote" in result
    assert (tmp_path / "hello.txt").read_text() == "hello world"


def test_write_file_creates_parent_dirs(tmp_path):
    result = _invoke("write_file", path="subdir/nested/file.txt", content="data")
    assert "Successfully wrote" in result
    assert (tmp_path / "subdir" / "nested" / "file.txt").exists()


def test_write_file_overwrites_existing(tmp_path):
    (tmp_path / "f.txt").write_text("old")
    _invoke("write_file", path="f.txt", content="new")
    assert (tmp_path / "f.txt").read_text() == "new"


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def test_read_file_returns_content(tmp_path):
    (tmp_path / "sample.txt").write_text("sample content")
    result = _invoke("read_file", path="sample.txt")
    assert result == "sample content"


def test_read_file_not_found(tmp_path):
    result = _invoke("read_file", path="nonexistent.txt")
    assert result.startswith("Error")
    assert "not found" in result


def test_read_file_on_directory(tmp_path):
    (tmp_path / "mydir").mkdir()
    result = _invoke("read_file", path="mydir")
    assert result.startswith("Error")
    assert "directory" in result.lower()


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

def test_list_directory_default(tmp_path):
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "b.txt").write_text("")
    result = _invoke("list_directory", path=".")
    assert "a.txt" in result
    assert "b.txt" in result


def test_list_directory_shows_subdirs(tmp_path):
    (tmp_path / "subdir").mkdir()
    result = _invoke("list_directory", path=".")
    assert "subdir/" in result


def test_list_directory_empty(tmp_path):
    (tmp_path / "empty").mkdir()
    result = _invoke("list_directory", path="empty")
    assert "empty" in result.lower()


def test_list_directory_not_found(tmp_path):
    result = _invoke("list_directory", path="nosuchdir")
    assert result.startswith("Error")


# ---------------------------------------------------------------------------
# delete_file
# ---------------------------------------------------------------------------

def test_delete_file_removes_it(tmp_path):
    (tmp_path / "todelete.txt").write_text("bye")
    result = _invoke("delete_file", path="todelete.txt")
    assert "Successfully deleted" in result
    assert not (tmp_path / "todelete.txt").exists()


def test_delete_file_not_found(tmp_path):
    result = _invoke("delete_file", path="ghost.txt")
    assert result.startswith("Error")


def test_delete_file_rejects_directory(tmp_path):
    (tmp_path / "adir").mkdir()
    result = _invoke("delete_file", path="adir")
    assert result.startswith("Error")
    assert "directory" in result.lower()


# ---------------------------------------------------------------------------
# Path traversal protection
# ---------------------------------------------------------------------------

def test_read_file_traversal_blocked(tmp_path):
    result = _invoke("read_file", path="../../etc/passwd")
    assert result.startswith("Error")
    assert "Access denied" in result or "outside" in result


def test_write_file_traversal_blocked(tmp_path):
    result = _invoke("write_file", path="../escape.txt", content="bad")
    assert result.startswith("Error")
    assert "Access denied" in result or "outside" in result


def test_delete_file_traversal_blocked(tmp_path):
    result = _invoke("delete_file", path="../../tmp/something")
    assert result.startswith("Error")


def test_list_directory_traversal_blocked(tmp_path):
    result = _invoke("list_directory", path="../../")
    assert result.startswith("Error")
