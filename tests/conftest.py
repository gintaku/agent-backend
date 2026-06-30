"""Shared pytest fixtures for the backend test suite."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure `backend/` is importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient


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
