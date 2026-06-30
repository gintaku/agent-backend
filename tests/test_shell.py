"""Tests for tools/shell.py."""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import config
    config.reset_settings()
    yield
    config.reset_settings()


async def _run(command: str) -> str:
    from tools.shell import run_command
    return await run_command.ainvoke({"command": command})


# ---------------------------------------------------------------------------
# _is_read_only helper
# ---------------------------------------------------------------------------

def test_is_read_only_dir():
    from tools.shell import _is_read_only
    assert _is_read_only("dir") is True
    assert _is_read_only("dir /b") is True


def test_is_read_only_ls():
    from tools.shell import _is_read_only
    assert _is_read_only("ls") is True
    assert _is_read_only("ls -la") is True


def test_is_read_only_git_status():
    from tools.shell import _is_read_only
    assert _is_read_only("git status") is True
    assert _is_read_only("git log --oneline") is True


def test_is_read_only_git_diff():
    from tools.shell import _is_read_only
    assert _is_read_only("git diff HEAD~1") is True


def test_is_not_read_only_rm():
    from tools.shell import _is_read_only
    assert _is_read_only("rm -rf /") is False


def test_is_not_read_only_pip_install():
    from tools.shell import _is_read_only
    assert _is_read_only("pip install requests") is False


def test_is_not_read_only_python_script():
    from tools.shell import _is_read_only
    assert _is_read_only("python script.py") is False


def test_is_not_read_only_git_push():
    from tools.shell import _is_read_only
    assert _is_read_only("git push origin main") is False


# ---------------------------------------------------------------------------
# bypass mode — all commands run without permission prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bypass_mode_runs_command(monkeypatch):
    monkeypatch.setenv("CMD_MODE", "bypass")
    import config
    config.reset_settings()

    result = await _run("echo hello_bypass")
    assert "hello_bypass" in result


@pytest.mark.asyncio
async def test_bypass_mode_non_readonly_runs_directly(monkeypatch):
    monkeypatch.setenv("CMD_MODE", "bypass")
    import config
    config.reset_settings()

    # "pip --version" may or may not be read-only, but bypass should always run
    result = await _run("echo non_readonly_test")
    assert "non_readonly_test" in result


# ---------------------------------------------------------------------------
# permission mode — read-only commands run without a gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_permission_mode_readonly_runs_without_gate(monkeypatch):
    monkeypatch.setenv("CMD_MODE", "permission")
    import config
    config.reset_settings()

    result = await _run("echo read_only_echo")
    # 'echo' is in _READ_ONLY_PREFIXES, so it should run immediately
    assert "read_only_echo" in result


# ---------------------------------------------------------------------------
# permission mode — non-read-only without WebSocket → error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_permission_mode_no_ws_returns_error(monkeypatch):
    monkeypatch.setenv("CMD_MODE", "permission")
    import config
    config.reset_settings()

    # session_id_var and ws_send_var are None by default
    result = await _run("pip install something")
    assert result.startswith("Error")
    assert "permission" in result.lower() or "WebSocket" in result


# ---------------------------------------------------------------------------
# permission mode — approved via resolve_permission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_permission_approved_executes_command(monkeypatch):
    monkeypatch.setenv("CMD_MODE", "permission")
    import config
    config.reset_settings()

    from tools import shell

    sent_messages: list[dict] = []

    async def fake_ws_send(msg: dict) -> None:
        sent_messages.append(msg)
        # Immediately approve the request so the tool doesn't block the test
        shell.resolve_permission("test-session", approved=True)

    token = shell.session_id_var.set("test-session")
    token2 = shell.ws_send_var.set(fake_ws_send)
    try:
        # 'python -c ...' is NOT in the read-only list, so permission gate fires
        result = await _run('python -c "print(\'approved_marker\')"')
    finally:
        shell.session_id_var.reset(token)
        shell.ws_send_var.reset(token2)

    assert "approved_marker" in result
    assert any(m.get("type") == "permission_request" for m in sent_messages)


# ---------------------------------------------------------------------------
# permission mode — denied via resolve_permission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_permission_denied_does_not_execute(monkeypatch):
    monkeypatch.setenv("CMD_MODE", "permission")
    import config
    config.reset_settings()

    from tools import shell

    async def fake_ws_send(msg: dict) -> None:
        shell.resolve_permission("deny-session", approved=False)

    token = shell.session_id_var.set("deny-session")
    token2 = shell.ws_send_var.set(fake_ws_send)
    try:
        # 'python -c ...' is NOT in the read-only list, so permission gate fires
        result = await _run('python -c "print(\'denied_marker\')"')
    finally:
        shell.session_id_var.reset(token)
        shell.ws_send_var.reset(token2)

    assert "denied" in result.lower()
    assert "denied_marker" not in result


# ---------------------------------------------------------------------------
# resolve_permission
# ---------------------------------------------------------------------------

def test_resolve_permission_returns_false_for_unknown_session():
    from tools.shell import resolve_permission
    assert resolve_permission("nonexistent-session", approved=True) is False


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_permission_timeout_returns_error(monkeypatch):
    monkeypatch.setenv("CMD_MODE", "permission")
    import config
    config.reset_settings()

    from tools import shell

    async def fake_ws_send(msg: dict) -> None:
        # Never resolve — we patch asyncio.wait_for instead
        pass

    with patch("tools.shell.asyncio.wait_for", side_effect=asyncio.TimeoutError):
        token = shell.session_id_var.set("timeout-session")
        token2 = shell.ws_send_var.set(fake_ws_send)
        try:
            result = await _run("pip install something")
        finally:
            shell.session_id_var.reset(token)
            shell.ws_send_var.reset(token2)

    assert result.startswith("Error")
    assert "timed out" in result.lower()
