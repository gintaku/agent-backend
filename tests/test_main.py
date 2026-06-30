"""Tests for backend/main.py — REST endpoints and WebSocket."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# GET /api/config
# ---------------------------------------------------------------------------

def test_get_config_returns_model_and_cmd_mode(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "gpt-4o")
    monkeypatch.setenv("CMD_MODE", "bypass")

    import config
    config.reset_settings()

    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gpt-4o"
    assert data["cmd_mode"] == "bypass"


def test_get_config_does_not_expose_api_key(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-super-secret")
    monkeypatch.setenv("MODEL", "test-model")
    monkeypatch.setenv("CMD_MODE", "permission")

    import config
    config.reset_settings()

    response = client.get("/api/config")
    assert response.status_code == 200
    payload = response.json()
    assert "openrouter_api_key" not in payload
    assert "sk-super-secret" not in str(payload)


def test_get_config_response_shape(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    import config
    config.reset_settings()

    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert {"model", "cmd_mode", "enabled_tools"}.issubset(data.keys())


# ---------------------------------------------------------------------------
# GET /api/tools
# ---------------------------------------------------------------------------

def test_get_tools_returns_list(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import config
    config.reset_settings()

    response = client.get("/api/tools")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_get_tools_items_have_name_and_enabled(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import config
    config.reset_settings()

    response = client.get("/api/tools")
    assert response.status_code == 200
    for item in response.json():
        assert "name" in item
        assert "enabled" in item
        assert isinstance(item["enabled"], bool)


def test_get_tools_all_enabled_by_default(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("ENABLED_TOOLS", raising=False)
    import config
    config.reset_settings()

    response = client.get("/api/tools")
    assert all(item["enabled"] for item in response.json())


# ---------------------------------------------------------------------------
# PUT /api/config
# ---------------------------------------------------------------------------

def test_put_config_updates_model(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "new-model")
    monkeypatch.setenv("CMD_MODE", "permission")

    import config
    config.reset_settings()

    with patch("main.update_env_file"):
        response = client.put("/api/config", json={"model": "new-model"})

    assert response.status_code == 200
    assert response.json()["model"] == "new-model"


def test_put_config_updates_cmd_mode(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "some-model")
    monkeypatch.setenv("CMD_MODE", "bypass")

    import config
    config.reset_settings()

    with patch("main.update_env_file"):
        response = client.put("/api/config", json={"cmd_mode": "bypass"})

    assert response.status_code == 200
    assert response.json()["cmd_mode"] == "bypass"


def test_put_config_calls_update_env_file_for_model(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "some-model")
    monkeypatch.setenv("CMD_MODE", "permission")

    import config
    config.reset_settings()

    with patch("main.update_env_file") as mock_update:
        client.put("/api/config", json={"model": "some-model"})
        mock_update.assert_called_once_with("MODEL", "some-model")


def test_put_config_calls_update_env_file_for_cmd_mode(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "some-model")
    monkeypatch.setenv("CMD_MODE", "bypass")

    import config
    config.reset_settings()

    with patch("main.update_env_file") as mock_update:
        client.put("/api/config", json={"cmd_mode": "bypass"})
        mock_update.assert_called_once_with("CMD_MODE", "bypass")


def test_put_config_empty_body_calls_no_update(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "m")
    monkeypatch.setenv("CMD_MODE", "permission")

    import config
    config.reset_settings()

    with patch("main.update_env_file") as mock_update:
        response = client.put("/api/config", json={})

    assert response.status_code == 200
    mock_update.assert_not_called()


def test_put_config_invalid_cmd_mode_returns_422(client):
    response = client.put("/api/config", json={"cmd_mode": "not_valid"})
    assert response.status_code == 422


def test_put_config_updates_both_fields(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "target-model")
    monkeypatch.setenv("CMD_MODE", "bypass")

    import config
    config.reset_settings()

    with patch("main.update_env_file") as mock_update:
        response = client.put(
            "/api/config", json={"model": "target-model", "cmd_mode": "bypass"}
        )

    assert response.status_code == 200
    calls = {call.args for call in mock_update.call_args_list}
    assert ("MODEL", "target-model") in calls
    assert ("CMD_MODE", "bypass") in calls


# ---------------------------------------------------------------------------
# WebSocket /ws/{session_id}
# ---------------------------------------------------------------------------

def test_websocket_receives_connected_message(client):
    with client.websocket_connect("/ws/abc-123") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "connected"
    assert msg["session_id"] == "abc-123"


def test_websocket_session_id_is_reflected(client):
    with client.websocket_connect("/ws/my-session") as ws:
        msg = ws.receive_json()
    assert msg["session_id"] == "my-session"


def test_websocket_message_type_dispatches_to_agent(client, monkeypatch):
    """A 'message' payload should invoke run_agent (mocked to avoid network)."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import config
    config.reset_settings()

    async def _fake_run_agent(session_id, content, ws_send):
        await ws_send({"type": "done"})

    with patch("main._run_agent_safe", side_effect=_fake_run_agent):
        with client.websocket_connect("/ws/test") as ws:
            ws.receive_json()  # connected
            ws.send_text('{"type": "message", "content": "hello"}')
            reply = ws.receive_json()
    assert reply["type"] == "done"


def test_websocket_unknown_message_type_is_ignored(client, monkeypatch):
    """Unknown message types must not raise; the connection stays open."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import config
    config.reset_settings()

    with client.websocket_connect("/ws/test") as ws:
        ws.receive_json()  # connected
        ws.send_text('{"type": "unknown_type", "content": "ping"}')
        # No reply expected; just verify the connection doesn't crash.
        ws.close()


def test_websocket_permission_response_calls_resolve(client, monkeypatch):
    """A 'permission_response' payload must call resolve_permission."""
    from unittest.mock import patch

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import config
    config.reset_settings()

    with patch("tools.shell.resolve_permission") as mock_resolve:
        with client.websocket_connect("/ws/test-perm") as ws:
            ws.receive_json()  # connected
            ws.send_text('{"type": "permission_response", "approved": true}')
            ws.close()

    mock_resolve.assert_called_once_with("test-perm", True)
