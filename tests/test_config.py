"""Tests for backend/config.py — settings loading and .env file helpers."""
from __future__ import annotations

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# get_settings / reset_settings
# ---------------------------------------------------------------------------

def test_default_model_value(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("CMD_MODE", raising=False)

    import config
    s = config.get_settings()
    assert s.model == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"


def test_default_cmd_mode_is_permission(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("CMD_MODE", raising=False)

    import config
    s = config.get_settings()
    assert s.cmd_mode == "permission"


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "gpt-4o")
    monkeypatch.setenv("CMD_MODE", "bypass")

    import config
    s = config.get_settings()
    assert s.model == "gpt-4o"
    assert s.cmd_mode == "bypass"


def test_api_key_is_read(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-custom-key")

    import config
    s = config.get_settings()
    assert s.openrouter_api_key == "sk-custom-key"


def test_settings_singleton_is_cached(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    import config
    s1 = config.get_settings()
    s2 = config.get_settings()
    assert s1 is s2


def test_reset_settings_clears_cache(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    import config
    s1 = config.get_settings()
    config.reset_settings()
    s2 = config.get_settings()
    assert s1 is not s2


def test_invalid_cmd_mode_raises_validation_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("CMD_MODE", "not_a_valid_mode")

    import config
    with pytest.raises(ValidationError):
        config.get_settings()


# ---------------------------------------------------------------------------
# update_env_file
# ---------------------------------------------------------------------------

def test_update_env_file_creates_file_if_missing(tmp_path, monkeypatch):
    import config
    fake_env = tmp_path / ".env"
    monkeypatch.setattr(config, "ENV_FILE", fake_env)

    config.update_env_file("MODEL", "gpt-4o-mini")

    assert fake_env.exists()
    assert "MODEL=gpt-4o-mini" in fake_env.read_text()


def test_update_env_file_updates_existing_key(tmp_path, monkeypatch):
    import config
    fake_env = tmp_path / ".env"
    fake_env.write_text("MODEL=old-model\nCMD_MODE=bypass\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", fake_env)

    config.update_env_file("MODEL", "new-model")

    content = fake_env.read_text()
    assert "MODEL=new-model" in content
    assert "old-model" not in content
    # Other lines must be preserved
    assert "CMD_MODE=bypass" in content


def test_update_env_file_appends_missing_key(tmp_path, monkeypatch):
    import config
    fake_env = tmp_path / ".env"
    fake_env.write_text("CMD_MODE=permission\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", fake_env)

    config.update_env_file("MODEL", "gpt-4o")

    content = fake_env.read_text()
    assert "MODEL=gpt-4o" in content
    assert "CMD_MODE=permission" in content


def test_update_env_file_preserves_all_other_lines(tmp_path, monkeypatch):
    import config
    initial = "OPENROUTER_API_KEY=sk-abc\nMODEL=old\nCMD_MODE=bypass\n"
    fake_env = tmp_path / ".env"
    fake_env.write_text(initial, encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", fake_env)

    config.update_env_file("MODEL", "updated")

    content = fake_env.read_text()
    assert "OPENROUTER_API_KEY=sk-abc" in content
    assert "MODEL=updated" in content
    assert "CMD_MODE=bypass" in content


def test_update_env_file_only_updates_exact_key(tmp_path, monkeypatch):
    """MODEL_EXTRA should not be touched when updating MODEL."""
    import config
    fake_env = tmp_path / ".env"
    fake_env.write_text("MODEL=old\nMODEL_EXTRA=keep\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", fake_env)

    config.update_env_file("MODEL", "new")

    content = fake_env.read_text()
    assert "MODEL=new" in content
    assert "MODEL_EXTRA=keep" in content
