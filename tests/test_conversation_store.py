"""Tests for backend/conversation_store.py.

conversation_store.py talks to Postgres through SQLAlchemy sessions
(``db.engine.get_write_session`` / ``get_read_session``). Rather than
requiring a live Postgres for every test run, these tests swap the session
factory for one bound to an in-memory SQLite database — same "swap the
backing implementation" approach ``tests/conftest.py``'s ``FakeVectorStore``
uses for the RAG tests, just applied to a session factory instead of a
vectorstore.

conversation_store.py does ``from db.engine import get_write_session,
get_read_session`` — a "from X import Y" copies the name into
conversation_store's own module namespace at import time. Patching
``db.engine.get_write_session`` afterward would NOT affect that already-bound
reference, so every fixture/helper here patches ``conversation_store``'s own
attributes instead.

Real Postgres-specific behavior (JSONB storage, the "user" reserved-word
column, concurrent access) is out of scope here, the same way
``test_rag_pg_integration.py`` is kept separate from the everyday RAG unit
tests — if you want that coverage, it belongs in a dedicated
integration-test file gated the same way (TEST_DATABASE_URL / testcontainers).
"""
from __future__ import annotations

from typing import Any, Callable
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Base, Conversation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(monkeypatch):
    """Point conversation_store at a fresh in-memory SQLite database.

    StaticPool + a single shared connection is required because a plain
    ``sqlite:///:memory:`` engine hands out a *new*, empty database to every
    connection by default — without it, the session that creates the schema
    and the session a test later opens would each see a different (and for
    the second one, table-less) database.
    """
    import conversation_store

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    monkeypatch.setattr(conversation_store, "get_write_session", session_factory)
    monkeypatch.setattr(conversation_store, "get_read_session", session_factory)

    yield session_factory

    Base.metadata.drop_all(engine)
    engine.dispose()


class _FailingSession:
    """Context-manager session stub that raises SQLAlchemyError on demand.

    Used to exercise conversation_store's except-SQLAlchemyError branches
    (rollback + log, then either re-raise or return a failure value)
    without needing to actually break a database connection mid-test.
    """

    def __init__(self, fail_on: str, get_return: Any = None):
        self._fail_on = fail_on
        self._get_return = get_return
        self.rolled_back = False

    def __enter__(self) -> "_FailingSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def _maybe_fail(self, name: str) -> None:
        if self._fail_on == name:
            raise SQLAlchemyError(f"simulated failure in {name}")

    def get(self, *args, **kwargs):
        self._maybe_fail("get")
        return self._get_return

    def add(self, obj) -> None:
        self._maybe_fail("add")

    def execute(self, *args, **kwargs):
        self._maybe_fail("execute")
        return MagicMock(rowcount=0)

    def commit(self) -> None:
        self._maybe_fail("commit")

    def rollback(self) -> None:
        self.rolled_back = True


def _patch_write(monkeypatch, session_factory: Callable[[], Any]) -> None:
    import conversation_store
    monkeypatch.setattr(conversation_store, "get_write_session", session_factory)


def _patch_read(monkeypatch, session_factory: Callable[[], Any]) -> None:
    import conversation_store
    monkeypatch.setattr(conversation_store, "get_read_session", session_factory)


# ---------------------------------------------------------------------------
# _validate_session_id
# ---------------------------------------------------------------------------

def test_validate_session_id_accepts_alphanumeric():
    from conversation_store import _validate_session_id
    _validate_session_id("abc-123_XYZ")  # must not raise


def test_validate_session_id_rejects_spaces():
    from conversation_store import _validate_session_id
    with pytest.raises(ValueError):
        _validate_session_id("has spaces")


def test_validate_session_id_rejects_path_traversal():
    from conversation_store import _validate_session_id
    with pytest.raises(ValueError):
        _validate_session_id("../../etc/passwd")


def test_validate_session_id_rejects_too_long():
    from conversation_store import _validate_session_id
    with pytest.raises(ValueError):
        _validate_session_id("a" * 129)


def test_validate_session_id_accepts_max_length():
    from conversation_store import _validate_session_id
    _validate_session_id("a" * 128)  # must not raise


# ---------------------------------------------------------------------------
# serialize_lc_messages / deserialize_lc_messages
# ---------------------------------------------------------------------------

def test_serialize_lc_messages_basic_roundtrip():
    from conversation_store import deserialize_lc_messages, serialize_lc_messages

    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    serialized = serialize_lc_messages(messages)
    restored = deserialize_lc_messages(serialized)

    assert [m.type for m in restored] == ["human", "ai"]
    assert [m.content for m in restored] == ["hi", "hello"]


def test_serialize_lc_messages_includes_name_when_present():
    from conversation_store import serialize_lc_messages

    msg = HumanMessage(content="hi", name="alice")
    result = serialize_lc_messages([msg])
    assert result[0]["name"] == "alice"


def test_serialize_lc_messages_omits_name_when_absent():
    from conversation_store import serialize_lc_messages

    result = serialize_lc_messages([HumanMessage(content="hi")])
    assert "name" not in result[0]


def test_deserialize_lc_messages_skips_unknown_types():
    from conversation_store import deserialize_lc_messages

    result = deserialize_lc_messages([
        {"type": "human", "content": "hi"},
        {"type": "tool", "content": "unsupported"},
        {"type": "ai", "content": "hello"},
    ])
    assert [m.content for m in result] == ["hi", "hello"]


def test_deserialize_lc_messages_restores_system_message():
    from conversation_store import deserialize_lc_messages

    result = deserialize_lc_messages([{"type": "system", "content": "be nice"}])
    assert len(result) == 1
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == "be nice"


# ---------------------------------------------------------------------------
# save_conversation — create
# ---------------------------------------------------------------------------

def test_save_conversation_creates_row(db):
    from conversation_store import load_conversation, save_conversation

    save_conversation("sess-1", "My Chat", [{"type": "human", "content": "hi"}], [{"role": "user"}])

    loaded = load_conversation("sess-1")
    assert loaded is not None
    assert loaded["title"] == "My Chat"
    assert loaded["lc_messages"] == [{"type": "human", "content": "hi"}]
    assert loaded["ui_messages"] == [{"role": "user"}]


def test_save_conversation_defaults_user_to_admin(db):
    from conversation_store import load_conversation, save_conversation

    save_conversation("sess-default-user", "Title", [], [])

    loaded = load_conversation("sess-default-user")
    assert loaded["user"] == "admin"


def test_save_conversation_accepts_explicit_user(db):
    from conversation_store import load_conversation, save_conversation

    save_conversation("sess-explicit-user", "Title", [], [], user="alice")

    loaded = load_conversation("sess-explicit-user")
    assert loaded["user"] == "alice"


def test_save_conversation_sets_created_and_updated_at(db):
    from conversation_store import load_conversation, save_conversation

    save_conversation("sess-timestamps", "Title", [], [])

    loaded = load_conversation("sess-timestamps")
    assert loaded["created_at"]
    assert loaded["updated_at"]


def test_save_conversation_rejects_invalid_session_id(db):
    from conversation_store import save_conversation

    with pytest.raises(ValueError):
        save_conversation("bad id!", "Title", [], [])


# ---------------------------------------------------------------------------
# save_conversation — update (existing row)
# ---------------------------------------------------------------------------

def test_save_conversation_updates_existing_row(db):
    from conversation_store import load_conversation, save_conversation

    save_conversation("sess-update", "First Title", [], [])
    save_conversation("sess-update", "Second Title", [{"type": "human", "content": "q"}], [{"a": 1}])

    loaded = load_conversation("sess-update")
    assert loaded["title"] == "Second Title"
    assert loaded["lc_messages"] == [{"type": "human", "content": "q"}]
    assert loaded["ui_messages"] == [{"a": 1}]


def test_save_conversation_preserves_created_at_on_update(db):
    from conversation_store import load_conversation, save_conversation

    save_conversation("sess-preserve", "First", [], [])
    first_created = load_conversation("sess-preserve")["created_at"]

    save_conversation("sess-preserve", "Second", [], [])
    second_created = load_conversation("sess-preserve")["created_at"]

    assert first_created == second_created


def test_save_conversation_updates_updated_at_timestamp(db):
    from conversation_store import load_conversation, save_conversation

    save_conversation("sess-touch", "First", [], [])
    first_updated = load_conversation("sess-touch")["updated_at"]

    save_conversation("sess-touch", "Second", [], [])
    second_updated = load_conversation("sess-touch")["updated_at"]

    assert second_updated >= first_updated


def test_save_conversation_does_not_duplicate_rows(db):
    """Saving the same session_id twice must update in place, not insert a second row."""
    from conversation_store import list_conversations, save_conversation

    save_conversation("sess-nodup", "First", [], [])
    save_conversation("sess-nodup", "Second", [], [])

    matches = [c for c in list_conversations() if c["id"] == "sess-nodup"]
    assert len(matches) == 1


# ---------------------------------------------------------------------------
# save_conversation — DB error handling
# ---------------------------------------------------------------------------

def test_save_conversation_reraises_and_rolls_back_on_db_error(monkeypatch):
    from conversation_store import save_conversation

    session = _FailingSession(fail_on="commit", get_return=None)
    _patch_write(monkeypatch, lambda: session)

    with pytest.raises(SQLAlchemyError):
        save_conversation("sess-err", "Title", [], [])

    assert session.rolled_back is True


# ---------------------------------------------------------------------------
# load_conversation
# ---------------------------------------------------------------------------

def test_load_conversation_returns_none_for_unknown_session(db):
    from conversation_store import load_conversation
    assert load_conversation("does-not-exist") is None


def test_load_conversation_returns_none_for_invalid_session_id(db):
    from conversation_store import load_conversation
    assert load_conversation("../etc/passwd") is None


def test_load_conversation_returns_none_on_db_error(monkeypatch):
    from conversation_store import load_conversation

    _patch_read(monkeypatch, lambda: _FailingSession(fail_on="get"))
    assert load_conversation("sess-1") is None


# ---------------------------------------------------------------------------
# list_conversations
# ---------------------------------------------------------------------------

def test_list_conversations_empty(db):
    from conversation_store import list_conversations
    assert list_conversations() == []


def test_list_conversations_sorted_by_updated_at_desc(db):
    from conversation_store import list_conversations, save_conversation

    save_conversation("sess-a", "A", [], [])
    save_conversation("sess-b", "B", [], [])
    save_conversation("sess-a", "A updated", [], [])  # bump sess-a's updated_at

    ids = [c["id"] for c in list_conversations()]
    assert ids[0] == "sess-a"
    assert "sess-b" in ids


def test_list_conversations_filters_by_user(db):
    from conversation_store import list_conversations, save_conversation

    save_conversation("sess-admin", "Admin chat", [], [], user="admin")
    save_conversation("sess-alice", "Alice chat", [], [], user="alice")

    admin_ids = {c["id"] for c in list_conversations(user="admin")}
    alice_ids = {c["id"] for c in list_conversations(user="alice")}

    assert admin_ids == {"sess-admin"}
    assert alice_ids == {"sess-alice"}


def test_list_conversations_returns_expected_fields(db):
    from conversation_store import list_conversations, save_conversation

    save_conversation("sess-fields", "Title", [], [])
    entry = list_conversations()[0]
    assert set(entry.keys()) == {"id", "title", "created_at", "updated_at"}


def test_list_conversations_returns_empty_list_on_db_error(monkeypatch):
    from conversation_store import list_conversations

    _patch_read(monkeypatch, lambda: _FailingSession(fail_on="execute"))
    assert list_conversations() == []


# ---------------------------------------------------------------------------
# delete_conversation
# ---------------------------------------------------------------------------

def test_delete_conversation_removes_row(db):
    from conversation_store import delete_conversation, load_conversation, save_conversation

    save_conversation("sess-del", "Title", [], [])
    result = delete_conversation("sess-del")

    assert result is True
    assert load_conversation("sess-del") is None


def test_delete_conversation_returns_false_when_missing(db):
    from conversation_store import delete_conversation
    assert delete_conversation("does-not-exist") is False


def test_delete_conversation_returns_false_for_invalid_session_id(db):
    from conversation_store import delete_conversation
    assert delete_conversation("../etc/passwd") is False


def test_delete_conversation_returns_false_on_db_error(monkeypatch):
    from conversation_store import delete_conversation

    session = _FailingSession(fail_on="execute")
    _patch_write(monkeypatch, lambda: session)

    assert delete_conversation("sess-1") is False
    assert session.rolled_back is True


# ---------------------------------------------------------------------------
# update_title
# ---------------------------------------------------------------------------

def test_update_title_changes_title(db):
    from conversation_store import load_conversation, save_conversation, update_title

    save_conversation("sess-rename", "Old Title", [], [])
    result = update_title("sess-rename", "New Title")

    assert result is True
    assert load_conversation("sess-rename")["title"] == "New Title"


def test_update_title_bumps_updated_at(db):
    from conversation_store import load_conversation, save_conversation, update_title

    save_conversation("sess-rename-ts", "Old Title", [], [])
    before = load_conversation("sess-rename-ts")["updated_at"]

    update_title("sess-rename-ts", "New Title")

    after = load_conversation("sess-rename-ts")["updated_at"]
    assert after >= before


def test_update_title_returns_false_when_missing(db):
    from conversation_store import update_title
    assert update_title("does-not-exist", "New Title") is False


def test_update_title_returns_false_for_invalid_session_id(db):
    from conversation_store import update_title
    assert update_title("../etc/passwd", "New Title") is False


def test_update_title_returns_false_on_db_error(monkeypatch):
    from conversation_store import update_title

    fake_conv = MagicMock(title="Old", updated_at=None)
    session = _FailingSession(fail_on="commit", get_return=fake_conv)
    _patch_write(monkeypatch, lambda: session)

    assert update_title("sess-1", "New Title") is False
    assert session.rolled_back is True


# ---------------------------------------------------------------------------
# Multi-user isolation (forward-looking — no auth module yet)
# ---------------------------------------------------------------------------

def test_save_conversation_does_not_change_owner_of_existing_row(db):
    """id is the primary key, so a second save() with the SAME session_id but
    a DIFFERENT `user` updates the existing row in place — it does not create
    a second, per-user row, and (deliberately) does not reassign ownership
    either: save_conversation only ever touches title/messages/timestamps on
    an update, never `user`. Flagging this explicitly now, since it's the
    kind of thing that becomes a surprise (or a security question) the moment
    a real auth module starts calling save_conversation with varying users.
    """
    from conversation_store import list_conversations, save_conversation

    save_conversation("shared-id", "Admin's chat", [], [], user="admin")
    save_conversation("shared-id", "Alice's chat", [], [], user="alice")

    admin_matches = [c for c in list_conversations(user="admin") if c["id"] == "shared-id"]
    alice_matches = [c for c in list_conversations(user="alice") if c["id"] == "shared-id"]

    assert len(admin_matches) == 1
    assert admin_matches[0]["title"] == "Alice's chat"  # title WAS updated
    assert alice_matches == []  # ownership was NOT transferred