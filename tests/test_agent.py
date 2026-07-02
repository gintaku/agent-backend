"""Tests for backend/agent.py."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_astream_events(*events: dict):
    """Return a callable that yields *events* from an async generator."""
    async def _gen(*args: Any, **kwargs: Any):
        for e in events:
            yield e

    return _gen


def _make_chunk(text: str):
    chunk = MagicMock()
    chunk.content = text
    return chunk


def _chain_end_event(messages: list):
    """Build an on_chain_end event with a messages output."""
    return {
        "event": "on_chain_end",
        "name": "LangGraph",
        "data": {"output": {"messages": messages}},
    }


async def _collect(session_id: str, message: str) -> list[dict]:
    """Run agent and return every dict emitted via ws_send."""
    from agent import run_agent

    events: list[dict] = []

    async def ws_send(data: dict) -> None:
        events.append(data)

    await run_agent(session_id, message, ws_send)
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "test-model")
    import config
    config.reset_settings()
    yield
    config.reset_settings()


@pytest.fixture(autouse=True)
def _clear_history():
    yield
    import agent
    agent._chat_histories.clear()


# ---------------------------------------------------------------------------
# History management (sync)
# ---------------------------------------------------------------------------

def test_get_history_creates_empty_list_on_first_call():
    from agent import clear_history, get_history
    clear_history("new-session")
    history = get_history("new-session")
    assert history == []


def test_get_history_returns_same_list_on_repeated_calls():
    from agent import get_history
    h1 = get_history("same-session")
    h2 = get_history("same-session")
    assert h1 is h2


def test_clear_history_removes_session():
    from agent import clear_history, get_history
    h = get_history("to-clear")
    h.append(HumanMessage(content="hi"))
    clear_history("to-clear")
    assert get_history("to-clear") == []


def test_clear_history_noop_for_unknown_session():
    from agent import clear_history
    clear_history("does-not-exist")


# ---------------------------------------------------------------------------
# run_agent — event emission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_done_event_always_emitted():
    ai_msg = AIMessage(content="Hello!")
    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        MockMakeAgent.return_value.astream_events = _fake_astream_events(
            _chain_end_event([ai_msg])
        )
        events = await _collect("sess-done", "hi")

    assert events[-1] == {"type": "done"}


@pytest.mark.asyncio
async def test_token_events_forwarded():
    stream_event = {
        "event": "on_chat_model_stream",
        "name": "llm",
        "data": {"chunk": _make_chunk("Hello")},
    }
    ai_msg = AIMessage(content="Hello")
    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        MockMakeAgent.return_value.astream_events = _fake_astream_events(
            stream_event, _chain_end_event([ai_msg])
        )
        events = await _collect("sess-token", "hi")

    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 1
    assert token_events[0]["content"] == "Hello"


@pytest.mark.asyncio
async def test_empty_token_chunks_not_forwarded():
    """Chunks with empty content must be suppressed."""
    stream_event = {
        "event": "on_chat_model_stream",
        "name": "llm",
        "data": {"chunk": _make_chunk("")},
    }
    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        MockMakeAgent.return_value.astream_events = _fake_astream_events(
            stream_event, _chain_end_event([AIMessage(content="")])
        )
        events = await _collect("sess-empty", "hi")

    assert not any(e["type"] == "token" for e in events)


@pytest.mark.asyncio
async def test_tool_start_event_forwarded():
    tool_start = {
        "event": "on_tool_start",
        "name": "calculator",
        "data": {"input": {"expression": "2+2"}},
    }
    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        MockMakeAgent.return_value.astream_events = _fake_astream_events(
            tool_start, _chain_end_event([AIMessage(content="4")])
        )
        events = await _collect("sess-ts", "what is 2+2?")

    starts = [e for e in events if e["type"] == "tool_start"]
    assert len(starts) == 1
    assert starts[0]["tool"] == "calculator"
    assert starts[0]["input"] == {"expression": "2+2"}


@pytest.mark.asyncio
async def test_tool_end_event_forwarded():
    tool_end = {
        "event": "on_tool_end",
        "name": "calculator",
        "data": {"output": "4"},
    }
    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        MockMakeAgent.return_value.astream_events = _fake_astream_events(
            tool_end, _chain_end_event([AIMessage(content="4")])
        )
        events = await _collect("sess-te", "what is 2+2?")

    ends = [e for e in events if e["type"] == "tool_end"]
    assert len(ends) == 1
    assert ends[0]["tool"] == "calculator"
    assert ends[0]["output"] == "4"


@pytest.mark.asyncio
async def test_multiple_tokens_in_order():
    def _stream(text: str):
        return {
            "event": "on_chat_model_stream",
            "name": "llm",
            "data": {"chunk": _make_chunk(text)},
        }

    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        MockMakeAgent.return_value.astream_events = _fake_astream_events(
            _stream("Hi"), _stream(" "), _stream("there"),
            _chain_end_event([AIMessage(content="Hi there")])
        )
        events = await _collect("sess-multi-tok", "hello")

    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == ["Hi", " ", "there"]


# ---------------------------------------------------------------------------
# run_agent — chat history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_history_updated_after_turn():
    from agent import clear_history, get_history

    clear_history("sess-hist")
    ai_msg = AIMessage(content="World")
    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        MockMakeAgent.return_value.astream_events = _fake_astream_events(
            _chain_end_event([ai_msg])
        )
        await _collect("sess-hist", "Hello")

    history = get_history("sess-hist")
    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "Hello"
    assert isinstance(history[1], AIMessage)
    assert history[1].content == "World"


@pytest.mark.asyncio
async def test_history_accumulates_across_turns():
    from agent import clear_history, get_history

    clear_history("sess-accum")

    def _make_graph(answer: str):
        mock = MagicMock()
        mock.astream_events = _fake_astream_events(
            _chain_end_event([AIMessage(content=answer)])
        )
        return mock

    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        MockMakeAgent.side_effect = [_make_graph("A1"), _make_graph("A2")]
        await _collect("sess-accum", "Q1")
        await _collect("sess-accum", "Q2")

    history = get_history("sess-accum")
    assert len(history) == 4
    assert history[0].content == "Q1"
    assert history[1].content == "A1"
    assert history[2].content == "Q2"
    assert history[3].content == "A2"


@pytest.mark.asyncio
async def test_history_not_updated_on_error():
    from agent import clear_history, get_history

    clear_history("sess-err")

    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        async def _boom(*args, **kwargs):
            raise RuntimeError("LLM exploded")
            yield

        MockMakeAgent.return_value.astream_events = _boom
        events = await _collect("sess-err", "hello")

    assert any(e["type"] == "error" for e in events)
    assert get_history("sess-err") == []


# ---------------------------------------------------------------------------
# run_agent — error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_event_emitted_on_exception():
    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        async def _boom(*args, **kwargs):
            raise ValueError("something went wrong")
            yield

        MockMakeAgent.return_value.astream_events = _boom
        events = await _collect("sess-exc", "trigger error")

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "something went wrong" in error_events[0]["content"]


@pytest.mark.asyncio
async def test_done_not_emitted_on_error():
    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        async def _boom(*args, **kwargs):
            raise ValueError("oops")
            yield

        MockMakeAgent.return_value.astream_events = _boom
        events = await _collect("sess-no-done", "hi")

    assert not any(e["type"] == "done" for e in events)


@pytest.mark.asyncio
async def test_llm_request_and_response_are_logged(caplog):
    """LLM request/response traffic goes through the llm.traffic logger."""
    chat_start = {
        "event": "on_chat_model_start",
        "name": "ChatOpenAI",
        "data": {
            "input": {
                "messages": [
                    SystemMessage(content="You are a helpful AI assistant."),
                    HumanMessage(content="Log this prompt"),
                ]
            }
        },
    }
    chat_end = {
        "event": "on_chat_model_end",
        "name": "ChatOpenAI",
        "data": {"output": AIMessage(content="Logged answer")},
    }

    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
        caplog.at_level("INFO", logger="llm.traffic"),
    ):
        MockMakeAgent.return_value.astream_events = _fake_astream_events(
            chat_start,
            chat_end,
            _chain_end_event([AIMessage(content="Logged answer")])
        )
        await _collect("sess-log", "Log this prompt")

    records = [r for r in caplog.records if r.name == "llm.traffic"]
    assert [r.direction for r in records] == ["request", "response"]
    assert records[0].session_id == "sess-log"
    assert records[0].provider == "ChatOpenAI"
    assert records[0].payload["messages"][-1]["content"] == "Log this prompt"
    assert records[1].payload["content"] == "Logged answer"


# ---------------------------------------------------------------------------
# run_agent — chain_end with empty messages list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chain_end_with_no_ai_message():
    """on_chain_end with no messages still completes cleanly."""
    from agent import clear_history, get_history

    clear_history("sess-nomsg")
    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        MockMakeAgent.return_value.astream_events = _fake_astream_events(
            _chain_end_event([])
        )
        events = await _collect("sess-nomsg", "question")

    # done should still emit, history final answer is empty string
    assert events[-1] == {"type": "done"}
    history = get_history("sess-nomsg")
    assert history[1].content == ""


# ---------------------------------------------------------------------------
# _maybe_compact_history — unit tests
# ---------------------------------------------------------------------------

def _build_history(n_turns: int) -> list:
    """Build a history list with *n_turns* Human/AI pairs."""
    from langchain_core.messages import AIMessage, HumanMessage
    msgs = []
    for i in range(n_turns):
        msgs.append(HumanMessage(content=f"Q{i + 1}"))
        msgs.append(AIMessage(content=f"A{i + 1}"))
    return msgs


@pytest.mark.asyncio
async def test_no_compaction_below_threshold(monkeypatch):
    """History below history_max_turns must not be compacted."""
    from agent import _maybe_compact_history
    import config

    monkeypatch.setenv("HISTORY_MAX_TURNS", "10")
    config.reset_settings()
    history = _build_history(9)  # 9 pairs < 10
    mock_llm = MagicMock()

    result = await _maybe_compact_history("sess", history, mock_llm)

    assert result is None
    assert len(history) == 18  # unchanged
    mock_llm.ainvoke.assert_not_called()
    config.reset_settings()


@pytest.mark.asyncio
async def test_compaction_fires_at_threshold(monkeypatch):
    """Compaction must fire when pair count == history_max_turns."""
    from agent import _maybe_compact_history
    import config

    monkeypatch.setenv("HISTORY_MAX_TURNS", "5")
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "2")
    config.reset_settings()

    history = _build_history(5)  # exactly at threshold
    mock_llm = MagicMock()
    summary_msg = MagicMock()
    summary_msg.content = "Summary of older turns."

    async def fake_ainvoke(msgs):
        return summary_msg

    mock_llm.ainvoke = fake_ainvoke

    result = await _maybe_compact_history("sess", history, mock_llm)

    assert result == "Summary of older turns."
    config.reset_settings()


@pytest.mark.asyncio
async def test_recent_turns_preserved_verbatim(monkeypatch):
    """After compaction the last history_recent_turns pairs must survive unchanged."""
    from agent import _maybe_compact_history
    import config

    monkeypatch.setenv("HISTORY_MAX_TURNS", "4")
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "2")
    config.reset_settings()

    history = _build_history(4)  # Q1 A1 Q2 A2 Q3 A3 Q4 A4

    async def fake_ainvoke(msgs):
        m = MagicMock()
        m.content = "old summary"
        return m

    mock_llm = MagicMock()
    mock_llm.ainvoke = fake_ainvoke

    await _maybe_compact_history("sess", history, mock_llm)

    # history[0] is the new SystemMessage summary
    from langchain_core.messages import SystemMessage
    assert isinstance(history[0], SystemMessage)
    # The last 2 pairs (Q3 A3 Q4 A4) must be verbatim
    assert history[1].content == "Q3"
    assert history[2].content == "A3"
    assert history[3].content == "Q4"
    assert history[4].content == "A4"
    assert len(history) == 5
    config.reset_settings()


@pytest.mark.asyncio
async def test_existing_summary_included_in_prompt(monkeypatch):
    """A prior compaction's SystemMessage must be prepended to the new summarization prompt."""
    from agent import _maybe_compact_history
    import config

    monkeypatch.setenv("HISTORY_MAX_TURNS", "4")
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "2")
    config.reset_settings()

    prior = SystemMessage(content="Conversation summary:\nPrevious context.")
    history = [prior] + _build_history(4)

    captured_prompt = {}

    async def fake_ainvoke(msgs):
        captured_prompt["text"] = msgs[0].content
        m = MagicMock()
        m.content = "new summary"
        return m

    mock_llm = MagicMock()
    mock_llm.ainvoke = fake_ainvoke

    result = await _maybe_compact_history("sess", history, mock_llm)

    assert result == "new summary"
    assert "Conversation summary:\nPrevious context." in captured_prompt["text"]
    config.reset_settings()


@pytest.mark.asyncio
async def test_history_compacted_ws_event_emitted_before_done(monkeypatch):
    """history_compacted WS event must be sent before done when compaction fires."""
    from agent import clear_history, get_history
    import config

    monkeypatch.setenv("HISTORY_MAX_TURNS", "1")
    monkeypatch.setenv("HISTORY_RECENT_TURNS", "0")
    config.reset_settings()

    clear_history("sess-compact-ws")
    # Pre-populate 1 pair so compaction fires at the START of the next turn.
    hist = get_history("sess-compact-ws")
    hist.extend([HumanMessage(content="prior Q"), AIMessage(content="prior A")])

    ai_msg = AIMessage(content="reply")

    async def fake_ainvoke(msgs):
        m = MagicMock()
        m.content = "summary text"
        return m

    with (
        patch("agent._make_llm") as MockLLM,
        patch("agent._make_agent") as MockMakeAgent,
    ):
        mock_llm_instance = MagicMock()
        mock_llm_instance.ainvoke = fake_ainvoke
        MockLLM.return_value = mock_llm_instance
        MockMakeAgent.return_value.astream_events = _fake_astream_events(
            _chain_end_event([ai_msg])
        )
        events = await _collect("sess-compact-ws", "hello")

    types = [e["type"] for e in events]
    done_idx = types.index("done")
    assert "history_compacted" in types
    # compacted event must arrive BEFORE done
    assert types.index("history_compacted") < done_idx
    # summary text must be included in the event
    compact_event = next(e for e in events if e["type"] == "history_compacted")
    assert compact_event.get("summary") == "summary text"
    config.reset_settings()


@pytest.mark.asyncio
async def test_no_history_compacted_event_below_threshold():
    """history_compacted must NOT be emitted when compaction does not fire."""
    from agent import clear_history
    import config

    config.reset_settings()  # default max_turns=10
    clear_history("sess-no-compact")
    ai_msg = AIMessage(content="reply")

    with (
        patch("agent._make_llm"),
        patch("agent._make_agent") as MockMakeAgent,
    ):
        MockMakeAgent.return_value.astream_events = _fake_astream_events(
            _chain_end_event([ai_msg])
        )
        events = await _collect("sess-no-compact", "hello")

    assert not any(e["type"] == "history_compacted" for e in events)

