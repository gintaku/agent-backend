"""LangChain tool-calling agent with per-session chat history and WebSocket streaming."""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings
from rag.retriever import retrieve as rag_retrieve
from skills_loader import load_skills
from tools import ALL_TOOLS

logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Per-session chat history  {session_id: [HumanMessage, AIMessage, ...]}
# ---------------------------------------------------------------------------

_chat_histories: dict[str, list[BaseMessage]] = {}

SYSTEM_PROMPT = (
    "You are a helpful AI assistant who will teach users about Kubernetes and related technologies. "
    """ Use below visual to provide more understandable answer if possible.
    +------------+             (1) Request Login
|            | -----------------------------------------> +-------------------+
|            |                                            |                   |
|            |             (2) Returns BOTH Tokens        |    Identity       |
|    Your    | <========================================= |    Provider       |
|    App     |     [ ID Token ]       [ Access Token ]    |     (IDP)         |
|  (Client)  |          |                     |           |                   |
|            |          |                     |           +-------------------+
|            |          v                     |
|            |   App reads this to             |
|            |   know WHO logged in            v
+------------+                          (3) Sends Access Token
                                              to fetch data
                                              |
                                              v
                                  +-------------------+
                                  |  Resource Server  |
                                  |     (API)         |
                                  +-------------------+
                                  """
)


#: Dedicated logger for LLM request/response traffic. Kept separate from the
#: "app" logger (see logging_setup.LOG_TYPES) so it can be routed to its own
#: Loki bucket via a pipeline stage that promotes log_type to a stream label.
_llm_logger = logging.getLogger("llm.traffic")


def _serialize_message(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message into a JSON-serializable payload."""
    payload: dict[str, Any] = {
        "type": message.type,
        "content": message.content,
    }

    name = getattr(message, "name", None)
    if name:
        payload["name"] = name

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = tool_calls

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if additional_kwargs:
        payload["additional_kwargs"] = additional_kwargs

    return payload


def _serialize_for_log(value: Any) -> Any:
    """Recursively convert event payloads into JSON-serializable data."""
    if isinstance(value, BaseMessage):
        return _serialize_message(value)
    if isinstance(value, list):
        return [_serialize_for_log(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_for_log(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_for_log(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _log_llm_traffic(
    *,
    turn_id: str,
    session_id: str,
    direction: str,
    provider: str,
    payload: Any = None,
    error: str | None = None,
) -> None:
    """Emit one structured LLM request/response/error event.

    Goes through the standard logging pipeline (JSON to stdout) rather than a
    hand-rolled file, so it gets the same timestamp/session_id handling as
    everything else, doesn't block the event loop on file I/O, and — via the
    "llm.traffic" logger name — can be routed to its own Loki stream.
    """
    extra: dict[str, Any] = {
        "turn_id": turn_id,
        "session_id": session_id,
        "direction": direction,
        "provider": provider,
        "model": get_settings().model,
    }
    if payload is not None:
        extra["payload"] = _serialize_for_log(payload)
    if error is not None:
        extra["error"] = error
    _llm_logger.info("llm_%s", direction, extra=extra)


def get_history(session_id: str) -> list[BaseMessage]:
    """Return (and lazily create) the message history for *session_id*."""
    if session_id not in _chat_histories:
        _chat_histories[session_id] = []
    return _chat_histories[session_id]


def clear_history(session_id: str) -> None:
    """Remove all history for *session_id*."""
    _chat_histories.pop(session_id, None)


# ---------------------------------------------------------------------------
# History compaction
# ---------------------------------------------------------------------------

async def _maybe_compact_history(
    session_id: str,  # noqa: ARG001 — reserved for future per-session logging
    history: list[BaseMessage],
    llm: ChatGoogleGenerativeAI,
) -> str | None:
    """Summarize the oldest turns when the history exceeds the configured limit.

    Returns the summary text if compaction was performed, None otherwise.
    The *history* list is mutated in-place.
    """
    s = get_settings()
    max_turns = s.history_max_turns
    recent_turns = s.history_recent_turns

    # Separate a leading summary SystemMessage (from a prior compaction) from
    # the conversational HumanMessage/AIMessage pairs.
    prior_summary: SystemMessage | None = None
    conv_messages: list[BaseMessage] = []
    for msg in history:
        if not conv_messages and isinstance(msg, SystemMessage):
            prior_summary = msg
        else:
            conv_messages.append(msg)

    # Count only complete Human/AI pairs.
    pair_count = len(conv_messages) // 2
    if pair_count < max_turns:
        return None

    # Keep only the most-recent N turns verbatim.
    keep = recent_turns * 2  # messages per pair
    older = conv_messages[:-keep] if keep > 0 else conv_messages
    recent_pairs = conv_messages[-keep:] if keep > 0 else []

    # Build the summarization prompt.
    prior_text = (
        f"{prior_summary.content}\n\n" if prior_summary else ""
    )
    older_text = "\n".join(
        f"{'Human' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in older
    )
    prompt = (
        f"{prior_text}"
        "Summarize the following conversation concisely, preserving all key facts, "
        "decisions, and context that may be relevant later:\n\n"
        f"{older_text}"
    )

    summarize_turn_id = str(uuid4())
    summarize_messages = [HumanMessage(content=prompt)]
    _log_llm_traffic(
        turn_id=summarize_turn_id,
        session_id=session_id,
        direction="request",
        provider="summarize",
        payload={"messages": [summarize_messages]},
    )

    summary_response = await llm.ainvoke(summarize_messages)

    # Extract plain text, discarding any extras/signature metadata that the
    # model may embed in structured content blocks.
    if isinstance(summary_response.content, str):
        summary_text = summary_response.content.strip()
    elif isinstance(summary_response.content, list):
        summary_text = "".join(
            part.get("text", "")
            for part in summary_response.content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    else:
        summary_text = str(summary_response.content).strip()

    _log_llm_traffic(
        turn_id=summarize_turn_id,
        session_id=session_id,
        direction="response",
        provider="summarize",
        payload={"type": "ai", "content": summary_text},
    )

    history[:] = [SystemMessage(content=f"Conversation summary:\n{summary_text}")] + recent_pairs
    return summary_text



def _make_llm() -> ChatGoogleGenerativeAI:
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=s.model,
        google_api_key=s.google_api_key,
        temperature=0,
    )


def _make_agent(llm: ChatGoogleGenerativeAI, system_prompt: str = SYSTEM_PROMPT):
    """Return a compiled LangGraph agent graph using only currently enabled tools."""
    s = get_settings()
    if s.enabled_tools is not None:
        enabled = set(s.enabled_tools)
        active_tools = [t for t in ALL_TOOLS if t.name in enabled]
    else:
        active_tools = ALL_TOOLS
    return create_agent(model=llm, tools=active_tools, system_prompt=system_prompt)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_agent(
    session_id: str,
    user_message: str,
    ws_send: Callable[[dict[str, Any]], Awaitable[None]],
    images: list[dict[str, str]] | None = None,
) -> None:
    """Run one agent turn, streaming events back over the WebSocket.

    Args:
        images: Optional list of dicts, each with keys ``data`` (base64 string)
                and ``mime_type`` (e.g. ``"image/jpeg"``).

    Emits these message types via *ws_send*:
      ``{"type": "token",      "content": "..."}``           — streaming text chunk
      ``{"type": "tool_start", "tool": "name", "input": {}}`` — tool invocation begins
      ``{"type": "tool_end",   "tool": "name", "output": ""}`` — tool invocation ends
      ``{"type": "error",      "content": "..."}``            — unhandled exception
      ``{"type": "done"}``                                     — turn complete
    """
    from tools.shell import session_id_var, ws_send_var

    history = get_history(session_id)

    llm = _make_llm()

    # Compact history before processing this new message so the summary is
    # available to the user immediately when the turn begins.
    summary_text = await _maybe_compact_history(session_id, history, llm)
    if summary_text is not None:
        await ws_send({"type": "history_compacted", "summary": summary_text})

    # Build system prompt: base prompt + any skill definitions.
    skills_content = load_skills()
    full_system_prompt = SYSTEM_PROMPT
    if skills_content:
        full_system_prompt = SYSTEM_PROMPT + "\n\n" + skills_content

    graph = _make_agent(llm, system_prompt=full_system_prompt)

    # Inject context variables consumed by the shell permission gate.
    token_sid = session_id_var.set(session_id)
    token_ws = ws_send_var.set(ws_send)

    final_answer = ""
    error_occurred = False
    turn_id = uuid4().hex
    logger.info("Agent turn started", extra={"session_id": session_id, "turn_id": turn_id})

    # Build the input messages: existing history + new human turn.
    valid_images = [
        img for img in (images or []) if img.get("data") and img.get("mime_type")
    ]
    if valid_images:
        parts: list[Any] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{img['mime_type']};base64,{img['data']}"},
            }
            for img in valid_images
        ]
        if user_message:
            parts.append({"type": "text", "text": user_message})
        human_msg: BaseMessage = HumanMessage(content=parts)
    else:
        human_msg = HumanMessage(content=user_message)
    input_messages = list(history) + [human_msg]

    # RAG context injection — prepend relevant knowledge-base chunks.
    rag_context = rag_retrieve(user_message)
    if rag_context:
        input_messages.insert(
            0,
            SystemMessage(
                content=f"Relevant context from knowledge base:\n{rag_context}"
            ),
        )

    try:
        async for event in graph.astream_events(
            {"messages": input_messages},
            version="v2",
        ):
            kind = event["event"]
            name = event.get("name", "")

            if kind == "on_chat_model_start":
                _log_llm_traffic(
                    turn_id=turn_id,
                    session_id=session_id,
                    direction="request",
                    provider=name,
                    payload=event["data"].get("input", {}),
                )

            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = chunk.content
                if isinstance(content, str) and content:
                    await ws_send({"type": "token", "content": content})
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                await ws_send({"type": "token", "content": text})

            elif kind == "on_chat_model_end":
                _log_llm_traffic(
                    turn_id=turn_id,
                    session_id=session_id,
                    direction="response",
                    provider=name,
                    payload=event["data"].get("output"),
                )

            elif kind == "on_tool_start":
                await ws_send(
                    {
                        "type": "tool_start",
                        "tool": name,
                        "input": event["data"].get("input", {}),
                    }
                )

            elif kind == "on_tool_end":
                await ws_send(
                    {
                        "type": "tool_end",
                        "tool": name,
                        "output": str(event["data"].get("output", "")),
                    }
                )

            elif kind == "on_chain_end":
                # Extract the final text from the last AIMessage in the output.
                output = event["data"].get("output", {})
                if isinstance(output, dict):
                    messages_out = output.get("messages", [])
                    if messages_out:
                        last = messages_out[-1]
                        if isinstance(last, AIMessage):
                            if isinstance(last.content, str):
                                final_answer = last.content
                            elif isinstance(last.content, list):
                                final_answer = "".join(
                                    part.get("text", "")
                                    for part in last.content
                                    if isinstance(part, dict) and part.get("type") == "text"
                                )

    except Exception as exc:  # noqa: BLE001
        error_occurred = True
        _llm_logger.exception(
            "llm_error",
            extra={
                "turn_id": turn_id,
                "session_id": session_id,
                "direction": "error",
                "provider": "agent",
                "model": get_settings().model,
                "error": str(exc),
            },
        )
        await ws_send({"type": "error", "content": str(exc)})

    finally:
        session_id_var.reset(token_sid)
        ws_send_var.reset(token_ws)

    if not error_occurred:
        # Persist this turn in the session history.
        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=final_answer))
        logger.info("Agent turn completed", extra={"session_id": session_id, "turn_id": turn_id})
        await ws_send({"type": "done"})
