# AI Agent Backend

A FastAPI backend for a tool-calling AI agent, built on **LangChain** / **LangGraph**, with streaming responses over WebSocket, a local **RAG** (Retrieval-Augmented Generation) knowledge base backed by **ChromaDB**, persistent conversation history, and a configurable tool belt (shell, file ops, web search/scrape, calculator, knowledge-base search).

The agent currently ships with a system prompt oriented around teaching Kubernetes concepts, but the architecture is generic — swap the system prompt and tools to repurpose it for any domain.

## Contents
- [AI Agent Backend](#ai-agent-backend)
  - [Contents](#contents)
  - [Architecture](#architecture)
  - [Project layout](#project-layout)
  - [Getting started](#getting-started)
  - [Configuration](#configuration)
  - [API](#api)
  - [WebSocket protocol](#websocket-protocol)
  - [Tools](#tools)
  - [RAG pipeline](#rag-pipeline)
  - [Agent skills](#agent-skills)
  - [Conversation persistence](#conversation-persistence)
  - [Logging](#logging)
  - [Testing](#testing)
  - [Docker / deployment](#docker--deployment)

## Architecture

```
Client (WebSocket) ──▶ main.py ──▶ agent.py ──▶ LangGraph agent (create_agent)
                          │                         │
                          │                         ├─ ChatGoogleGenerativeAI (LLM)
                          │                         ├─ ALL_TOOLS (tools/*)
                          │                         └─ RAG context (rag/retriever.py)
                          │
                          ├─ conversation_store.py  (JSON files on disk)
                          ├─ config.py              (env-driven settings, hot-reloadable)
                          └─ rag/watcher.py         (auto-ingests rag_docs/ on change)
```

- **`main.py`** — FastAPI app. Exposes REST endpoints for config/tools/conversations and a single WebSocket endpoint that drives the actual chat turns. On startup it ingests any new/changed files in `rag_docs/` and starts a filesystem watcher for future changes.
- **`agent.py`** — Builds a LangGraph tool-calling agent per turn (`create_agent`), streams `astream_events` back over the WebSocket as `token` / `tool_start` / `tool_end` / `done` events, injects RAG context and skill definitions into the system prompt, and compacts (summarizes) chat history once it exceeds a configurable number of turns.
- **`config.py`** — A `pydantic-settings` `Settings` singleton loaded from `.env`. Supports safe runtime updates (`update_env_file`) so the frontend can change model/tooling without a restart.
- **`tools/`** — The agent's tool belt (see [Tools](#tools)).
- **`rag/`** — Document ingestion, embeddings, Chroma vectorstore access, retrieval, and a filesystem watcher (see [RAG pipeline](#rag-pipeline)).
- **`conversation_store.py`** — Simple JSON-file persistence for chat sessions.
- **`logging_setup.py`** — Structured, single-line JSON logging to stdout, designed for Kubernetes/Loki (separate log streams for app logs, LLM request/response traffic, and shell-command audit logs).
- **`skills_loader.py`** — Loads markdown files from `skills/` and appends them to the system prompt, letting you extend agent behavior without touching code.

## Project layout

```
backend/
├── main.py                 # FastAPI app, REST + WebSocket endpoints
├── agent.py                 # LangGraph agent construction & turn execution
├── config.py                 # Settings (env-driven, hot-reloadable)
├── conversation_store.py     # Conversation persistence (JSON files)
├── logging_setup.py          # Structured JSON logging
├── skills_loader.py           # Loads skills/*.md into the system prompt
├── tools/                    # Agent tools (calculator, file_ops, shell, web_search, web_scrape, rag_search)
├── rag/                      # RAG: ingestor, embeddings, chroma_client, retriever, watcher
├── skills/                   # Markdown skill definitions injected into the system prompt
├── rag_docs/                  # Drop .txt/.md/.pdf here to auto-ingest into the knowledge base
├── workspace/                 # Sandboxed directory for the file_ops tool
├── conversations/             # Persisted chat sessions (JSON)
├── chroma_db/                 # Persistent Chroma vector store
├── logs/                      # JSONL logs
├── tests/                     # pytest suite
├── Dockerfile
├── requirements.txt
├── .env.example
└── AGENTS.md                 # Conventions & commands for AI coding agents working in this repo
```

## Getting started

**Requirements:** Python 3.11+ (Dockerfile targets `python:3.11-slim`).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env             # then fill in the values below
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`, with the chat WebSocket at `ws://127.0.0.1:8000/ws/{session_id}`.

## Configuration

Settings are defined in `config.py` (`pydantic-settings`) and read from `backend/.env`. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | `""` | API key for the Gemini model used by the agent (`ChatGoogleGenerativeAI`). |
| `MODEL` | `gemini-2.0-flash` | Chat model name. Changeable at runtime via `PUT /api/config`. |
| `OPENROUTER_API_KEY` | `sk-placeholder` | Present for OpenRouter-compatible usage (see `.env.example`). |
| `OPENAI_API_KEY` | `""` | Used for embeddings (`rag/embeddings.py`) when set; otherwise falls back to an OpenRouter-compatible embeddings endpoint. |
| `APP_API_KEY` | — | Shared secret for REST + WebSocket auth (see `.env.example`; sent as `x-api-key` header / `?api_key=` query param by the frontend). |
| `CMD_MODE` | `permission` | `permission` requires explicit user approval before the shell tool runs a command; `bypass` skips the gate. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |
| `WORKSPACE_DIR` | `./workspace` | Sandbox root for `file_ops` tools — paths outside it are rejected. |
| `LOGS_DIR` | `./logs` | Where JSONL logs are written. |
| `RAG_DOCS_DIR` | `./rag_docs` | Source directory auto-ingested into the vector store. |
| `SKILLS_DIR` | `./skills` | Markdown skill files appended to the system prompt. |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Chroma vector store location. |
| `RAG_COLLECTION_NAME` | `default` | Chroma collection name. |
| `RAG_TOP_K` | `5` | Number of chunks retrieved per query. |
| `RAG_SCORE_THRESHOLD` | `1.0` | Similarity-distance cutoff for retrieved chunks. |
| `CONVERSATIONS_DIR` | `./conversations` | Where chat sessions are persisted. |
| `ENABLED_TOOLS` | `None` (all enabled) | JSON list of tool names to restrict the agent to; `[]` disables all tools. |
| `HISTORY_MAX_TURNS` | `3` | Turn count that triggers history compaction (summarization). |
| `HISTORY_RECENT_TURNS` | `1` | Turns kept verbatim after compaction. |

All relative directory paths are resolved against the `backend/` project root regardless of the process's current working directory.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/config` | Current model, `cmd_mode`, and enabled tools (never exposes API keys). |
| `PUT` | `/api/config` | Update model / `cmd_mode` / enabled tools; persists to `.env` and hot-reloads settings. |
| `GET` | `/api/tools` | List all registered tools with enabled/disabled status. |
| `GET` | `/api/conversations` | List saved conversations, most recently updated first. |
| `PATCH` | `/api/conversations/{session_id}` | Rename a conversation. |
| `DELETE` | `/api/conversations/{session_id}` | Delete a conversation (from disk and in-memory history). |
| `WS` | `/ws/{session_id}` | Chat channel — see below. |

## WebSocket protocol

**Client → Server**

| `type` | Payload | Meaning |
|---|---|---|
| `message` (or legacy `token`) | `content`, optional `images: [{data, mime_type}]` | Send a user turn. |
| `permission_response` | `approved: bool` | Approve/deny a pending shell command. |
| `save_messages` | `messages`, `title` | Persist the current conversation. |
| `rename_conversation` | `title` | Rename the conversation. |

**Server → Client**

| `type` | Meaning |
|---|---|
| `connected` | Handshake ack with `session_id`. |
| `history_restored` | Sent if a saved conversation exists for this session. |
| `history_compacted` | Old turns were summarized; includes the `summary`. |
| `token` | Streamed text chunk from the model. |
| `tool_start` / `tool_end` | Tool invocation lifecycle, with `tool` name and `input`/`output`. |
| `error` | Unhandled exception during the turn. |
| `done` | Turn complete. |

Note: `AGENTS.md` also lists a `permission_request` server event for the shell approval gate (emitted from `tools/shell.py`).

## Tools

Registered in `tools/__init__.py` (`ALL_TOOLS`), restrictable via `ENABLED_TOOLS`:

- **`calculator`** — safe arithmetic/math evaluation via `numexpr`.
- **`read_file` / `write_file` / `list_directory` / `delete_file`** — sandboxed to `WORKSPACE_DIR`; paths outside it are rejected.
- **`run_command`** — shell execution gated by `CMD_MODE`; in `permission` mode it blocks on a `permission_request`/`permission_response` round-trip over the WebSocket, and every attempt is written to a dedicated shell audit log.
- **`web_search`** — DuckDuckGo search (no API key required).
- **`scrape_url`** — fetches a URL and returns cleaned plain text; blocks loopback/private/link-local IPs (SSRF protection) and caps redirects/response size.
- **`search_knowledge_base`** — queries the RAG vector store (see below).

## RAG pipeline

1. Drop `.txt`, `.md`, or `.pdf` files into `rag_docs/`.
2. On startup, `main.py` scans that directory and ingests anything not already recorded in the ingestion manifest (hash-based, so unchanged files are skipped).
3. A `watchdog`-based observer (`rag/watcher.py`) keeps ingesting new/changed files and removing chunks for deleted files while the app is running.
4. `rag/ingestor.py` splits documents with a `RecursiveCharacterTextSplitter` and upserts them into Chroma with deterministic chunk IDs, so re-ingesting a source replaces rather than duplicates its chunks.
5. Each agent turn, `rag/retriever.py` embeds the user's message and pulls the top-`RAG_TOP_K` chunks (above `RAG_SCORE_THRESHOLD`) into a `SystemMessage` prepended to the model input. The same retrieval is also exposed directly to the agent as the `search_knowledge_base` tool.

## Agent skills

Any `.md` file placed in `skills/` is loaded by `skills_loader.py` and appended to the system prompt under an `## Agent Skills` section — a lightweight way to extend the agent's instructions/behavior without redeploying code.

## Conversation persistence

Each session is stored as `conversations/{session_id}.json` containing the title, timestamps, serialized LangChain messages (`lc_messages`, used to restore agent context) and raw UI messages (`ui_messages`, used to restore the frontend view). Session IDs are validated against `^[a-zA-Z0-9_-]{1,128}$` to prevent path traversal.

## Logging

`logging_setup.py` emits structured, single-line JSON log records to stdout — convenient for Kubernetes + Loki/Promtail pipelines. Three log streams:

- `app` — general application logs.
- `llm.traffic` — every LLM request/response/error (including history-compaction calls), tagged with `turn_id`/`session_id`.
- `audit.shell` — every shell command attempted, whether executed, blocked, or denied.

## Testing

```bash
pytest tests/
```

Covers config, the agent, RAG (search/retriever/ingestor/watcher), and each tool (calculator, file ops, shell, web search, web scrape).

## Docker / deployment

The `Dockerfile` is a two-stage build (build deps → slim runtime), runs as a non-root user, and does **not** bake in `.env` — configuration is expected via environment variables (e.g. Kubernetes Secrets/ConfigMaps). Data directories (`chroma_db/`, `conversations/`, `logs/`, `rag_docs/`, `skills/`, `workspace/`) are pre-created for volume mounts.

```bash
docker build -t ai-agent-backend .
docker run -p 8000:8000 --env-file .env ai-agent-backend
```

See `AGENTS.md` for day-to-day conventions (build/run/test commands, workspace boundaries, WebSocket message types) when working in this codebase.