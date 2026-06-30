# AI Agent — Coding Agent Instructions

This file provides essential instructions and conventions for AI coding agents working in this project. It summarizes build/test commands, architecture, key settings, and important constraints to ensure agents are immediately productive and avoid common pitfalls.

## Project Structure
- **Backend**: Python (FastAPI, LangChain, ChromaDB) — all in `backend/`
- **Frontend**: React (Vite), Tailwind CSS — all in `frontend/`
- **Vector DB**: ChromaDB (persistent at `backend/chroma_db/`)
- **Workspace boundary**: All file operations are sandboxed to `backend/workspace/`.
- **Config/secrets**: `.env` lives at `backend/.env` (loaded automatically by `config.py`).

## Build & Run Commands
- **Backend**:
  - Setup: `cd backend && python -m venv .venv && .venv\Scripts\Activate.ps1 && pip install -r requirements.txt`
  - Run: `python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000`
- **Frontend**:
  - Setup: `cd frontend && npm install`
  - Run: `npm run dev` (dev server at :5173, proxies to backend :8000)
- **All**: `./run_all.ps1` (runs both frontend and backend)

## Test Commands
- **Backend**: `pytest backend/tests/`
- **Frontend**: `npm test` in `frontend/`

## Key Conventions & Constraints
- **File operations**: Only allowed within `workspace/` (see `file_ops.py`). Rejects paths outside this directory.
- **Shell tool**: Permission gate enforced (`CMD_MODE` in `.env` or config). Requires explicit user approval unless in `bypass` mode.
- **Config**: Environment variables loaded from `.env` at project root. See `backend/config.py` for details.
- **RAG pipeline**: Documents in `rag_docs/` are auto-ingested at startup and live via watcher. Only new/changed files are processed (hash-based deduplication).
- **Logs**: LLM request/response logs are stored as JSONL in `logs/`.

## WebSocket Protocol
- **Server → Client**: `connected`, `token`, `tool_start`, `tool_end`, `permission_request`, `done`, `error`
- **Client → Server**: `message`, `permission_response`

## Useful Links
- [README.md](README.md) — Full project overview, stack, and directory layout

---

This file should be updated as project conventions or architecture evolve. For detailed documentation, always refer to the linked README.
