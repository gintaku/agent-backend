# ── Stage 1: build deps ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some native extensions (psycopg, sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source (exclude data dirs — those are mounted via PVC in k8s)
COPY agent.py config.py conversation_store.py main.py skills_loader.py ./
COPY rag/ ./rag/
COPY tools/ ./tools/
COPY db/ ./db/

# Alembic migrations for the conversations table (same Postgres instance as
# RAG). Run `alembic upgrade head` as a release step / init container before
# the app starts — this image intentionally does not auto-migrate on boot.
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Pre-create data directories so the app can start without volumes attached.
# conversations/ is no longer written to (that data now lives in Postgres —
# see conversation_store.py) but is left here harmlessly in case anything
# still expects the path to exist.
RUN mkdir -p conversations logs rag_docs skills workspace

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Secrets come from k8s Secrets/ConfigMap via environment variables.
# The .env file is NOT baked in — config.py reads env vars at runtime.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]