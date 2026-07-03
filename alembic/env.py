"""Alembic environment — migrates only the app-owned tables in db/models.py.

Deliberately does NOT import rag/pg_client.py or anything from
langchain-postgres: PGVector manages its own tables lazily and must never be
driven by this Alembic setup, or the two schema-management systems could
fight each other over the same database.

The connection DSN comes from config.get_settings().db_write_url (the same
primary DSN conversation_store.py writes through) rather than from
alembic.ini, so there's exactly one place (.env / environment variables)
that defines "where is the database", not two.
"""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Allow `alembic` (run from backend/) to import sibling modules.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from db.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    return get_settings().db_write_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection (`alembic upgrade head --sql`)."""
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection — the normal `alembic upgrade head` path."""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()