"""create conversations table

Revision ID: 0001
Revises:
Create Date: 2026-07-03 00:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(length=128), primary_key=True),
        sa.Column("user", sa.String(length=128), nullable=False, server_default="admin"),
        sa.Column("title", sa.String(), nullable=False, server_default="Untitled"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "lc_messages",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "ui_messages",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
    )
    op.create_index("ix_conversations_user", "conversations", ["user"])
    op.create_index("ix_conversations_updated_at", "conversations", ["updated_at"])


def downgrade() -> None:
    op.drop_index("ix_conversations_updated_at", table_name="conversations")
    op.drop_index("ix_conversations_user", table_name="conversations")
    op.drop_table("conversations")