"""Add data_quality_flags table and rows_new/rows_revised to ingestion_run_items.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-10-15

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "ingestion_run_items",
        sa.Column("rows_new", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.add_column(
        "ingestion_run_items",
        sa.Column("rows_revised", sa.BigInteger(), nullable=False, server_default="0"),
    )

    op.create_table(
        "data_quality_flags",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.BigInteger(),
            sa.ForeignKey("ingestion_runs.id"),
            nullable=True,
        ),
        sa.Column(
            "instrument_id",
            sa.String(64),
            sa.ForeignKey("instruments.id"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("obs_date", sa.Date(), nullable=False),
        sa.Column("rule", sa.String(64), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
    )
    op.create_index(
        "ix_data_quality_flags_run_id",
        "data_quality_flags",
        ["run_id"],
    )
    op.create_index(
        "ix_data_quality_flags_instrument",
        "data_quality_flags",
        ["instrument_id", "provider", "obs_date"],
    )


def downgrade() -> None:
    op.drop_index("ix_data_quality_flags_instrument", table_name="data_quality_flags")
    op.drop_index("ix_data_quality_flags_run_id", table_name="data_quality_flags")
    op.drop_table("data_quality_flags")
    op.drop_column("ingestion_run_items", "rows_revised")
    op.drop_column("ingestion_run_items", "rows_new")
