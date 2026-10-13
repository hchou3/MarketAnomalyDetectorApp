"""Initial schema: instruments, symbol_map, prices, labels, ingestion_runs.

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-10-13

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "instruments",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("asset_class", sa.String(32), nullable=False),
        sa.Column("units", sa.String(32), nullable=False),
        sa.Column("quote", sa.String(128), nullable=False),
        sa.Column("native_frequency", sa.String(16), nullable=False),
        sa.Column("valid_from", sa.Date(), nullable=True),
        sa.Column("valid_to", sa.Date(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )

    op.create_table(
        "instrument_groups",
        sa.Column("instrument_id", sa.String(64), nullable=False),
        sa.Column("group", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("instrument_id", "group"),
    )

    op.create_table(
        "symbol_map",
        sa.Column("instrument_id", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(128), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("fidelity", sa.String(16), nullable=False),
        sa.Column("source_frequency", sa.String(16), nullable=True),
        sa.Column("publication_lag_days", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("revised", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("instrument_id", "provider"),
    )

    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("instruments_ok", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("instruments_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rows_written", sa.BigInteger(), nullable=False, server_default="0"),
    )

    op.create_table(
        "prices",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("instrument_id", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("obs_date", sa.Date(), nullable=False),
        sa.Column("value", sa.Numeric(20, 8), nullable=False),
        sa.Column("known_at", sa.Date(), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(["instrument_id"], ["instruments.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["ingestion_runs.id"]),
        sa.UniqueConstraint("instrument_id", "provider", "obs_date"),
    )
    op.create_index(
        "ix_prices_instrument_provider_date",
        "prices",
        ["instrument_id", "provider", sa.text("obs_date DESC")],
    )

    op.create_table(
        "labels",
        sa.Column("label_set", sa.String(64), nullable=False),
        sa.Column("week", sa.Date(), nullable=False),
        sa.Column("value", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(128), nullable=False),
        sa.PrimaryKeyConstraint("label_set", "week"),
    )

    op.create_table(
        "ingestion_run_items",
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("instrument_id", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("rows_written", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("first_date", sa.Date(), nullable=True),
        sa.Column("last_date", sa.Date(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["ingestion_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "instrument_id", "provider"),
    )


def downgrade() -> None:
    op.drop_table("ingestion_run_items")
    op.drop_table("labels")
    op.drop_index("ix_prices_instrument_provider_date", table_name="prices")
    op.drop_table("prices")
    op.drop_table("ingestion_runs")
    op.drop_table("symbol_map")
    op.drop_table("instrument_groups")
    op.drop_table("instruments")
