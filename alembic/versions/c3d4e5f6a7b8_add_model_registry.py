"""Add model_versions and regime_scores tables.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-10-29

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("version", sa.String(128), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label_set", sa.String(64), nullable=False),
        sa.Column("spec_hash", sa.String(64), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("heads", sa.Text(), nullable=False),
        sa.Column("metrics", sa.Text(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("platt_a", sa.Float(), nullable=False),
        sa.Column("platt_b", sa.Float(), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    # At most one active model version at a time (partial unique index).
    op.create_index(
        "ix_model_versions_active_unique",
        "model_versions",
        ["active"],
        unique=True,
        postgresql_where=sa.text("active = true"),
    )

    op.create_table(
        "regime_scores",
        sa.Column("week", sa.Date(), nullable=False),
        sa.Column(
            "model_version",
            sa.String(128),
            sa.ForeignKey("model_versions.version"),
            nullable=False,
        ),
        sa.Column("head", sa.String(16), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("alert", sa.Boolean(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("top_features", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("week", "model_version", "head"),
    )
    op.create_index(
        "ix_regime_scores_model_version_head",
        "regime_scores",
        ["model_version", "head"],
    )


def downgrade() -> None:
    op.drop_index("ix_regime_scores_model_version_head", table_name="regime_scores")
    op.drop_table("regime_scores")
    op.drop_index("ix_model_versions_active_unique", table_name="model_versions")
    op.drop_table("model_versions")
