"""Make regime_scores.probability and alert nullable.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-11-16

probability and alert must be NULL for insufficient_data rows (weeks where
core features VIX / S&P 500 / US 10Y are missing).
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("regime_scores", "probability", existing_type=sa.Float(), nullable=True)
    op.alter_column("regime_scores", "alert", existing_type=sa.Boolean(), nullable=True)


def downgrade() -> None:
    op.alter_column("regime_scores", "probability", existing_type=sa.Float(), nullable=False)
    op.alter_column("regime_scores", "alert", existing_type=sa.Boolean(), nullable=False)
