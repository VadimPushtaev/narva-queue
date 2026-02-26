"""create captures table

Revision ID: 20260226_0001
Revises:
Create Date: 2026-02-26 19:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260226_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "captures",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=False),
        sa.Column("people_count", sa.Integer(), nullable=True),
        sa.Column("confidence_threshold", sa.Float(), nullable=False),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        sa.Column("image_width", sa.Integer(), nullable=True),
        sa.Column("image_height", sa.Integer(), nullable=True),
        sa.Column("image_bytes", sa.LargeBinary(), nullable=True),
        sa.Column("image_mime_type", sa.String(length=100), nullable=True),
        sa.Column("annotated_image_bytes", sa.LargeBinary(), nullable=True),
        sa.Column("annotated_image_mime_type", sa.String(length=100), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_index("ix_captures_captured_at", "captures", ["captured_at"])
    op.create_index("ix_captures_camera_id", "captures", ["camera_id"])
    op.create_index("ix_captures_people_count", "captures", ["people_count"])
    op.create_index("ix_captures_status", "captures", ["status"])
    op.create_index("ix_captures_status_captured_at", "captures", ["status", "captured_at"])


def downgrade() -> None:
    op.drop_index("ix_captures_status_captured_at", table_name="captures")
    op.drop_index("ix_captures_status", table_name="captures")
    op.drop_index("ix_captures_people_count", table_name="captures")
    op.drop_index("ix_captures_camera_id", table_name="captures")
    op.drop_index("ix_captures_captured_at", table_name="captures")
    op.drop_table("captures")

