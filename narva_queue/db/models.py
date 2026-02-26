"""Database models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, LargeBinary, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Capture(Base):
    """Single capture + inference result snapshot."""

    __tablename__ = "captures"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    camera_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    people_count: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_bytes: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    image_mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True, default="image/jpeg")
    annotated_image_bytes: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    annotated_image_mime_type: Mapped[str | None] = mapped_column(
        String(100), nullable=True, default="image/png"
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_captures_status_captured_at", "status", "captured_at"),
    )

