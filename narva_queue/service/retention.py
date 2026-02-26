"""Image retention rules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, update
from sqlalchemy.orm import Session

from narva_queue.db.models import Capture


def prune_old_images(session: Session, image_ttl_days: int) -> int:
    """Drop old image binaries while keeping count history."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=image_ttl_days)
    stmt = (
        update(Capture)
        .where(
            Capture.captured_at < cutoff,
            or_(Capture.image_bytes.is_not(None), Capture.annotated_image_bytes.is_not(None)),
        )
        .values(
            image_bytes=None,
            image_mime_type=None,
            annotated_image_bytes=None,
            annotated_image_mime_type=None,
        )
    )
    result = session.execute(stmt)
    return int(result.rowcount or 0)

