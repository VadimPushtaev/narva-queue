"""Capture/inference ingestion flow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from sqlalchemy.orm import Session

from narva_queue.config import AppSettings
from narva_queue.db.models import Capture
from narva_queue.camera import capture_frame_to_file
from narva_queue.detection import count_people_in_image, get_scaled_roi_polygon, annotate_image_png


def ingest_capture(session: Session, model, settings: AppSettings) -> Capture:
    """Capture one frame, run inference, and persist result row."""
    temp_jpg = NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_jpg_path = Path(temp_jpg.name)
    temp_jpg.close()

    try:
        capture_result = capture_frame_to_file(
            output_path=str(temp_jpg_path),
            camera_id=settings.camera_id,
            timeout_sec=30.0,
        )

        people_count, image_width, image_height, person_boxes = count_people_in_image(
            model=model,
            image_path=str(temp_jpg_path),
            confidence=settings.yolo_conf,
            image_width=capture_result.width,
            image_height=capture_result.height,
        )
        if image_width is None:
            image_width = capture_result.width
        if image_height is None:
            image_height = capture_result.height

        roi_polygon = get_scaled_roi_polygon(image_width, image_height)
        annotated_png_bytes = annotate_image_png(
            image_path=str(temp_jpg_path),
            person_boxes=person_boxes,
            roi_polygon=roi_polygon,
        )

        row = Capture(
            captured_at=datetime.now(timezone.utc),
            camera_id=settings.camera_id,
            people_count=people_count,
            confidence_threshold=settings.yolo_conf,
            model_name=settings.yolo_model,
            image_width=image_width,
            image_height=image_height,
            image_bytes=temp_jpg_path.read_bytes(),
            image_mime_type="image/jpeg",
            annotated_image_bytes=annotated_png_bytes,
            annotated_image_mime_type="image/png",
            status="ok",
            error=None,
        )
        session.add(row)
        session.flush()
        return row
    except Exception as exc:
        row = Capture(
            captured_at=datetime.now(timezone.utc),
            camera_id=settings.camera_id,
            people_count=None,
            confidence_threshold=settings.yolo_conf,
            model_name=settings.yolo_model,
            image_width=None,
            image_height=None,
            image_bytes=None,
            image_mime_type=None,
            annotated_image_bytes=None,
            annotated_image_mime_type=None,
            status="error",
            error=str(exc),
        )
        session.add(row)
        session.flush()
        return row
    finally:
        temp_jpg_path.unlink(missing_ok=True)

