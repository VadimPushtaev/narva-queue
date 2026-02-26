"""Periodic ingestion worker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import time

from narva_queue.config import load_settings
from narva_queue.db.session import get_session
from narva_queue.detection import load_yolo_model
from narva_queue.service import ingest_capture, prune_old_images


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("narva_queue.worker")


def run() -> None:
    """Run periodic capture/inference loop forever."""
    settings = load_settings()
    LOGGER.info("Starting worker with interval=%ss", settings.capture_interval_seconds)
    model = load_yolo_model(settings.yolo_model)
    last_retention_at = datetime.now(timezone.utc) - timedelta(days=2)

    while True:
        loop_started = time.monotonic()
        try:
            with get_session() as session:
                capture = ingest_capture(session=session, model=model, settings=settings)
                if capture.status == "ok":
                    LOGGER.info(
                        "Capture id=%s count=%s captured_at=%s",
                        capture.id,
                        capture.people_count,
                        capture.captured_at,
                    )
                else:
                    LOGGER.warning("Capture id=%s failed: %s", capture.id, capture.error)

                now = datetime.now(timezone.utc)
                if now - last_retention_at >= timedelta(hours=24):
                    pruned = prune_old_images(session=session, image_ttl_days=settings.image_ttl_days)
                    last_retention_at = now
                    LOGGER.info("Retention cleanup done; pruned rows=%s", pruned)
        except Exception as exc:
            LOGGER.exception("Worker iteration failed: %s", exc)

        elapsed = time.monotonic() - loop_started
        sleep_seconds = max(0.0, settings.capture_interval_seconds - elapsed)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    run()

