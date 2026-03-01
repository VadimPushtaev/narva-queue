from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from narva_queue.db.base import Base
from narva_queue.db.models import Capture
from narva_queue.web.app import AUTO_TARGET_POINTS, _timeline_data


class MetricsTimelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.session: Session = self.session_factory()

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()

    def _insert_capture(self, captured_at: datetime, people_count: int, status: str = "ok") -> None:
        row = Capture(
            captured_at=captured_at,
            camera_id=461,
            people_count=people_count,
            confidence_threshold=0.15,
            model_name="yolov8n.pt",
            image_width=1280,
            image_height=720,
            status=status,
            error=None,
        )
        self.session.add(row)

    def test_empty_dataset_returns_empty_points(self) -> None:
        payload = _timeline_data(self.session, from_ts=None, to_ts=datetime.now(timezone.utc))
        self.assertEqual(payload["mode"], "raw")
        self.assertEqual(payload["points"], [])
        self.assertEqual(payload["bucket_seconds"], None)
        self.assertEqual(payload["target_points"], AUTO_TARGET_POINTS)

    def test_from_must_be_earlier_than_to(self) -> None:
        now = datetime.now(timezone.utc)
        with self.assertRaises(HTTPException) as ctx:
            _timeline_data(self.session, from_ts=now, to_ts=now)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_raw_mode_when_points_are_below_target(self) -> None:
        start = datetime(2026, 2, 1, 8, 0, tzinfo=timezone.utc)
        for i in range(8):
            self._insert_capture(start + timedelta(minutes=i), people_count=i + 1)
        self.session.commit()

        payload = _timeline_data(
            self.session,
            from_ts=start,
            to_ts=start + timedelta(minutes=8),
            tz="Europe/Helsinki",
        )

        self.assertEqual(payload["mode"], "raw")
        self.assertEqual(payload["bucket_seconds"], None)
        self.assertEqual(len(payload["points"]), 8)
        self.assertTrue(all(point["samples"] == 1 for point in payload["points"]))

    def test_aggregated_mode_when_points_exceed_target(self) -> None:
        start = datetime(2026, 2, 1, 8, 0, tzinfo=timezone.utc)
        for i in range(AUTO_TARGET_POINTS + 1):
            self._insert_capture(start + timedelta(minutes=i), people_count=(i % 20) + 1)
        self.session.commit()

        payload = _timeline_data(
            self.session,
            from_ts=start,
            to_ts=start + timedelta(minutes=AUTO_TARGET_POINTS + 1),
            tz="Europe/Helsinki",
        )

        self.assertEqual(payload["mode"], "aggregated")
        self.assertIsNotNone(payload["bucket_seconds"])
        self.assertLessEqual(len(payload["points"]), AUTO_TARGET_POINTS)
        self.assertTrue(any(point["samples"] > 1 for point in payload["points"]))

    def test_all_time_uses_earliest_point_when_from_missing(self) -> None:
        first = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        second = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        self._insert_capture(first, people_count=3)
        self._insert_capture(second, people_count=5)
        self.session.commit()

        payload = _timeline_data(
            self.session,
            from_ts=None,
            to_ts=datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc),
        )
        returned_from = datetime.fromisoformat(payload["from"])
        self.assertEqual(returned_from, first)
        self.assertEqual(len(payload["points"]), 2)


if __name__ == "__main__":
    unittest.main()
