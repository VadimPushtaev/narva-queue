from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from narva_queue.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_default_capture_interval_is_three_minutes(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings()
        self.assertEqual(settings.capture_interval_seconds, 180)

    def test_default_yolo_confidence_is_0_15(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings()
        self.assertEqual(settings.yolo_conf, 0.15)


if __name__ == "__main__":
    unittest.main()
