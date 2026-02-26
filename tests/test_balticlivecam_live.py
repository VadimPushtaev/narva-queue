from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from narva_queue.camera import capture_frame_to_file


@unittest.skipUnless(
    os.getenv("RUN_LIVE_TESTS") == "1",
    "Set RUN_LIVE_TESTS=1 to run live integration tests.",
)
class BalticLiveCamLiveTests(unittest.TestCase):
    def test_capture_one_live_frame(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is required for live integration test")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "live-frame.jpg"
            result = capture_frame_to_file(str(output_path), timeout_sec=45.0)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 1024)
            with output_path.open("rb") as f:
                self.assertEqual(f.read(2), b"\xff\xd8")
            self.assertEqual(result.output_path, str(output_path))


if __name__ == "__main__":
    unittest.main()

