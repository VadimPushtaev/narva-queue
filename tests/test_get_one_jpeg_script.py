from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import unittest
from unittest.mock import patch

from narva_queue.camera.balticlivecam import CaptureResult
from narva_queue.camera.exceptions import FrameCaptureError
from scripts import get_one_jpeg


class GetOneJpegScriptTests(unittest.TestCase):
    @patch("scripts.get_one_jpeg.capture_frame_to_file")
    def test_main_success(self, mock_capture) -> None:
        mock_capture.return_value = CaptureResult(
            output_path="/tmp/frame.jpg",
            width=1920,
            height=1080,
            stream_url_host="edge01.balticlivecam.com",
            captured_at_utc=datetime.now(timezone.utc),
            source_page_url="https://balticlivecam.com/ru/cameras/estonia/narva/narva/",
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = get_one_jpeg.main(["--output", "/tmp/frame.jpg"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Saved JPEG: /tmp/frame.jpg (1920x1080)", stdout.getvalue())

    @patch("scripts.get_one_jpeg.capture_frame_to_file")
    def test_main_camera_error(self, mock_capture) -> None:
        mock_capture.side_effect = FrameCaptureError("ffmpeg failed")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = get_one_jpeg.main(["--output", "/tmp/frame.jpg"])

        self.assertEqual(exit_code, 2)
        self.assertIn("Camera capture error: ffmpeg failed", stderr.getvalue())

    def test_main_requires_output(self) -> None:
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stderr(stderr):
                get_one_jpeg.main([])

        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--output", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

