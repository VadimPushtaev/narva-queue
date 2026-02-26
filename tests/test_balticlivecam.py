from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from narva_queue.camera import balticlivecam as blc
from narva_queue.camera.exceptions import (
    DependencyMissingError,
    FrameCaptureError,
    StreamDiscoveryError,
)


SAMPLE_STREAM_URL = "https://edge01.balticlivecam.com/blc/narva/index.m3u8?token=abc123:1772118515932"
SAMPLE_AUTH_RESPONSE = f"""
<script>
player = videojs('my-video', {{
    sources: [{{
        src: '{SAMPLE_STREAM_URL}',
        type: 'application/x-mpegurl'
    }}]
}});
</script>
"""


class GetStreamUrlTests(unittest.TestCase):
    @patch("narva_queue.camera.balticlivecam._request_auth_token_payload")
    def test_get_stream_url_success(self, mock_payload) -> None:
        mock_payload.return_value = SAMPLE_AUTH_RESPONSE

        stream_url = blc.get_stream_url()

        self.assertEqual(stream_url, SAMPLE_STREAM_URL)

    @patch("narva_queue.camera.balticlivecam._request_auth_token_payload")
    def test_get_stream_url_retries_once(self, mock_payload) -> None:
        mock_payload.side_effect = [
            StreamDiscoveryError("temporary parse issue"),
            SAMPLE_AUTH_RESPONSE,
        ]

        stream_url = blc.get_stream_url()

        self.assertEqual(stream_url, SAMPLE_STREAM_URL)
        self.assertEqual(mock_payload.call_count, 2)

    @patch("narva_queue.camera.balticlivecam._request_auth_token_payload")
    def test_get_stream_url_failure_when_no_m3u8(self, mock_payload) -> None:
        mock_payload.return_value = "<html>no stream here</html>"

        with self.assertRaises(StreamDiscoveryError):
            blc.get_stream_url()


class CaptureFrameTests(unittest.TestCase):
    @patch("narva_queue.camera.balticlivecam.shutil.which")
    def test_capture_frame_to_file_requires_ffmpeg(self, mock_which) -> None:
        mock_which.return_value = None

        with self.assertRaises(DependencyMissingError):
            blc.capture_frame_to_file("/tmp/narva-test.jpg")

    @patch("narva_queue.camera.balticlivecam.subprocess.run")
    def test_ffmpeg_error_message_redacts_token(self, mock_run) -> None:
        stderr_text = (
            "Failed to open "
            "https://edge01.balticlivecam.com/blc/narva/index.m3u8?token=supersecret"
        )
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr=stderr_text,
        )

        with self.assertRaises(FrameCaptureError) as ctx:
            blc._run_ffmpeg_capture(  # pylint: disable=protected-access
                stream_url=SAMPLE_STREAM_URL,
                output_path=Path("/tmp/narva-redaction-test.jpg"),
                ffmpeg_bin="ffmpeg",
                timeout_sec=2,
                jpeg_quality=2,
            )

        msg = str(ctx.exception)
        self.assertIn("token=<redacted>", msg)
        self.assertNotIn("token=supersecret", msg)

    @patch("narva_queue.camera.balticlivecam._probe_dimensions")
    @patch("narva_queue.camera.balticlivecam._run_ffmpeg_capture")
    @patch("narva_queue.camera.balticlivecam.get_stream_url")
    @patch("narva_queue.camera.balticlivecam.shutil.which")
    def test_capture_frame_to_file_success(
        self,
        mock_which,
        mock_get_stream_url,
        mock_run_capture,
        mock_probe,
    ) -> None:
        def which_side_effect(name: str) -> str | None:
            if name == "ffmpeg":
                return "/usr/bin/ffmpeg"
            return "/usr/bin/ffprobe"

        mock_which.side_effect = which_side_effect
        mock_get_stream_url.return_value = SAMPLE_STREAM_URL
        mock_probe.return_value = (1920, 1080)
        mock_run_capture.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "frame.jpg"
            result = blc.capture_frame_to_file(str(output_path))

        self.assertEqual(result.output_path, str(output_path))
        self.assertEqual(result.width, 1920)
        self.assertEqual(result.height, 1080)
        self.assertEqual(result.stream_url_host, "edge01.balticlivecam.com")
        self.assertEqual(result.source_page_url, blc.DEFAULT_PAGE_URL)


if __name__ == "__main__":
    unittest.main()

