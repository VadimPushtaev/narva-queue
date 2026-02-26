from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from narva_queue.camera.balticlivecam import CaptureResult
from narva_queue.camera.exceptions import FrameCaptureError
from scripts import count_people


class _DummyTempFile:
    def __init__(self, name: str) -> None:
        self.name = name

    def close(self) -> None:
        return None


def _capture_result(path: str) -> CaptureResult:
    return CaptureResult(
        output_path=path,
        width=1920,
        height=1080,
        stream_url_host="edge01.balticlivecam.com",
        captured_at_utc=datetime.now(timezone.utc),
        source_page_url="https://balticlivecam.com/ru/cameras/estonia/narva/narva/",
    )


class _DummyClsValues:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def tolist(self) -> list[int]:
        return self._values


class _DummyXYXYValues:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def tolist(self) -> list[list[float]]:
        return self._values


class _DummyBoxes:
    def __init__(self, values: list[int], coords: list[list[float]]) -> None:
        self.cls = _DummyClsValues(values)
        self.xyxy = _DummyXYXYValues(coords)


class _DummyResult:
    def __init__(self, values: list[int], coords: list[list[float]], shape: tuple[int, int]) -> None:
        self.boxes = _DummyBoxes(values, coords)
        self.orig_shape = shape


class _DummyModel:
    def __init__(self, result: _DummyResult) -> None:
        self._result = result
        self.last_predict_kwargs: dict[str, object] = {}

    def predict(self, **kwargs):
        self.last_predict_kwargs = kwargs
        return [self._result]


class CountPeopleScriptTests(unittest.TestCase):
    def test_count_people_in_image_uses_native_resolution(self) -> None:
        model = _DummyModel(
            _DummyResult(
                values=[0, 1, 0],
                coords=[
                    [20.0, 20.0, 40.0, 40.0],  # person inside ROI
                    [25.0, 30.0, 45.0, 50.0],  # non-person (ignored)
                    [1.0, 1.0, 5.0, 5.0],      # person outside ROI
                ],
                shape=(100, 100),
            )
        )

        with patch.object(count_people, "ROI_BASE_WIDTH", 100), patch.object(
            count_people, "ROI_BASE_HEIGHT", 100
        ), patch.object(
            count_people,
            "ROI_POLYGON_BASE",
            [(10, 10), (90, 10), (90, 90), (10, 90)],
        ):
            people_count, image_width, image_height, person_boxes = count_people.count_people_in_image(
                model=model,
                image_path="/tmp/frame.jpg",
                confidence=0.25,
                image_width=100,
                image_height=100,
            )

        self.assertEqual(people_count, 1)
        self.assertEqual(image_width, 100)
        self.assertEqual(image_height, 100)
        self.assertEqual(model.last_predict_kwargs["imgsz"], (100, 100))
        self.assertEqual(model.last_predict_kwargs["rect"], True)
        self.assertEqual(model.last_predict_kwargs["conf"], 0.25)
        self.assertEqual(person_boxes, [(20, 20, 40, 40)])

    @patch("scripts.count_people.save_annotated_png")
    @patch("scripts.count_people.count_people_in_image")
    @patch("scripts.count_people.load_yolo_model")
    @patch("scripts.count_people.capture_frame_to_file")
    def test_main_success(
        self,
        mock_capture,
        mock_load_model,
        mock_count_people,
        mock_save_annotated_png,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "frame.jpg"
            tmp_path.write_bytes(b"abc")

            mock_capture.return_value = _capture_result(str(tmp_path))
            mock_load_model.return_value = object()
            mock_count_people.return_value = (5, 1920, 1080, [(1, 2, 3, 4)])
            mock_save_annotated_png.return_value = str(Path(tmpdir) / "out.png")

            stdout = io.StringIO()
            with patch(
                "scripts.count_people.NamedTemporaryFile",
                return_value=_DummyTempFile(str(tmp_path)),
            ), redirect_stdout(stdout):
                exit_code = count_people.main([])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["people_count"], 5)
        self.assertEqual(payload["image_width"], 1920)
        self.assertEqual(payload["image_height"], 1080)
        self.assertEqual(payload["model"], "yolov8n.pt")
        self.assertEqual(payload["camera_id"], 461)
        self.assertEqual(payload["confidence_threshold"], 0.25)
        self.assertIsNone(payload["annotated_png_path"])
        mock_save_annotated_png.assert_not_called()
        self.assertFalse(tmp_path.exists(), "Temporary JPEG should be removed")

    @patch("scripts.count_people.save_annotated_png")
    @patch("scripts.count_people.count_people_in_image")
    @patch("scripts.count_people.load_yolo_model")
    @patch("scripts.count_people.capture_frame_to_file")
    def test_main_success_with_annotation_path(
        self,
        mock_capture,
        mock_load_model,
        mock_count_people,
        mock_save_annotated_png,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "frame.jpg"
            out_path = Path(tmpdir) / "annotated.png"
            tmp_path.write_bytes(b"abc")

            mock_capture.return_value = _capture_result(str(tmp_path))
            mock_load_model.return_value = object()
            mock_count_people.return_value = (3, 1920, 1080, [(10, 10, 20, 20)])
            mock_save_annotated_png.return_value = str(out_path)

            stdout = io.StringIO()
            with patch(
                "scripts.count_people.NamedTemporaryFile",
                return_value=_DummyTempFile(str(tmp_path)),
            ), redirect_stdout(stdout):
                exit_code = count_people.main(["--annotated-png", str(out_path)])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["people_count"], 3)
        self.assertEqual(payload["annotated_png_path"], str(out_path))
        mock_save_annotated_png.assert_called_once()
        self.assertIn("roi_polygon", mock_save_annotated_png.call_args.kwargs)
        self.assertIsNotNone(mock_save_annotated_png.call_args.kwargs["roi_polygon"])
        self.assertFalse(tmp_path.exists(), "Temporary JPEG should be removed")

    @patch("scripts.count_people.capture_frame_to_file")
    def test_main_camera_error(self, mock_capture) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "frame.jpg"
            tmp_path.write_bytes(b"abc")

            mock_capture.side_effect = FrameCaptureError("camera failed")

            stdout = io.StringIO()
            with patch(
                "scripts.count_people.NamedTemporaryFile",
                return_value=_DummyTempFile(str(tmp_path)),
            ), redirect_stdout(stdout):
                exit_code = count_people.main([])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "error")
        self.assertIn("camera failed", payload["error"])
        self.assertIsNone(payload["people_count"])
        self.assertIsNone(payload["annotated_png_path"])
        self.assertFalse(tmp_path.exists(), "Temporary JPEG should be removed on error")

    @patch("scripts.count_people.load_yolo_model")
    @patch("scripts.count_people.capture_frame_to_file")
    def test_main_model_error(self, mock_capture, mock_load_model) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "frame.jpg"
            tmp_path.write_bytes(b"abc")

            mock_capture.return_value = _capture_result(str(tmp_path))
            mock_load_model.side_effect = RuntimeError("ultralytics is not installed")

            stdout = io.StringIO()
            with patch(
                "scripts.count_people.NamedTemporaryFile",
                return_value=_DummyTempFile(str(tmp_path)),
            ), redirect_stdout(stdout):
                exit_code = count_people.main([])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "error")
        self.assertIn("ultralytics is not installed", payload["error"])
        self.assertIsNone(payload["annotated_png_path"])
        self.assertFalse(tmp_path.exists(), "Temporary JPEG should be removed on model failure")


if __name__ == "__main__":
    unittest.main()
