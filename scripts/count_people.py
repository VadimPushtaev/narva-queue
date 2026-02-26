"""Capture one temp frame and count people on it with YOLOv8."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from narva_queue.camera import capture_frame_to_file
from narva_queue.camera.exceptions import CameraModuleError
from narva_queue.detection import (
    count_people_in_image,
    get_scaled_roi_polygon,
    load_yolo_model,
    save_annotated_png,
)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Capture one frame from Narva stream and count people with YOLOv8."
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=461,
        help="BalticLiveCam camera id (default: 461).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Capture timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model path or model name (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--annotated-png",
        default=None,
        help="Optional path to save PNG with detected people marked in yellow.",
    )
    return parser


def _build_output(
    *,
    camera_id: int,
    model_name: str,
    confidence: float,
    people_count: int | None,
    image_width: int | None,
    image_height: int | None,
    annotated_png_path: str | None,
    status: str,
    error: str | None,
) -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "camera_id": camera_id,
        "model": model_name,
        "confidence_threshold": confidence,
        "people_count": people_count,
        "image_width": image_width,
        "image_height": image_height,
        "annotated_png_path": annotated_png_path,
        "status": status,
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the people counting script and return process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    temp_file = NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        capture_result = capture_frame_to_file(
            output_path=str(temp_path),
            camera_id=args.camera_id,
            timeout_sec=args.timeout,
        )
        model = load_yolo_model(args.model)
        people_count, image_width, image_height, person_boxes = count_people_in_image(
            model=model,
            image_path=str(temp_path),
            confidence=args.conf,
            image_width=capture_result.width,
            image_height=capture_result.height,
        )
        if image_width is None:
            image_width = capture_result.width
        if image_height is None:
            image_height = capture_result.height

        annotated_png_path = None
        roi_polygon = get_scaled_roi_polygon(image_width, image_height)
        if args.annotated_png:
            annotated_png_path = save_annotated_png(
                image_path=str(temp_path),
                output_path=args.annotated_png,
                person_boxes=person_boxes,
                roi_polygon=roi_polygon,
            )

        payload = _build_output(
            camera_id=args.camera_id,
            model_name=args.model,
            confidence=args.conf,
            people_count=people_count,
            image_width=image_width,
            image_height=image_height,
            annotated_png_path=annotated_png_path,
            status="ok",
            error=None,
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except CameraModuleError as exc:
        payload = _build_output(
            camera_id=args.camera_id,
            model_name=args.model,
            confidence=args.conf,
            people_count=None,
            image_width=None,
            image_height=None,
            annotated_png_path=None,
            status="error",
            error=str(exc),
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 2
    except Exception as exc:
        payload = _build_output(
            camera_id=args.camera_id,
            model_name=args.model,
            confidence=args.conf,
            people_count=None,
            image_width=None,
            image_height=None,
            annotated_png_path=None,
            status="error",
            error=str(exc),
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 1
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

