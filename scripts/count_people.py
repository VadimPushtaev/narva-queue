"""Capture one temp frame and count people on it with YOLOv8."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import sys
from typing import TypeAlias

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from narva_queue.camera import capture_frame_to_file
from narva_queue.camera.exceptions import CameraModuleError

PersonBox: TypeAlias = tuple[int, int, int, int]
Polygon: TypeAlias = list[tuple[int, int]]
ROI_BASE_WIDTH = 1920
ROI_BASE_HEIGHT = 1080
ROI_POLYGON_BASE: Polygon = [
    (303, 465),
    (354, 465),
    (890, 527),
    (1279, 588),
    (1510, 641),
    (1683, 702),
    (1820, 783),
    (1888, 841),
    (1739, 900),
    (1195, 817),
    (876, 705),
    (293, 500),
]


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


def load_yolo_model(model_name: str):
    """Load YOLO model lazily to keep imports optional outside this script."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install dependencies with `poetry install`."
        ) from exc
    return YOLO(model_name)


def scale_polygon(polygon: Polygon, frame_width: int, frame_height: int) -> Polygon:
    """Scale polygon from 1920x1080 base resolution to current frame size."""
    x_scale = frame_width / ROI_BASE_WIDTH
    y_scale = frame_height / ROI_BASE_HEIGHT
    return [
        (int(round(x * x_scale)), int(round(y * y_scale)))
        for x, y in polygon
    ]


def point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """Return True when point is inside polygon using ray casting."""
    inside = False
    points_count = len(polygon)
    if points_count < 3:
        return False

    j = points_count - 1
    for i in range(points_count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def bottom_center(box: PersonBox) -> tuple[float, float]:
    """Return bottom-center point of bounding box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, float(y2)


def get_scaled_roi_polygon(image_width: int | None, image_height: int | None) -> Polygon | None:
    """Get ROI polygon scaled to current frame dimensions."""
    if not image_width or not image_height:
        return None
    return scale_polygon(ROI_POLYGON_BASE, image_width, image_height)


def count_people_in_image(
    model,
    image_path: str,
    confidence: float,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[int, int | None, int | None, list[PersonBox]]:
    """Run inference and return number of detected people plus image dimensions."""
    predict_kwargs: dict[str, object] = {
        "source": image_path,
        "conf": confidence,
        "verbose": False,
    }
    if image_width and image_height:
        predict_kwargs["imgsz"] = (image_height, image_width)
        predict_kwargs["rect"] = True

    results = model.predict(**predict_kwargs)
    if not results:
        return 0, None, None, []

    result = results[0]
    image_height = None
    image_width = None
    if hasattr(result, "orig_shape") and result.orig_shape:
        image_height = int(result.orig_shape[0])
        image_width = int(result.orig_shape[1])

    boxes = getattr(result, "boxes", None)
    if (
        boxes is None
        or getattr(boxes, "cls", None) is None
        or getattr(boxes, "xyxy", None) is None
    ):
        return 0, image_width, image_height, []

    cls_values = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else list(boxes.cls)
    xyxy_values = boxes.xyxy.tolist() if hasattr(boxes.xyxy, "tolist") else list(boxes.xyxy)

    person_boxes_all: list[PersonBox] = []
    for cls_id, box in zip(cls_values, xyxy_values):
        if int(cls_id) != 0:
            continue
        x1, y1, x2, y2 = (int(round(coord)) for coord in box[:4])
        person_boxes_all.append((x1, y1, x2, y2))

    roi_polygon = get_scaled_roi_polygon(image_width, image_height)
    if roi_polygon is None:
        person_boxes_filtered = person_boxes_all
    else:
        person_boxes_filtered = [
            box for box in person_boxes_all if point_in_polygon(*bottom_center(box), roi_polygon)
        ]

    return len(person_boxes_filtered), image_width, image_height, person_boxes_filtered


def save_annotated_png(
    image_path: str,
    output_path: str,
    person_boxes: list[PersonBox],
    roi_polygon: Polygon | None = None,
) -> str:
    """Save image with yellow rectangles for detected people and return output path."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is not installed. Reinstall project dependencies with `poetry install`."
        ) from exc

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        draw = ImageDraw.Draw(image)
        for x1, y1, x2, y2 in person_boxes:
            draw.rectangle(((x1, y1), (x2, y2)), outline=(255, 255, 0), width=3)
        if roi_polygon:
            closed_roi = [*roi_polygon, roi_polygon[0]]
            draw.line(closed_roi, fill=(255, 255, 0), width=3)
        image.save(target, format="PNG")

    return str(target)


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
