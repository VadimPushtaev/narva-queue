"""YOLO-based person detection and annotation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias


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


def load_yolo_model(model_name: str):
    """Load YOLO model lazily to keep imports optional outside detection code."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install dependencies with `poetry install`."
        ) from exc
    return YOLO(model_name)


def scale_polygon(polygon: Polygon, frame_width: int, frame_height: int) -> Polygon:
    """Scale polygon from base resolution to current frame size."""
    x_scale = frame_width / ROI_BASE_WIDTH
    y_scale = frame_height / ROI_BASE_HEIGHT
    return [(int(round(x * x_scale)), int(round(y * y_scale))) for x, y in polygon]


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
    """Run inference and return count, dimensions and filtered person boxes."""
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


def annotate_image_png(
    image_path: str,
    person_boxes: list[PersonBox],
    roi_polygon: Polygon | None = None,
) -> bytes:
    """Render yellow boxes and ROI onto image and return PNG bytes."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is not installed. Reinstall project dependencies with `poetry install`."
        ) from exc

    with Image.open(image_path) as image:
        draw = ImageDraw.Draw(image)
        for x1, y1, x2, y2 in person_boxes:
            draw.rectangle(((x1, y1), (x2, y2)), outline=(255, 255, 0), width=3)
        if roi_polygon:
            closed_roi = [*roi_polygon, roi_polygon[0]]
            draw.line(closed_roi, fill=(255, 255, 0), width=3)

        from io import BytesIO

        buff = BytesIO()
        image.save(buff, format="PNG")
        return buff.getvalue()


def save_annotated_png(
    image_path: str,
    output_path: str,
    person_boxes: list[PersonBox],
    roi_polygon: Polygon | None = None,
) -> str:
    """Save annotated PNG to path and return absolute location."""
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = annotate_image_png(image_path, person_boxes, roi_polygon=roi_polygon)
    target.write_bytes(image_bytes)
    return str(target)

