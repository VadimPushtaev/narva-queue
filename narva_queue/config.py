"""Runtime configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppSettings:
    """Environment-backed settings for worker and web app."""

    database_url: str
    capture_interval_seconds: int
    image_ttl_days: int
    camera_id: int
    yolo_model: str
    yolo_conf: float
    default_page_size: int


DEFAULT_DATABASE_URL = "postgresql+psycopg://narva:narva@pg:5432/narva_queue"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for {name}: {value}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float for {name}: {value}") from exc


def load_settings() -> AppSettings:
    """Load all app settings from the environment."""
    return AppSettings(
        database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
        capture_interval_seconds=_env_int("CAPTURE_INTERVAL_SECONDS", 60),
        image_ttl_days=_env_int("IMAGE_TTL_DAYS", 30),
        camera_id=_env_int("CAMERA_ID", 461),
        yolo_model=os.getenv("YOLO_MODEL", "yolov8n.pt"),
        yolo_conf=_env_float("YOLO_CONF", 0.25),
        default_page_size=_env_int("DEFAULT_PAGE_SIZE", 50),
    )

