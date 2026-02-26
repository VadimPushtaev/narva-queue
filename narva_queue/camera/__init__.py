"""Camera integrations for Narva queue tracking."""

from .balticlivecam import (
    DEFAULT_CAMERA_ID,
    DEFAULT_PAGE_URL,
    CaptureResult,
    capture_frame_bytes,
    capture_frame_to_file,
    get_stream_url,
)
from .exceptions import (
    CameraModuleError,
    DependencyMissingError,
    FrameCaptureError,
    StreamDiscoveryError,
)

__all__ = [
    "CameraModuleError",
    "CaptureResult",
    "DEFAULT_CAMERA_ID",
    "DEFAULT_PAGE_URL",
    "DependencyMissingError",
    "FrameCaptureError",
    "StreamDiscoveryError",
    "capture_frame_bytes",
    "capture_frame_to_file",
    "get_stream_url",
]

