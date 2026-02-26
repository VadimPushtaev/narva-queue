"""Custom exceptions for camera stream handling."""


class CameraModuleError(Exception):
    """Base exception for camera-related failures."""


class StreamDiscoveryError(CameraModuleError):
    """Raised when the live stream URL cannot be discovered."""


class FrameCaptureError(CameraModuleError):
    """Raised when a frame cannot be captured from the stream."""


class DependencyMissingError(CameraModuleError):
    """Raised when required external tools are missing."""

