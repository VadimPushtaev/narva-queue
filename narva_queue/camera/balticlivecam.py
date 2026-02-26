"""Helpers to fetch one frame from BalticLiveCam Narva stream."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import re
import shutil
import subprocess

from .exceptions import DependencyMissingError, FrameCaptureError, StreamDiscoveryError


DEFAULT_PAGE_URL: Final[str] = "https://balticlivecam.com/ru/cameras/estonia/narva/narva/"
AUTH_ENDPOINT: Final[str] = "https://balticlivecam.com/wp-admin/admin-ajax.php"
DEFAULT_CAMERA_ID: Final[int] = 461
DEFAULT_STREAM_FETCH_ATTEMPTS: Final[int] = 2
DEFAULT_CAPTURE_ATTEMPTS: Final[int] = 2
DEFAULT_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)
STREAM_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"https://[^\s'\"<>]+\.m3u8\?token=[^\s'\"<>]+"
)
TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"token=[^&\s'\"<>]+")


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Metadata for a completed frame capture."""

    output_path: str
    width: int | None
    height: int | None
    stream_url_host: str
    captured_at_utc: datetime
    source_page_url: str


def get_stream_url(
    camera_id: int = DEFAULT_CAMERA_ID,
    page_url: str = DEFAULT_PAGE_URL,
    timeout_sec: float = 15.0,
) -> str:
    """Return a tokenized HLS URL for the requested BalticLiveCam camera."""
    last_error: Exception | None = None
    for _ in range(DEFAULT_STREAM_FETCH_ATTEMPTS):
        try:
            response_text = _request_auth_token_payload(
                camera_id=camera_id,
                page_url=page_url,
                timeout_sec=timeout_sec,
            )
            return _extract_stream_url(response_text)
        except Exception as exc:  # pragma: no cover - exercised by tests via typed asserts
            last_error = exc

    message = "Unable to discover stream URL from auth_token response."
    if last_error is not None:
        message += f" Last error: {_redact_tokens(str(last_error))}"
    raise StreamDiscoveryError(message) from last_error


def capture_frame_to_file(
    output_path: str,
    camera_id: int = DEFAULT_CAMERA_ID,
    ffmpeg_bin: str = "ffmpeg",
    timeout_sec: float = 30.0,
    jpeg_quality: int = 2,
) -> CaptureResult:
    """Capture one frame from the stream and save it as JPEG."""
    if shutil.which(ffmpeg_bin) is None:
        raise DependencyMissingError(f"Required binary not found in PATH: {ffmpeg_bin}")

    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for _ in range(DEFAULT_CAPTURE_ATTEMPTS):
        stream_url: str | None = None
        try:
            stream_url = get_stream_url(camera_id=camera_id, timeout_sec=timeout_sec)
            _run_ffmpeg_capture(
                stream_url=stream_url,
                output_path=target,
                ffmpeg_bin=ffmpeg_bin,
                timeout_sec=timeout_sec,
                jpeg_quality=jpeg_quality,
            )
            width, height = _probe_dimensions(target)
            parsed = urlparse(stream_url)
            return CaptureResult(
                output_path=str(target),
                width=width,
                height=height,
                stream_url_host=parsed.netloc,
                captured_at_utc=datetime.now(timezone.utc),
                source_page_url=DEFAULT_PAGE_URL,
            )
        except (StreamDiscoveryError, FrameCaptureError) as exc:
            last_error = exc
            if target.exists():
                target.unlink(missing_ok=True)

    message = "Failed to capture frame from live stream."
    if last_error is not None:
        message += f" Last error: {_redact_tokens(str(last_error))}"
    raise FrameCaptureError(message) from last_error


def capture_frame_bytes(
    camera_id: int = DEFAULT_CAMERA_ID,
    ffmpeg_bin: str = "ffmpeg",
    timeout_sec: float = 30.0,
    jpeg_quality: int = 2,
) -> bytes:
    """Capture one frame and return JPEG bytes."""
    temp_file = NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        capture_frame_to_file(
            output_path=str(temp_path),
            camera_id=camera_id,
            ffmpeg_bin=ffmpeg_bin,
            timeout_sec=timeout_sec,
            jpeg_quality=jpeg_quality,
        )
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)


def _request_auth_token_payload(camera_id: int, page_url: str, timeout_sec: float) -> str:
    referer = page_url
    origin = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"
    payload = urlencode(
        {
            "action": "auth_token",
            "id": str(camera_id),
            "embed": "0",
            "main_referer": "",
        }
    ).encode("utf-8")

    request = Request(
        AUTH_ENDPOINT,
        data=payload,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Origin": origin,
            "Referer": referer,
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_sec) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - covered via get_stream_url failure tests
        raise StreamDiscoveryError(f"auth_token request failed: {_redact_tokens(str(exc))}") from exc


def _extract_stream_url(response_text: str) -> str:
    match = STREAM_URL_PATTERN.search(response_text)
    if not match:
        raise StreamDiscoveryError("No tokenized m3u8 URL found in auth_token response.")
    return match.group(0)


def _run_ffmpeg_capture(
    stream_url: str,
    output_path: Path,
    ffmpeg_bin: str,
    timeout_sec: float,
    jpeg_quality: int,
) -> None:
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        stream_url,
        "-frames:v",
        "1",
        "-q:v",
        str(jpeg_quality),
        str(output_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError as exc:
        raise DependencyMissingError(f"Required binary not found in PATH: {ffmpeg_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise FrameCaptureError("ffmpeg timed out while capturing frame.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = _redact_tokens(exc.stderr or "")
        raise FrameCaptureError(f"ffmpeg failed: {stderr.strip()}") from exc

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise FrameCaptureError("ffmpeg completed but output JPEG was not created.")


def _probe_dimensions(image_path: Path) -> tuple[int | None, int | None]:
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None:
        return None, None

    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(image_path),
    ]

    try:
        proc = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except Exception:
        return None, None

    output = proc.stdout.strip()
    if "x" not in output:
        return None, None

    width_str, height_str = output.split("x", 1)
    if not (width_str.isdigit() and height_str.isdigit()):
        return None, None
    return int(width_str), int(height_str)


def _redact_tokens(value: str) -> str:
    return TOKEN_PATTERN.sub("token=<redacted>", value)

