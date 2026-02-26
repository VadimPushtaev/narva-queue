"""Capture one JPEG frame from the Narva BalticLiveCam stream."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from narva_queue.camera import capture_frame_to_file
from narva_queue.camera.exceptions import CameraModuleError


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Capture one JPEG frame from BalticLiveCam Narva stream."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JPEG file.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Capture timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=461,
        help="BalticLiveCam camera id (default: 461).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the script and return process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    output_path = str(Path(args.output).expanduser().resolve())
    try:
        result = capture_frame_to_file(
            output_path=output_path,
            camera_id=args.camera_id,
            timeout_sec=args.timeout,
        )
    except CameraModuleError as exc:
        print(f"Camera capture error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    dimensions = f"{result.width}x{result.height}" if result.width and result.height else "unknown"
    print(f"Saved JPEG: {result.output_path} ({dimensions})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
