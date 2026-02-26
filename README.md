# Narva Queue Tracker

This project starts a queue-monitoring pipeline for the Narva EU-Russia border crossing.

Current milestone: fetch one JPEG frame and estimate how many people are visible on that frame with YOLOv8.

## What is implemented

- BalticLiveCam stream URL discovery via `auth_token` endpoint.
- One-frame JPEG capture with `ffmpeg`.
- People counting on a single frame with YOLOv8 (`yolov8n` by default).
- Python library API in `narva_queue.camera`.
- CLI script to capture one frame: `scripts/get_one_jpeg.py`.
- CLI script to capture + count people: `scripts/count_people.py`.
- Unit tests plus optional live integration test.
- Poetry-based dependency management.

## Prerequisites

- Python 3.12+
- `ffmpeg` available in `PATH`
- Internet access to `balticlivecam.com`
- Poetry (`pip install poetry`)

## Setup (Poetry)

```bash
poetry install
```

## Quick start

Capture one frame JPEG:

```bash
poetry run python scripts/get_one_jpeg.py --output /tmp/narva.jpg
```

Optional flags:

- `--timeout` (seconds, default `30`)
- `--camera-id` (default `461`)

Capture one frame into a temp file and count people with YOLOv8:

```bash
poetry run python scripts/count_people.py
```

The script always deletes the temporary JPEG after inference.
People are counted only inside the hardcoded queue ROI polygon.

Optional flags:

- `--model` (default `yolov8n.pt`)
- `--conf` (default `0.25`)
- `--timeout` (seconds, default `30`)
- `--camera-id` (default `461`)
- `--annotated-png /path/to/file.png` (optional yellow person boxes + ROI outline)

Example JSON output:

```json
{
  "timestamp_utc": "2026-02-26T17:00:00.000000+00:00",
  "camera_id": 461,
  "model": "yolov8n.pt",
  "confidence_threshold": 0.25,
  "people_count": 12,
  "image_width": 1920,
  "image_height": 1080,
  "annotated_png_path": "/tmp/narva-annotated.png",
  "status": "ok",
  "error": null
}
```

## Python usage

```python
from narva_queue.camera import capture_frame_to_file

result = capture_frame_to_file("/tmp/narva.jpg")
print(result)
```

## Run tests

```bash
poetry run python -m unittest discover -s tests -p 'test_*.py' -v
```

Run optional live integration test:

```bash
RUN_LIVE_TESTS=1 poetry run python -m unittest tests/test_balticlivecam_live.py -v
```

## Current scope and next step

- Current scope is single-frame detection only.
- Next step is more robust queue estimation (multi-frame tracking, zones, and trend metrics).
