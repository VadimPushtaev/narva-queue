# Narva Queue Tracker (Phase 1)

This project starts a queue-monitoring pipeline for the Narva EU-Russia border crossing.

Current milestone: fetch one JPEG frame from the BalticLiveCam Narva stream.

## What is implemented

- BalticLiveCam stream URL discovery via `auth_token` endpoint.
- One-frame JPEG capture with `ffmpeg`.
- Python library API in `narva_queue.camera`.
- CLI script to capture one frame.
- Unit tests plus optional live integration test.

## Prerequisites

- Python 3.12+
- `ffmpeg` available in `PATH`
- Internet access to `balticlivecam.com`

## Quick start

Capture one frame:

```bash
python3 scripts/get_one_jpeg.py --output /tmp/narva.jpg
```

Optional flags:

- `--timeout` (seconds, default `30`)
- `--camera-id` (default `461`)

## Python usage

```python
from narva_queue.camera import capture_frame_to_file

result = capture_frame_to_file("/tmp/narva.jpg")
print(result)
```

## Run tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Run optional live integration test:

```bash
RUN_LIVE_TESTS=1 python3 -m unittest tests/test_balticlivecam_live.py -v
```

## Current scope and next step

- Current scope is frame acquisition only.
- Next step is queue-size estimation from captured frames.

