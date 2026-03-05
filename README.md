# Narva Queue Service

Service for periodic people counting on Narva live camera snapshots.

## Stack

- PostgreSQL for storage
- Alembic for schema migrations
- Worker service for periodic capture + YOLO inference
- FastAPI + Jinja + HTMX + Chart.js for Web UI
- Docker Compose for local orchestration

## Data model

Each capture row stores:

- timestamp and camera id
- people count + model/confidence metadata
- downscaled original image bytes (JPEG)
- downscaled annotated image bytes (JPEG with yellow boxes and ROI)
- status (`ok` / `error`) and error text

Retention policy:

- rows are kept forever
- image bytes are nulled after 30 days (counts remain)

Default capture cadence is every 3 minutes (`CAPTURE_INTERVAL_SECONDS=180`).

## Quick start

1. Create env file:

```bash
cp .env.example .env
```

2. Build and start:

```bash
docker compose up -d --build
```

On startup:

- `pg` starts
- `migrate` runs `alembic upgrade head`
- after successful migration, `worker` and `webui` start

`pg`, `worker`, and `webui` use `restart: always`, so they auto-start after machine/docker restarts.

Compose builds three service-specific images from one multi-stage `Dockerfile`:

- `narva-queue-migrate` (lean migration runtime)
- `narva-queue-webui` (lean web runtime without YOLO deps/model)
- `narva-queue-worker` (includes YOLO deps and baked `yolov8n.pt`)

Web UI:

- http://localhost:8444/

Key environment knobs:

- `CAPTURE_INTERVAL_SECONDS` (default `180`)
- `YOLO_CONF` (default `0.15`)
- `STORAGE_MAX_WIDTH` / `STORAGE_MAX_HEIGHT` (default `960x540`)
- `STORAGE_JPEG_QUALITY` (default `70`)

## Pages

- `/` dashboard with latest capture/status
- `/plots` interactive timeline explorer with two independent stacked plots
- `/captures` paginated table of captures
- `/captures/{id}` details with original and annotated image

## API

- `GET /healthz`
- `GET /api/metrics/timeline?from=<ISO8601>&to=<ISO8601>&tz=Europe/Helsinki`
- `GET /api/captures?page=1&page_size=50`
- `GET /api/captures/{id}`
- `GET /captures/{id}/image`
- `GET /captures/{id}/annotated`

## Development (Poetry)

Install deps:

```bash
poetry install
```

Install worker deps (YOLO / local worker run):

```bash
poetry install --with worker
```

Run tests:

```bash
poetry run python -m unittest discover -s tests -p 'test_*.py' -v
```

Run worker locally:

```bash
poetry run python -m narva_queue.worker.main
```

Run web app locally:

```bash
poetry run uvicorn narva_queue.web.app:app --reload
```

Run migrations locally:

```bash
poetry run alembic upgrade head
```
