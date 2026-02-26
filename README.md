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
- raw image bytes (JPEG)
- annotated image bytes (PNG with yellow boxes and ROI)
- status (`ok` / `error`) and error text

Retention policy:

- rows are kept forever
- image bytes are nulled after 30 days (counts remain)

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

Web UI:

- http://localhost:8444/

## Pages

- `/` dashboard with latest capture/status
- `/plots` charts for:
  - last hour
  - last day
  - last month
  - all time
- `/captures` paginated table of captures
- `/captures/{id}` details with original and annotated image

## API

- `GET /healthz`
- `GET /api/metrics/series?range=hour|day|month|all`
- `GET /api/captures?page=1&page_size=50`
- `GET /api/captures/{id}`
- `GET /captures/{id}/image`
- `GET /captures/{id}/annotated`

## Development (Poetry)

Install deps:

```bash
poetry install
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
