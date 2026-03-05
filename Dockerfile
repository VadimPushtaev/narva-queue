FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.4 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

FROM base AS deps-main
RUN poetry install --no-interaction --no-ansi --only main --no-root

FROM base AS deps-worker
RUN poetry install --no-interaction --no-ansi --with worker --no-root

FROM deps-main AS webui
COPY narva_queue ./narva_queue
CMD ["uvicorn", "narva_queue.web.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM deps-main AS migrate
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY narva_queue ./narva_queue
CMD ["alembic", "upgrade", "head"]

FROM deps-worker AS worker
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY narva_queue ./narva_queue
COPY yolov8n.pt ./yolov8n.pt
CMD ["python", "-m", "narva_queue.worker.main"]
