"""FastAPI web app for metrics and capture browsing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from narva_queue.config import load_settings
from narva_queue.db.models import Capture
from narva_queue.db.session import get_session


APP_ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_ROOT / "templates"))
settings = load_settings()
app = FastAPI(title="Narva Queue Service")
app.mount("/static", StaticFiles(directory=str(APP_ROOT / "static")), name="static")
MAX_ALL_POINTS = 1000


def get_db() -> Session:
    """FastAPI dependency for DB session."""
    with get_session() as session:
        yield session


def _range_to_bucket_and_since(range_name: str) -> tuple[str, datetime | None]:
    now = datetime.now(timezone.utc)
    if range_name == "hour":
        return "minute", now - timedelta(hours=1)
    if range_name == "day":
        return "raw", now - timedelta(days=1)
    if range_name == "month":
        return "hour", now - timedelta(days=30)
    if range_name == "all":
        return "raw", None
    raise HTTPException(status_code=400, detail="range must be one of: hour, day, month, all")


def _downsample_points(points: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    """Uniformly sample points while preserving first/last and chronology."""
    size = len(points)
    if size <= max_points or max_points <= 0:
        return points
    if max_points == 1:
        return [points[-1]]

    raw_indices = [
        round(i * (size - 1) / (max_points - 1))
        for i in range(max_points)
    ]
    dedup_indices: list[int] = []
    seen: set[int] = set()
    for idx in raw_indices:
        if idx not in seen:
            dedup_indices.append(idx)
            seen.add(idx)
    return [points[idx] for idx in dedup_indices]


def _series_data(db: Session, range_name: str) -> dict[str, Any]:
    bucket_name, since = _range_to_bucket_and_since(range_name)
    base_where = [Capture.status == "ok", Capture.people_count.is_not(None)]
    if since is not None:
        base_where.append(Capture.captured_at >= since)

    if bucket_name == "raw":
        stmt = (
            select(Capture.captured_at, Capture.people_count)
            .where(*base_where)
            .order_by(Capture.captured_at)
        )
        rows = db.execute(stmt).all()
        points = [
            {"timestamp": row.captured_at.isoformat(), "value": float(row.people_count)}
            for row in rows
        ]
        if range_name == "all":
            points = _downsample_points(points, MAX_ALL_POINTS)
    else:
        bucket = func.date_trunc(bucket_name, Capture.captured_at).label("bucket")
        stmt = (
            select(bucket, func.avg(Capture.people_count).label("avg_count"))
            .where(*base_where)
            .group_by(bucket)
            .order_by(bucket)
        )
        rows = db.execute(stmt).all()
        points = [
            {"timestamp": row.bucket.isoformat(), "value": float(row.avg_count)}
            for row in rows
        ]
    return {"range": range_name, "points": points}


def _latest_capture(db: Session) -> Capture | None:
    return db.execute(
        select(Capture).order_by(desc(Capture.captured_at)).limit(1)
    ).scalar_one_or_none()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    latest = _latest_capture(db)
    total_captures = db.execute(select(func.count(Capture.id))).scalar_one()
    ok_captures = db.execute(
        select(func.count(Capture.id)).where(Capture.status == "ok")
    ).scalar_one()
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "latest": latest,
            "total_captures": total_captures,
            "ok_captures": ok_captures,
        },
    )


@app.get("/plots", response_class=HTMLResponse)
def plots_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("plots.html", {"request": request})


@app.get("/captures", response_class=HTMLResponse)
def captures_page(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=settings.default_page_size, ge=1, le=500),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    offset = (page - 1) * page_size
    total = db.execute(select(func.count(Capture.id))).scalar_one()
    rows = db.execute(
        select(Capture)
        .order_by(desc(Capture.captured_at))
        .offset(offset)
        .limit(page_size)
    ).scalars().all()
    return templates.TemplateResponse(
        "captures.html",
        {
            "request": request,
            "captures": rows,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": max(1, ((total + page_size - 1) // page_size)),
        },
    )


@app.get("/captures/{capture_id}", response_class=HTMLResponse)
def capture_detail(
    request: Request,
    capture_id: int,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    capture = db.get(Capture, capture_id)
    if capture is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    return templates.TemplateResponse(
        "capture_detail.html",
        {"request": request, "capture": capture},
    )


@app.get("/api/metrics/series")
def metrics_series(
    range: str = Query(default="hour"),  # noqa: A002
    db: Session = Depends(get_db),
) -> JSONResponse:
    return JSONResponse(_series_data(db, range))


@app.get("/api/captures")
def captures_api(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=settings.default_page_size, ge=1, le=500),
    db: Session = Depends(get_db),
) -> JSONResponse:
    offset = (page - 1) * page_size
    total = db.execute(select(func.count(Capture.id))).scalar_one()
    rows = db.execute(
        select(Capture).order_by(desc(Capture.captured_at)).offset(offset).limit(page_size)
    ).scalars().all()
    payload = {
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": [
            {
                "id": row.id,
                "captured_at": row.captured_at.isoformat(),
                "camera_id": row.camera_id,
                "people_count": row.people_count,
                "status": row.status,
                "error": row.error,
                "has_image": row.image_bytes is not None,
                "has_annotated_image": row.annotated_image_bytes is not None,
            }
            for row in rows
        ],
    }
    return JSONResponse(payload)


@app.get("/api/captures/{capture_id}")
def capture_api(capture_id: int, db: Session = Depends(get_db)) -> JSONResponse:
    row = db.get(Capture, capture_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    return JSONResponse(
        {
            "id": row.id,
            "captured_at": row.captured_at.isoformat(),
            "camera_id": row.camera_id,
            "people_count": row.people_count,
            "status": row.status,
            "error": row.error,
            "confidence_threshold": row.confidence_threshold,
            "model_name": row.model_name,
            "image_width": row.image_width,
            "image_height": row.image_height,
            "has_image": row.image_bytes is not None,
            "has_annotated_image": row.annotated_image_bytes is not None,
        }
    )


@app.get("/captures/{capture_id}/image")
def capture_image(capture_id: int, db: Session = Depends(get_db)) -> Response:
    row = db.get(Capture, capture_id)
    if row is None or row.image_bytes is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=row.image_bytes, media_type=row.image_mime_type or "image/jpeg")


@app.get("/captures/{capture_id}/annotated")
def capture_annotated_image(capture_id: int, db: Session = Depends(get_db)) -> Response:
    row = db.get(Capture, capture_id)
    if row is None or row.annotated_image_bytes is None:
        raise HTTPException(status_code=404, detail="Annotated image not found")
    return Response(
        content=row.annotated_image_bytes,
        media_type=row.annotated_image_mime_type or "image/png",
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
