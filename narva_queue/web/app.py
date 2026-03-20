"""FastAPI web app for metrics and capture browsing."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
AUTO_TARGET_POINTS = 400
AUTO_BUCKET_CANDIDATES_SECONDS = (60, 300, 900, 1800, 3600, 10800, 21600, 43200, 86400, 604800)


def get_db() -> Session:
    """FastAPI dependency for DB session."""
    with get_session() as session:
        yield session


def _coerce_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise HTTPException(status_code=400, detail="datetime query params must include timezone")
    return dt.astimezone(timezone.utc)


def _timestamp_to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz)
    except ZoneInfoNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported timezone: {tz}") from exc


def _pick_bucket_seconds(
    span_seconds: float,
    target_points: int = AUTO_TARGET_POINTS,
    candidates: tuple[int, ...] = AUTO_BUCKET_CANDIDATES_SECONDS,
) -> int:
    if span_seconds <= 0:
        return candidates[0]
    for bucket_seconds in candidates:
        if math.ceil(span_seconds / bucket_seconds) <= target_points:
            return bucket_seconds
    return candidates[-1]


def _aggregate_timeline_rows(
    rows: list[Any],
    from_utc: datetime,
    bucket_seconds: int,
) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, Any]] = {}

    for row in rows:
        row_ts = _timestamp_to_utc(row.captured_at)
        bucket_idx = int((row_ts - from_utc).total_seconds() // bucket_seconds)
        bucket = buckets.setdefault(
            bucket_idx,
            {"sum": 0.0, "count": 0, "last_timestamp": row_ts},
        )
        bucket["sum"] += float(row.people_count)
        bucket["count"] += 1
        if row_ts > bucket["last_timestamp"]:
            bucket["last_timestamp"] = row_ts

    points: list[dict[str, Any]] = []
    for bucket_idx in sorted(buckets):
        bucket = buckets[bucket_idx]
        points.append(
            {
                "timestamp": bucket["last_timestamp"].isoformat(),
                "value": bucket["sum"] / bucket["count"],
                "samples": bucket["count"],
            }
        )
    return points


def _timeline_data(
    db: Session,
    from_ts: datetime | None,
    to_ts: datetime | None,
    tz: str = "Europe/Helsinki",
) -> dict[str, Any]:
    to_utc = _coerce_utc(to_ts) if to_ts is not None else datetime.now(timezone.utc)
    from_utc = _coerce_utc(from_ts) if from_ts is not None else None

    if from_utc is not None and from_utc >= to_utc:
        raise HTTPException(status_code=400, detail="'from' must be earlier than 'to'")

    base_where = [Capture.status == "ok", Capture.people_count.is_not(None), Capture.captured_at <= to_utc]
    if from_utc is None:
        earliest = db.execute(
            select(Capture.captured_at).where(*base_where).order_by(Capture.captured_at).limit(1)
        ).scalar_one_or_none()
        if earliest is None:
            return {
                "from": to_utc.isoformat(),
                "to": to_utc.isoformat(),
                "tz": tz,
                "mode": "raw",
                "bucket_seconds": None,
                "target_points": AUTO_TARGET_POINTS,
                "points": [],
            }
        from_utc = _timestamp_to_utc(earliest)

    where = [Capture.status == "ok", Capture.people_count.is_not(None)]
    where.append(Capture.captured_at >= from_utc)
    where.append(Capture.captured_at <= to_utc)

    rows = db.execute(
        select(Capture.captured_at, Capture.people_count).where(*where).order_by(Capture.captured_at)
    ).all()

    span_seconds = max((to_utc - from_utc).total_seconds(), 1.0)
    if len(rows) <= AUTO_TARGET_POINTS:
        points = [
            {
                "timestamp": _timestamp_to_utc(row.captured_at).isoformat(),
                "value": float(row.people_count),
                "samples": 1,
            }
            for row in rows
        ]
        return {
            "from": from_utc.isoformat(),
            "to": to_utc.isoformat(),
            "tz": tz,
            "mode": "raw",
            "bucket_seconds": None,
            "target_points": AUTO_TARGET_POINTS,
            "points": points,
        }

    bucket_seconds = _pick_bucket_seconds(span_seconds)
    return {
        "from": from_utc.isoformat(),
        "to": to_utc.isoformat(),
        "tz": tz,
        "mode": "aggregated",
        "bucket_seconds": bucket_seconds,
        "target_points": AUTO_TARGET_POINTS,
        "points": _aggregate_timeline_rows(rows, from_utc, bucket_seconds),
    }


def _daily_average_data(
    db: Session,
    from_ts: datetime | None,
    to_ts: datetime | None,
    tz: str = "Europe/Helsinki",
) -> dict[str, Any]:
    zone = _resolve_zone(tz)
    to_utc = _coerce_utc(to_ts) if to_ts is not None else datetime.now(timezone.utc)
    from_utc = _coerce_utc(from_ts) if from_ts is not None else None

    if from_utc is not None and from_utc >= to_utc:
        raise HTTPException(status_code=400, detail="'from' must be earlier than 'to'")

    filters = [Capture.status == "ok", Capture.people_count.is_not(None), Capture.captured_at <= to_utc]

    if from_utc is None:
        earliest = db.execute(
            select(Capture.captured_at).where(*filters).order_by(Capture.captured_at).limit(1)
        ).scalar_one_or_none()
        if earliest is None:
            return {
                "from": to_utc.isoformat(),
                "to": to_utc.isoformat(),
                "tz": tz,
                "unit": "people_per_capture",
                "points": [],
            }
        from_utc = _timestamp_to_utc(earliest)

    rows = db.execute(
        select(Capture.captured_at, Capture.people_count)
        .where(
            Capture.status == "ok",
            Capture.people_count.is_not(None),
            Capture.captured_at >= from_utc,
            Capture.captured_at <= to_utc,
        )
        .order_by(Capture.captured_at)
    ).all()

    if not rows:
        return {
            "from": from_utc.isoformat(),
            "to": to_utc.isoformat(),
            "tz": tz,
            "unit": "people_per_capture",
            "points": [],
        }

    day_buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        day_key = _timestamp_to_utc(row.captured_at).astimezone(zone).date().isoformat()
        bucket = day_buckets.setdefault(
            day_key,
            {
                "day": day_key,
                "sum": 0.0,
                "samples": 0,
            },
        )
        bucket["sum"] += float(row.people_count)
        bucket["samples"] += 1

    points = [
        {
            "day": key,
            "value": day_buckets[key]["sum"] / day_buckets[key]["samples"],
            "samples": day_buckets[key]["samples"],
        }
        for key in sorted(day_buckets)
    ]
    return {
        "from": from_utc.isoformat(),
        "to": to_utc.isoformat(),
        "tz": tz,
        "unit": "people_per_capture",
        "points": points,
    }


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


@app.get("/days", response_class=HTMLResponse)
def days_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("days.html", {"request": request})


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


@app.get("/top-captures", response_class=HTMLResponse)
def top_captures_page(
    request: Request,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    rows = db.execute(
        select(Capture)
        .where(
            Capture.status == "ok",
            Capture.people_count.is_not(None),
            Capture.annotated_image_bytes.is_not(None),
        )
        .order_by(desc(Capture.people_count), desc(Capture.captured_at))
        .limit(12)
    ).scalars().all()
    return templates.TemplateResponse(
        "top_captures.html",
        {
            "request": request,
            "captures": rows,
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


@app.get("/api/metrics/timeline")
def metrics_timeline(
    from_ts: datetime | None = Query(default=None, alias="from"),
    to_ts: datetime | None = Query(default=None, alias="to"),
    tz: str = Query(default="Europe/Helsinki"),
    db: Session = Depends(get_db),
) -> JSONResponse:
    return JSONResponse(_timeline_data(db, from_ts=from_ts, to_ts=to_ts, tz=tz))


@app.get("/api/metrics/days")
def metrics_days(
    from_ts: datetime | None = Query(default=None, alias="from"),
    to_ts: datetime | None = Query(default=None, alias="to"),
    tz: str = Query(default="Europe/Helsinki"),
    db: Session = Depends(get_db),
) -> JSONResponse:
    return JSONResponse(_daily_average_data(db, from_ts=from_ts, to_ts=to_ts, tz=tz))


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
        media_type=row.annotated_image_mime_type or "image/jpeg",
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
