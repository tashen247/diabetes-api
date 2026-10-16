"""Intake API: accept a study, publish it once, and let the models work through the queues.

    POST /studies          left and right fundus images  ->  202 with the study id
    GET  /studies/{id}     PENDING / NEEDS_ATTENTION / FINAL, with each model's result

Replaces the synchronous /predict-all, which called all eight models in one request and
returned whatever came back, so a model that was down meant an incomplete report.

Run on its own:   uvicorn messaging.intake:app --port 8100
or mount it in app.py with app.include_router(messaging.intake.router).
"""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile, status

from .broker import PublishError, SharedPublisher
from .settings import Settings
from .store import StudyStore

log = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MAX_IMAGE_BYTES = 30 * 1024 * 1024

router = APIRouter(tags=["studies"])

_settings: Settings | None = None
_store: StudyStore | None = None
_publisher: SharedPublisher | None = None


def configure(settings: Settings, store: StudyStore | None = None,
              publisher: SharedPublisher | None = None) -> None:
    """Sets the dependencies. Called on startup, or by tests with their own settings."""
    global _settings, _store, _publisher
    _settings = settings
    _store = store or StudyStore(settings.db_url)
    _publisher = publisher or SharedPublisher(settings)


def _deps() -> tuple[Settings, StudyStore, SharedPublisher]:
    if _settings is None:
        configure(Settings.from_env())
    assert _settings and _store and _publisher
    return _settings, _store, _publisher


def _save_image(upload: UploadFile, folder: Path, eye: str) -> str:
    suffix = Path(upload.filename or "").suffix.lower() or ".jpg"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{eye} eye image must be one of {sorted(ALLOWED_SUFFIXES)}")
    data = upload.file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{eye} eye image is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"{eye} eye image is over 30 MB")
    path = folder / f"{eye}{suffix}"
    path.write_bytes(data)
    return path.name


# Plain def, not async: FastAPI runs it in a worker thread, so the blocking publish
# (which waits for the broker's confirm) doesn't stall the event loop.
@router.post("/studies", status_code=status.HTTP_202_ACCEPTED)
def create_study(referral_id: str = Form(..., min_length=1, max_length=64),
                 left_eye_file: UploadFile = File(...),
                 right_eye_file: UploadFile = File(...)) -> dict:
    s, store, publisher = _deps()
    study_id = str(uuid.uuid4())
    folder = Path(s.image_store_dir) / study_id
    folder.mkdir(parents=True, exist_ok=False)
    try:
        left = _save_image(left_eye_file, folder, "left")
        right = _save_image(right_eye_file, folder, "right")
    except HTTPException:
        _remove(folder)
        raise

    left_rel, right_rel = f"{study_id}/{left}", f"{study_id}/{right}"
    # The study row exists before the message, so the aggregator always knows the study a
    # result belongs to. If the publish then fails, the row is removed and the caller retries.
    store.create_study(study_id, referral_id, left_rel, right_rel, s.conditions)
    message = {
        "study_id": study_id,
        "referral_id": referral_id,
        "left_image": left_rel,
        "right_image": right_rel,
        "conditions": list(s.conditions),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        publisher.publish(s.studies_exchange, "", message, message_id=study_id)
    except PublishError as e:
        log.error("Publishing study %s failed: %s", study_id, e)
        store.delete_study(study_id)
        _remove(folder)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "The screening queue is unavailable. Nothing was stored; please retry.")

    log.info("Study %s for referral %s queued for %d tests", study_id, referral_id, len(s.conditions))
    return {"study_id": study_id, "status": "PENDING", "tests": list(s.conditions),
            "status_url": f"/studies/{study_id}"}


@router.get("/studies/{study_id}")
def get_study(study_id: str) -> dict:
    _, store, _ = _deps()
    study = store.get_study(study_id)
    if study is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Study not found")
    return study


def _remove(folder: Path) -> None:
    for f in folder.glob("*"):
        f.unlink(missing_ok=True)
    try:
        os.rmdir(folder)
    except OSError:
        pass


@asynccontextmanager
async def _lifespan(_: FastAPI):
    yield
    if _publisher is not None:
        _publisher.close()


app = FastAPI(title="Fundus screening intake", version="1.0.0", lifespan=_lifespan)
app.include_router(router)
