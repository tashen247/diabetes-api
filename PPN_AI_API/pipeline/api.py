"""Intake and operations HTTP API.

POST /intake/studies stores the images, creates the eight pending tests, and
publishes the study once. GET report stays PENDING until all eight results
are in. It is never returned as a complete report before that.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipeline.catalog import EXPECTED_TESTS, require_test
from pipeline.errors import BrokerUnavailable
from pipeline.flow import accept_study, queue_snapshot, recover_unpublished, replay_dead_letter
from pipeline.runtime import Deps, get_singleton

router = APIRouter()

MAX_IMAGES = 32
MAX_IMAGE_BYTES = 32 * 1024 * 1024


class ReplayRequest(BaseModel):
    study_id: str
    test: str


def get_deps(request: Request) -> Deps:
    deps = getattr(request.app.state, "screening", None)
    if deps is None:
        deps = get_singleton()
        request.app.state.screening = deps
    return deps


def create_app(deps: Deps | None = None) -> FastAPI:
    app = FastAPI(
        title="Fundus screening intake",
        version="1.0.0",
        description="One study, eight models. The report stays pending until every result is in.",
    )
    if deps is not None:
        app.state.screening = deps
    app.include_router(router)
    return app


@router.get("/intake/health")
def health(deps: Deps = Depends(get_deps)):
    broker = "up"
    reason = None
    results_depth = None
    try:
        results_depth = deps.broker.depth("q.results")
    except Exception as exc:
        broker = "down"
        reason = str(exc)
    body = {
        "broker": broker,
        "tests_expected": EXPECTED_TESTS,
        "results_queue_depth": results_depth,
        **deps.store.counts(),
    }
    if reason:
        body["broker_reason"] = reason
    return body


@router.post("/intake/studies", status_code=202)
async def create_study(
    referral_id: str = Form(...),
    images: list[UploadFile] = File(...),
    crm_callback_url: str | None = Form(None),
    deps: Deps = Depends(get_deps),
):
    referral_id = referral_id.strip()
    if not referral_id:
        raise HTTPException(status_code=400, detail="referral_id is required")
    if not images:
        raise HTTPException(status_code=400, detail="upload at least one fundus image")
    if len(images) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"at most {MAX_IMAGES} images per study")

    callback = (crm_callback_url or os.getenv("CRM_CALLBACK_URL") or "").strip() or None
    if callback is not None:
        parsed = urlparse(callback)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(status_code=400, detail="crm_callback_url must be an http(s) URL")

    stored: list[tuple[str, str, bytes]] = []
    for upload in images:
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"{upload.filename or 'image'} is empty")
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail=f"{upload.filename or 'image'} is larger than 32MB")
        stored.append((upload.filename or "fundus.jpg", upload.content_type or "application/octet-stream", data))

    report = accept_study(deps, referral_id, stored, callback)
    return JSONResponse(status_code=202, content=report, headers={"Cache-Control": "no-store"})


@router.get("/intake/studies/{study_id}")
def get_study(study_id: str, deps: Deps = Depends(get_deps)):
    return _report_response(deps, study_id)


@router.get("/intake/studies/{study_id}/report")
def get_report(study_id: str, deps: Deps = Depends(get_deps)):
    return _report_response(deps, study_id)


@router.get("/operations/studies/{study_id}")
def inspect_study(study_id: str, deps: Deps = Depends(get_deps)):
    report = _load_report(deps, study_id)
    report["alerts"] = deps.store.alerts(study_id)
    return JSONResponse(content=report, headers={"Cache-Control": "no-store"})


@router.get("/operations/alerts")
def list_alerts(deps: Deps = Depends(get_deps)):
    return {"alerts": deps.store.alerts()}


@router.get("/operations/dead-letters")
def list_dead_letters(deps: Deps = Depends(get_deps)):
    depth = None
    try:
        depth = deps.broker.depth("q.dead-letter")
    except Exception:
        depth = None
    return {"queue_depth": depth, "items": deps.store.list_dead()}


@router.get("/operations/queues")
def list_queues(deps: Deps = Depends(get_deps)):
    try:
        return {"queues": queue_snapshot(deps.broker)}
    except BrokerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/operations/replay")
def replay(payload: ReplayRequest, deps: Deps = Depends(get_deps)):
    study_id = payload.study_id.strip()
    test_name = payload.test.strip()
    if not study_id or not test_name:
        raise HTTPException(status_code=400, detail="study_id and test are required")
    try:
        require_test(test_name)
        result = replay_dead_letter(deps, study_id, test_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no dead letter to replay ({exc})") from exc
    except BrokerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JSONResponse(status_code=202, content=result)


@router.post("/operations/studies/{study_id}/publish")
def republish(study_id: str, deps: Deps = Depends(get_deps)):
    try:
        deps.store.work_body(study_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="study not found") from exc
    # recover_unpublished publishes every unpublished study. Force this one by
    # clearing published_at only when the caller asked for this id.
    _mark_unpublished(deps, study_id)
    try:
        published = recover_unpublished(deps)
    except BrokerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    report = _load_report(deps, study_id)
    report["queued"] = report["published"]
    report["republished_studies"] = published
    return JSONResponse(status_code=202, content=report)


def _report_response(deps: Deps, study_id: str) -> JSONResponse:
    return JSONResponse(content=_load_report(deps, study_id), headers={"Cache-Control": "no-store"})


def _load_report(deps: Deps, study_id: str) -> dict:
    try:
        return deps.store.report(study_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="study not found") from exc


def _mark_unpublished(deps: Deps, study_id: str) -> None:
    with deps.store._tx() as conn:
        conn.execute("UPDATE studies SET published_at=NULL WHERE study_id=?", (study_id,))
