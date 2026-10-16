"""Stand-in model server for demos: the same endpoints as the real models, without GPUs or weights.

    uvicorn messaging.stub_models:app --port 8090

Each /predict-<condition> endpoint answers like a model. Switch a model's behaviour while the
pipeline is running to show what RabbitMQ does with each failure:

    curl -X PUT localhost:8090/behaviour/rvo/down        # 503: retried, then dead-lettered
    curl -X PUT localhost:8090/behaviour/cnv/bad_image   # 422: dead-lettered at once
    curl -X PUT localhost:8090/behaviour/rvo/ok          # back to normal
"""
from __future__ import annotations

import random

from fastapi import FastAPI, File, HTTPException, UploadFile

from .settings import DEFAULT_MODEL_URLS

app = FastAPI(title="Stand-in screening models")
behaviour: dict[str, str] = {c: "ok" for c in DEFAULT_MODEL_URLS}
calls: dict[str, int] = {c: 0 for c in DEFAULT_MODEL_URLS}


def _endpoint(condition: str):
    async def predict(referral_id: str, left_eye_file: UploadFile = File(...),
                      right_eye_file: UploadFile = File(...)) -> dict:
        calls[condition] += 1
        mode = behaviour[condition]
        if mode == "down":
            raise HTTPException(503, f"{condition} model is down")
        if mode == "bad_image":
            raise HTTPException(422, "image is ungradable")
        await left_eye_file.read(), await right_eye_file.read()
        return {
            "referral_id": referral_id,
            "left": {"label": "No Suspect or Negative", "confidence": round(random.uniform(0.8, 0.99), 3)},
            "right": {"label": "No Suspect or Negative", "confidence": round(random.uniform(0.8, 0.99), 3)},
        }
    return predict


for _condition, _url in DEFAULT_MODEL_URLS.items():
    app.post("/" + _url.rsplit("/", 1)[1])(_endpoint(_condition))


@app.put("/behaviour/{condition}/{mode}")
def set_behaviour(condition: str, mode: str) -> dict:
    if condition not in behaviour or mode not in ("ok", "down", "bad_image"):
        raise HTTPException(400, "condition must be a known model; mode one of ok, down, bad_image")
    behaviour[condition] = mode
    return {"behaviour": behaviour}


@app.get("/behaviour")
def get_behaviour() -> dict:
    return {"behaviour": behaviour, "calls": calls}
