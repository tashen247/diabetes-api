"""Calls one existing fundus endpoint for a queued study.

The worker reads images from the image store. Bytes are not carried on the
queue. A connection failure, timeout, or 5xx delays the test. A 4xx that
rejects the image dead-letters it.
"""

from __future__ import annotations

import os

from pipeline.catalog import TestSpec
from pipeline.errors import BadImageError, TransientModelError
from pipeline.store import StoredImage


def transient_reason(status_code: int, text: str) -> str:
    if status_code == 429 or "gpu" in (text or "").lower():
        return "gpu_busy"
    return "model_down"


def classify_http(status_code: int, text: str) -> None:
    """Raise when the model endpoint did not return a result. 200 returns None."""
    if status_code == 200:
        return
    if status_code in {400, 413, 415, 422}:
        raise BadImageError(text[:500] or f"HTTP {status_code}", reason="bad_image")
    raise TransientModelError(text[:500] or f"HTTP {status_code}", reason=transient_reason(status_code, text))


def assign_eyes(images: list[StoredImage]) -> tuple[StoredImage | None, StoredImage | None, list[StoredImage]]:
    left = next((image for image in images if image.eye == "left"), None)
    right = next((image for image in images if image.eye == "right"), None)
    unknown = [image for image in images if image.eye not in {"left", "right"}]
    if left is None and unknown:
        left = unknown.pop(0)
    if right is None and unknown:
        right = unknown.pop(0)
    return left, right, unknown


class HttpModelRunner:
    def __init__(self, timeout: float | None = None):
        self.timeout = timeout if timeout is not None else float(os.getenv("SCREENING_MODEL_TIMEOUT", "120"))

    def run(self, test: TestSpec, images: list[StoredImage], body: dict) -> dict:
        import requests

        url = os.getenv(test.url_env, test.default_url)
        left, right, extras = assign_eyes(images)
        files = {}
        if left is not None:
            files["left_eye_file"] = (left.filename, left.data, left.content_type)
        if right is not None:
            files["right_eye_file"] = (right.filename, right.data, right.content_type)
        for index, extra in enumerate(extras):
            files[f"image_{index}"] = (extra.filename, extra.data, extra.content_type)
        if not files:
            raise BadImageError("study has no readable images", reason="unreadable_image")

        headers = {}
        token = os.getenv("AI_BEARER_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = requests.post(
                url,
                params={"referral_id": body.get("referral_id", "")},
                files=files,
                headers=headers or None,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise TransientModelError(str(exc), reason="model_down") from exc
        except requests.ConnectionError as exc:
            raise TransientModelError(str(exc), reason="model_down") from exc

        classify_http(response.status_code, response.text)
        try:
            payload = response.json()
        except ValueError:
            payload = {"raw_text": response.text}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        return payload


class ScriptedRunner:
    """Test double. Each test name maps to a list of exceptions or result dicts."""

    def __init__(self, scripts: dict | None = None):
        self.scripts = {name: list(steps) for name, steps in (scripts or {}).items()}
        self.calls: list[str] = []

    def run(self, test: TestSpec, images: list[StoredImage], body: dict) -> dict:
        del body
        self.calls.append(test.name)
        for image in images:
            if not image.data:
                raise BadImageError("empty image", reason="unreadable_image")
        steps = self.scripts.get(test.name)
        if not steps:
            return {"label": "No Suspect or Negative", "test": test.name}
        step = steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step
