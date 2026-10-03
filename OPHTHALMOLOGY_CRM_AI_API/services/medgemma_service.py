"""
MedGemma fundus screening service.

Calls a Gemma / MedGemma server that is already running on the same machine
(Ollama, llama.cpp, LM Studio, or any OpenAI-compatible vision API).
This process does not download or load model weights.

  GEMMA_BASE_URL   default http://127.0.0.1:11434
  GEMMA_MODEL      default: a pulled tag containing "medgemma", else "gemma"
  GEMMA_TIMEOUT    seconds to wait for one generation (default 180)
  GEMMA_API_KEY    optional bearer token
  MEDGEMMA_PORT    default 8030

  POST /screen-pair   left_eye_file + right_eye_file
  POST /screen        one eye (form field eye=left|right)
  GET  /health
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from typing import Any, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("medgemma_service")

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DISCLAIMER = (
    "Screening support from a local Gemma model. "
    "Not a clinical diagnosis and not a substitute for an ophthalmologist."
)

# Labels match the rest of OPHTHALMOLOGY_CRM_AI_API. First label of each condition is the negative grade.
ADS: dict[str, dict[str, int]] = {
    "diabetes": {
        "No Suspect or Negative": 2,
        "Mild": 3,
        "Moderate": 4,
        "Severe": 5,
        "Proliferative Diabetic Retinopathy (PDR)": 6,
    },
    "hypertension": {
        "No Suspect or Negative": 7,
        "Decreased Retinal Artery Elasticity": 8,
        "Hypertensive Retinopathy Grade 1 or 2": 9,
        "Hypertensive Retinopathy Grade 3 or 4": 10,
    },
    "macular_degeneration": {
        "No Suspect or Negative": 11,
        "Drusen": 12,
        "AMD Early or Intermediate Stage": 13,
        "AMD Advanced Stage": 14,
        "Large Drusen with Abnormal Pigmentation": 15,
    },
    "glaucoma": {
        "No Suspect or Negative": 16,
        "Suspect or Positive": 17,
    },
    "pathological": {
        "No Suspect or Negative": 18,
        "Minor Pathological": 19,
        "Obvious Pathological": 20,
    },
    "cnv": {
        "No Suspect or Negative": 22,
        "Suspect or Positive": 23,
    },
    "rvo": {
        "No Suspect or Negative": 24,
        "Suspect or Positive": 25,
    },
    "myopia": {
        "No Suspect or Negative": 26,
        "Myopia": 27,
    },
}
NEG_LABEL = {name: next(iter(labels)) for name, labels in ADS.items()}
NEG_ADS = {name: labels[NEG_LABEL[name]] for name, labels in ADS.items()}

SCREEN_PROMPT = """You are screening a fundus photograph of the patient's {eye} eye.
Act as an ophthalmologist doing a screening read. Judge only what is visible in this image.
If the photo is not a fundus image, is too dark, or is ungradable, set image_quality to "ungradable" and use the negative grade for every condition.

Return ONLY JSON, no markdown, with this shape:
{{
  "image_quality": "adequate" | "limited" | "ungradable",
  "impression": "one sentence",
  "referral_recommended": false,
  "conditions": {{
    "diabetes": {{"grade": "No Suspect or Negative | Mild | Moderate | Severe | Proliferative Diabetic Retinopathy (PDR)", "confidence": 0.0, "findings": ""}},
    "hypertension": {{"grade": "No Suspect or Negative | Decreased Retinal Artery Elasticity | Hypertensive Retinopathy Grade 1 or 2 | Hypertensive Retinopathy Grade 3 or 4", "confidence": 0.0, "findings": ""}},
    "macular_degeneration": {{"grade": "No Suspect or Negative | Drusen | AMD Early or Intermediate Stage | AMD Advanced Stage | Large Drusen with Abnormal Pigmentation", "confidence": 0.0, "findings": ""}},
    "cnv": {{"grade": "No Suspect or Negative | Suspect or Positive", "confidence": 0.0, "findings": ""}},
    "rvo": {{"grade": "No Suspect or Negative | Suspect or Positive", "confidence": 0.0, "findings": ""}},
    "glaucoma": {{"grade": "No Suspect or Negative | Suspect or Positive", "confidence": 0.0, "findings": ""}},
    "pathological": {{"grade": "No Suspect or Negative | Minor Pathological | Obvious Pathological", "confidence": 0.0, "findings": ""}},
    "myopia": {{"grade": "No Suspect or Negative | Myopia", "confidence": 0.0, "findings": ""}}
  }}
}}"""

SUMMARY_PROMPT = """You are comparing the left-eye and right-eye fundus screening JSON below.
Write a short bilateral screening note. Return ONLY JSON:
{{
  "impression": "one or two sentences comparing the eyes",
  "referral_recommended": false,
  "asymmetry_note": "what differs between the eyes, or an empty string"
}}

LEFT:
{left}

RIGHT:
{right}
"""


def base_url() -> str:
    return os.getenv("GEMMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def request_timeout() -> float:
    return float(os.getenv("GEMMA_TIMEOUT", "180"))


def _headers() -> dict[str, str]:
    key = os.getenv("GEMMA_API_KEY", "").strip()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _timeout(read_seconds: float) -> httpx.Timeout:
    return httpx.Timeout(connect=5.0, read=read_seconds, write=30.0, pool=5.0)


def jpeg_bytes(image: Image.Image, max_edge: int = 896) -> bytes:
    image = image.convert("RGB")
    width, height = image.size
    scale = max_edge / max(width, height)
    if scale < 1:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def load_upload(upload: UploadFile) -> bytes:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")
    ctype = (upload.content_type or "").lower()
    if ctype.startswith("text/"):
        raise HTTPException(status_code=400, detail="Invalid image type. Only JPEG and PNG are supported.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read the image file.") from exc
    return jpeg_bytes(image)


def parse_json_object(raw: str) -> Optional[dict[str, Any]]:
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(min(1.0, max(0.0, number)), 3)


def resolve_grade(condition: str, raw_grade: str) -> tuple[str, int, Optional[str]]:
    labels = ADS[condition]
    grade = (raw_grade or "").strip()
    if grade in labels:
        return grade, labels[grade], None
    lowered = grade.lower()
    for label, ads_id in labels.items():
        if label.lower() == lowered or label.lower() in lowered or lowered in label.lower():
            return label, ads_id, None
    negative = NEG_LABEL[condition]
    return negative, NEG_ADS[condition], grade or None


def normalize_eye(parsed: Optional[dict[str, Any]], raw: str) -> dict[str, Any]:
    if parsed is None:
        conditions = {
            name: {
                "grade": NEG_LABEL[name],
                "ads_id": NEG_ADS[name],
                "confidence": 0.0,
                "findings": "The model response was not JSON.",
                "unrecognized_grade": None,
            }
            for name in ADS
        }
        return {
            "image_quality": "limited",
            "impression": "",
            "referral_recommended": False,
            "conditions": conditions,
            "any_positive": False,
            "parse_failed": True,
            "raw": raw[:2000],
        }

    source = parsed.get("conditions") if isinstance(parsed.get("conditions"), dict) else {}
    conditions: dict[str, Any] = {}
    for name in ADS:
        item = source.get(name) if isinstance(source.get(name), dict) else {}
        grade, ads_id, unrecognized = resolve_grade(name, str(item.get("grade", "")))
        conditions[name] = {
            "grade": grade,
            "ads_id": ads_id,
            "confidence": _clamp_confidence(item.get("confidence", 0.0)),
            "findings": str(item.get("findings", "") or ""),
            "unrecognized_grade": unrecognized,
        }
    any_positive = any(item["ads_id"] != NEG_ADS[name] for name, item in conditions.items())
    model_referral = bool(parsed.get("referral_recommended"))
    return {
        "image_quality": str(parsed.get("image_quality") or "limited"),
        "impression": str(parsed.get("impression") or ""),
        "referral_recommended": model_referral or any_positive,
        "conditions": conditions,
        "any_positive": any_positive,
        "parse_failed": False,
        "raw": raw[:2000],
    }


class GemmaClient:
    """Talks to a local Ollama or OpenAI-compatible Gemma server."""

    def __init__(self, root: Optional[str] = None, model: Optional[str] = None, timeout: Optional[float] = None):
        self.root = (root or base_url()).rstrip("/")
        self.model_override = model if model is not None else os.getenv("GEMMA_MODEL", "").strip()
        self.timeout = timeout if timeout is not None else request_timeout()
        self.protocol: Optional[str] = None
        self.model: Optional[str] = None
        self.openai_url: Optional[str] = None

    def probe(self) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=_timeout(5.0), headers=_headers()) as client:
                ollama = client.get(f"{self.root}/api/tags")
                if ollama.status_code == 200:
                    names = _ollama_names(ollama.json())
                    self.protocol = "ollama"
                    self.model = _choose_model(names, self.model_override)
                    self.openai_url = None
                    return self._status(True, None)
                openai, url = _find_openai(client, self.root)
                if openai is not None:
                    names = _openai_names(openai.json())
                    self.protocol = "openai"
                    self.model = _choose_model(names, self.model_override)
                    self.openai_url = url
                    return self._status(True, None)
            return self._status(False, "No Ollama /api/tags or OpenAI /v1/models response")
        except httpx.HTTPError as exc:
            self.protocol = None
            self.model = self.model_override or None
            return self._status(False, str(exc))

    def complete(self, prompt: str, images: Optional[list[bytes]] = None) -> str:
        status = self.probe()
        if not status["reachable"]:
            raise GemmaUnreachable(status["error"] or "Gemma server is not reachable", status)
        images = images or []
        try:
            with httpx.Client(timeout=_timeout(self.timeout), headers=_headers()) as client:
                if self.protocol == "ollama":
                    response = client.post(f"{self.root}/api/chat", json=_ollama_body(self.model, prompt, images))
                else:
                    response = client.post(self.openai_url, json=_openai_body(self.model, prompt, images))
        except httpx.HTTPError as exc:
            raise GemmaUnreachable(str(exc), self._status(False, str(exc))) from exc
        if response.status_code >= 400:
            detail = response.text[:500]
            raise GemmaUpstreamError(response.status_code, detail)
        return _message_text(response.json(), self.protocol)

    def _status(self, reachable: bool, error: Optional[str]) -> dict[str, Any]:
        return {
            "reachable": reachable,
            "base_url": self.root,
            "protocol": self.protocol,
            "model": self.model,
            "error": error,
        }


class GemmaUnreachable(Exception):
    def __init__(self, message: str, status: dict[str, Any]):
        super().__init__(message)
        self.status = status


class GemmaUpstreamError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _ollama_names(payload: Any) -> list[str]:
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []
    return [str(item.get("name")) for item in models if isinstance(item, dict) and item.get("name")]


def _openai_names(payload: Any) -> list[str]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    return [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]


def _choose_model(names: list[str], override: str) -> str:
    if override:
        return override
    lowered = [(name, name.lower()) for name in names]
    for name, label in lowered:
        if "medgemma" in label:
            return name
    for name, label in lowered:
        if "gemma" in label:
            return name
    if names:
        return names[0]
    return "medgemma"


def _find_openai(client: httpx.Client, root: str) -> tuple[Optional[httpx.Response], Optional[str]]:
    candidates = []
    if root.endswith("/v1"):
        candidates.append((f"{root}/models", f"{root}/chat/completions"))
    else:
        candidates.append((f"{root}/v1/models", f"{root}/v1/chat/completions"))
        candidates.append((f"{root}/models", f"{root}/chat/completions"))
    for models_url, chat_url in candidates:
        try:
            response = client.get(models_url)
        except httpx.HTTPError:
            continue
        if response.status_code == 200:
            return response, chat_url
    return None, None


def _ollama_body(model: str, prompt: str, images: list[bytes]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if images:
        message["images"] = [base64.b64encode(image).decode("ascii") for image in images]
    return {"model": model, "stream": False, "messages": [message]}


def _openai_body(model: str, prompt: str, images: list[bytes]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        encoded = base64.b64encode(image).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    return {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }


def _message_text(payload: Any, protocol: Optional[str]) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    if protocol == "ollama":
        message = payload.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content") or "")
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
            return "".join(parts)
    if isinstance(payload.get("response"), str):
        return payload["response"]
    return json.dumps(payload)


def screen_eye(client: GemmaClient, image: bytes, eye: str) -> dict[str, Any]:
    raw = client.complete(SCREEN_PROMPT.format(eye=eye.upper()), [image])
    result = normalize_eye(parse_json_object(raw), raw)
    result["eye"] = eye
    return result


def bilateral_summary(client: GemmaClient, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    raw = client.complete(
        SUMMARY_PROMPT.format(
            left=json.dumps(_public_eye(left), ensure_ascii=True),
            right=json.dumps(_public_eye(right), ensure_ascii=True),
        ),
        [],
    )
    parsed = parse_json_object(raw)
    if parsed is None:
        return {"impression": "", "referral_recommended": False, "asymmetry_note": "", "parse_failed": True, "raw": raw[:2000]}
    return {
        "impression": str(parsed.get("impression") or ""),
        "referral_recommended": bool(parsed.get("referral_recommended")),
        "asymmetry_note": str(parsed.get("asymmetry_note") or ""),
        "parse_failed": False,
        "raw": raw[:2000],
    }


def _public_eye(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "eye": result.get("eye"),
        "image_quality": result.get("image_quality"),
        "impression": result.get("impression"),
        "referral_recommended": result.get("referral_recommended"),
        "conditions": result.get("conditions"),
    }


def screen_pair(client: GemmaClient, left_image: bytes, right_image: bytes) -> dict[str, Any]:
    left = screen_eye(client, left_image, "left")
    right = screen_eye(client, right_image, "right")
    summary = bilateral_summary(client, left, right)
    any_positive = bool(left["any_positive"] or right["any_positive"])
    return {
        "disclaimer": DISCLAIMER,
        "model": client.model,
        "upstream": client.root,
        "left": _public_eye(left) | {"any_positive": left["any_positive"], "parse_failed": left["parse_failed"]},
        "right": _public_eye(right) | {"any_positive": right["any_positive"], "parse_failed": right["parse_failed"]},
        "bilateral_summary": summary,
        "any_positive": any_positive,
        "referral_recommended": any_positive or bool(summary.get("referral_recommended")),
    }


def _upstream_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, GemmaUnreachable):
        return HTTPException(
            status_code=503,
            detail={
                "message": "The local Gemma server is not reachable. Start it on this machine or set GEMMA_BASE_URL.",
                "base_url": exc.status.get("base_url"),
                "error": exc.status.get("error"),
            },
        )
    if isinstance(exc, GemmaUpstreamError):
        return HTTPException(
            status_code=502,
            detail={"message": "The local Gemma server returned an error.", "status_code": exc.status_code, "body": exc.detail},
        )
    raise exc


app = FastAPI(
    title="MedGemma fundus screening",
    description="Screens a patient's left and right fundus photographs with a Gemma server already running on this machine.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "MedGemma fundus screening API",
        "disclaimer": DISCLAIMER,
        "gemma_base_url": base_url(),
        "usage": "POST /screen-pair with left_eye_file and right_eye_file",
    }


@app.get("/health")
def health():
    status = GemmaClient().probe()
    return {
        "status": "ok",
        "model_reachable": status["reachable"],
        "base_url": status["base_url"],
        "protocol": status["protocol"],
        "model": status["model"],
        "error": status["error"],
    }


@app.post("/screen")
def screen(
    file: UploadFile = File(...),
    eye: str = Form("left"),
    referral_id: str = Form("0"),
):
    eye_name = eye.strip().lower()
    if eye_name not in {"left", "right"}:
        raise HTTPException(status_code=400, detail="eye must be left or right")
    image = load_upload(file)
    client = GemmaClient()
    try:
        result = screen_eye(client, image, eye_name)
    except (GemmaUnreachable, GemmaUpstreamError) as exc:
        raise _upstream_http_error(exc) from exc
    return JSONResponse({
        "referral_id": referral_id,
        "disclaimer": DISCLAIMER,
        "model": client.model,
        "upstream": client.root,
        "result": _public_eye(result) | {"any_positive": result["any_positive"], "parse_failed": result["parse_failed"]},
    })


@app.post("/screen-pair")
def screen_pair_endpoint(
    left_eye_file: UploadFile = File(...),
    right_eye_file: UploadFile = File(...),
    referral_id: str = Form("0"),
):
    left_image = load_upload(left_eye_file)
    right_image = load_upload(right_eye_file)
    client = GemmaClient()
    try:
        payload = screen_pair(client, left_image, right_image)
    except (GemmaUnreachable, GemmaUpstreamError) as exc:
        raise _upstream_http_error(exc) from exc
    payload["referral_id"] = referral_id
    payload["left"]["filename"] = left_eye_file.filename
    payload["right"]["filename"] = right_eye_file.filename
    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("MEDGEMMA_PORT", "8030")))
