# pathology_service.py

# -*- coding: utf-8 -*-

"""

FastAPI inference service for 4-class Macular Pathology (ADS-aligned)

Classes (fixed indices):

  0: No Suspect or Negative (ADS_ID=18)

  1: Minor Pathological     (ADS_ID=19)

  2: Obvious Pathological   (ADS_ID=20)

  3: Other Macular Pathology(ADS_ID=21)

"""

from __future__ import annotations

import io

import os

import json

import base64

import logging

from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException

from fastapi.responses import JSONResponse

from pydantic import BaseModel

from PIL import Image, ImageOps

import numpy as np

import torch
import torchvision.transforms as T
try:
    import timm
except Exception as e:
    raise RuntimeError("timm not installed. Run: pip install timm") from e

# ----------------------- Constants & Schema -----------------------
IDX2NAME = [
    "No Suspect or Negative",  # 0 -> ADS 18
    "Minor Pathological",  # 1 -> ADS 19
    "Obvious Pathological",  # 2 -> ADS 20
    "Other Macular Pathology",  # 3 -> ADS 21
]

IDX2ADS = {0: 18, 1: 19, 2: 20, 3: 21}
N_CLASSES = 4

# ----------------------- Config helpers --------------------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


# Per-class thresholds from environment (or hardcoded defaults)

_default_thresh = {
    0: 0.0014610290527343,
    1: 0.93994140625,
    2: 0.9814453125,
    3: 0.9072265625,
}

_env_thresh = os.getenv("MACULAR_THRESH")

if _env_thresh:
    try:
        _loaded = json.loads(_env_thresh)

        CLASS_THRESH: Dict[int, float] = {int(k): float(v) for k, v in _loaded.items()}

    except Exception:

        CLASS_THRESH = _default_thresh

else:

    CLASS_THRESH = _default_thresh
# ----------------------- Threshold Configuration --------------------------

# First, try to load thresholds from your fine-tune analyzer JSON file.

# If unavailable, fall back to defaults or environment variables.

THRESHOLD_FILE = r"C:\AI_Images\pathology_reports\pathology_thresholds.json"


def load_thresholds_from_json(path: str):
    """Load thresholds JSON created by the fine-tune discrepancy analyzer."""

    if not os.path.exists(path):
        logging.warning(f"Threshold file not found: {path}. Using defaults.")

        return None

    try:

        with open(path, "r", encoding="utf-8") as f:

            data = json.load(f)

        if "CLASS_THRESH" in data:

            logging.info(f"✅ Loaded thresholds from {path}")

            return data

        else:

            logging.warning(f"No CLASS_THRESH found in {path}.")

    except Exception as e:

        logging.error(f"⚠️ Failed to load thresholds: {e}")

    return None


# --- Load thresholds from JSON, or fallback to env/default ---

# _custom = load_thresholds_from_json(THRESHOLD_FILE)
#
# if _custom:
#
#     CLASS_THRESH = {int(k): float(v) for k, v in _custom["CLASS_THRESH"].items()}
#
#     MIN_MARGIN = float(_custom.get("MIN_MARGIN", 0.10))
#
#     FALLBACK_NO_SUSPECT = float(_custom.get("FALLBACK_NO_SUSPECT", 0.50))
#
# else:
#
#     logging.warning("⚠️ Using default thresholds since JSON not found or invalid.")
#
#     CLASS_THRESH = {0: 0.0014, 1: 0.94, 2: 0.98, 3: 0.91}
#
#     MIN_MARGIN = 0.10
#
#     FALLBACK_NO_SUSPECT = 0.50
#
# logging.info(f"CLASS_THRESH = {CLASS_THRESH}")
#
# logging.info(f"MIN_MARGIN = {MIN_MARGIN}, FALLBACK_NO_SUSPECT = {FALLBACK_NO_SUSPECT}")


MODEL_NAME = os.getenv("MODEL_NAME", "tf_efficientnet_b0_ns")

WEIGHTS = os.getenv("WEIGHTS", r"C:\source_controls\development_sanele\weights\pathology_last_model.pt")

IMG_SIZE = _env_int("IMG_SIZE", 512)

# margin gate for pathology vs no-suspect

MIN_MARGIN = _env_float("MIN_MARGIN", 0.10)

# if we demote pathology, we only accept "No Suspect" if its prob is at least this:

FALLBACK_NO_SUSPECT = _env_float("FALLBACK_NO_SUSPECT", 0.50)

MAX_IMAGE_MB = _env_int("MAX_IMAGE_MB", 20)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AMP = torch.cuda.is_available()

# ----------------------- Logging ---------------------------------

logging.basicConfig(

    level=logging.INFO,

    format="[%(asctime)s] %(levelname)s: %(message)s"

)

logger = logging.getLogger("pathology_api")


# ----------------------- Model Service ----------------------------

class ModelService:

    def __init__(self, model_name: str, weights_path: str, img_size: int):

        self.model_name = model_name

        self.weights_path = weights_path

        self.img_size = img_size

        self.device = DEVICE

        self.amp = AMP

        self.model = None

        self.tf = T.Compose([

            T.Resize((self.img_size, self.img_size)),

            T.ToTensor(),  # matches training: no torchvision Normalize here

        ])

        self._load()

    def _load(self):

        if not os.path.exists(self.weights_path):
            logger.warning("Weights not found at %s", self.weights_path)

        logger.info("Loading model %s on %s", self.model_name, self.device)

        model = timm.create_model(self.model_name,

                                  pretrained=False,

                                  num_classes=N_CLASSES)

        if os.path.exists(self.weights_path):

            ckpt = torch.load(self.weights_path, map_location=self.device)

            state_dict = ckpt.get("model", ckpt)

            cleaned = {}

            for k, v in state_dict.items():
                nk = k[7:] if k.startswith("module.") else k

                cleaned[nk] = v

            model.load_state_dict(cleaned, strict=True)

        self.model = model.to(self.device).eval()

        logger.info("Model ready. Weights: %s", self.weights_path)

    @torch.inference_mode()
    def predict_from_pil(self, img: Image.Image) -> np.ndarray:

        img = ImageOps.exif_transpose(img).convert("RGB")

        tensor = self.tf(img).unsqueeze(0)  # (1,C,H,W)

        with torch.cuda.amp.autocast(enabled=self.amp):
            logits = self.model(tensor.to(self.device, non_blocking=True))

            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        return probs


service = ModelService(MODEL_NAME, WEIGHTS, IMG_SIZE)

app = FastAPI(title="Macular Pathology API", version="1.0.0")


# ----------------------- Post-process -----------------------------

def postprocess_probs(probs_np: np.ndarray) -> Dict[str, Any]:
    """

    Apply dynamic thresholds and margin gating.

    Uses CLASS_THRESH, MIN_MARGIN, and FALLBACK_NO_SUSPECT loaded at startup.

    """

    top = int(np.argmax(probs_np))

    top_p = float(probs_np[top])

    # Sorted indices for margin calculation

    sorted_idx = probs_np.argsort()[::-1]

    margin = (

        float(probs_np[sorted_idx[0]] - probs_np[sorted_idx[1]])

        if probs_np.shape[0] > 1

        else 1.0

    )

    # --- Case 1: Pathology class fails its threshold

    if top != 0 and top_p < CLASS_THRESH.get(top, 0.0):

        reason = f"{IDX2NAME[top]} below threshold {CLASS_THRESH.get(top):.3f}"

        if float(probs_np[0]) >= FALLBACK_NO_SUSPECT:

            return {

                "pred_idx": 0,

                "pred_name": IDX2NAME[0],

                "ads_id": IDX2ADS[0],

                "prob": float(probs_np[0]),

                "gated": True,

                "reason": reason,

                "margin": margin,

                "fallback": "NoSuspect",

            }

        else:

            return {

                "pred_idx": -1,

                "pred_name": "Uncertain",

                "ads_id": None,

                "prob": top_p,

                "gated": True,

                "reason": reason,

                "margin": margin,

                "fallback": "Uncertain",

            }

    # --- Case 2: Margin too small → uncertain or fallback

    if top != 0 and margin < MIN_MARGIN:

        reason = f"Margin {margin:.3f} < MIN_MARGIN {MIN_MARGIN:.3f}"

        if float(probs_np[0]) >= FALLBACK_NO_SUSPECT:

            return {

                "pred_idx": 0,

                "pred_name": IDX2NAME[0],

                "ads_id": IDX2ADS[0],

                "prob": float(probs_np[0]),

                "gated": True,

                "reason": reason,

                "margin": margin,

                "fallback": "NoSuspect",

            }

        else:

            return {

                "pred_idx": -1,

                "pred_name": "Uncertain",

                "ads_id": None,

                "prob": top_p,

                "gated": True,

                "reason": reason,

                "margin": margin,

                "fallback": "Uncertain",

            }

    # --- Case 3: Normal accepted top prediction

    return {

        "pred_idx": top,

        "pred_name": IDX2NAME[top],

        "ads_id": IDX2ADS[top],

        "prob": top_p,

        "gated": False,

        "reason": None,

        "margin": margin,

        "fallback": None,

    }


# ----------------------- Pydantic req model -----------------------

class PredictB64Request(BaseModel):
    image_b64: str


# ----------------------- Utility: confidence map ------------------

def probs_to_confmap(probs_np: np.ndarray) -> Dict[str, float]:
    return {

        IDX2NAME[i]: float(probs_np[i])

        for i in range(N_CLASSES)

    }


# ----------------------- Health / metadata ------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {

        "status": "ok",

        "device": DEVICE,

        "amp": AMP,

        "model_name": MODEL_NAME,

        "weights": WEIGHTS,

        "img_size": IMG_SIZE,

    }


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    return {

        "classes": [

            {"index": i, "name": IDX2NAME[i], "ads_id": IDX2ADS[i]}

            for i in range(N_CLASSES)

        ],

        "thresholds": CLASS_THRESH,

        "min_margin": MIN_MARGIN,

        "fallback_no_suspect": FALLBACK_NO_SUSPECT,

        "model_name": MODEL_NAME,

        "img_size": IMG_SIZE,

    }


# ----------------------- Single-eye predict -----------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    try:

        raw = await file.read()

        img = Image.open(io.BytesIO(raw))

        probs = service.predict_from_pil(img)

    except Exception as e:

        logger.exception("Prediction failed: %s", e)

        raise HTTPException(status_code=400, detail=f"Invalid image or prediction error: {e}")

    decision = postprocess_probs(probs)

    preds_sorted = sorted(

        [

            {

                "index": i,

                "name": IDX2NAME[i],

                "ads_id": IDX2ADS[i],

                "prob": float(probs[i]),

            }

            for i in range(N_CLASSES)

        ],

        key=lambda x: x["prob"],

        reverse=True,

    )

    return JSONResponse({

        "top": preds_sorted[0],

        "decision": decision,

        "predictions": preds_sorted,

        "version": "1.0.0",

    })


# ----------------------- Single-eye base64 ------------------------

@app.post("/predict-b64")
async def predict_b64(req: PredictB64Request) -> JSONResponse:
    try:

        b64 = req.image_b64.split(",")[-1]

        raw = base64.b64decode(b64)

        img = Image.open(io.BytesIO(raw))

        probs = service.predict_from_pil(img)

    except Exception as e:

        logger.exception("Prediction failed: %s", e)

        raise HTTPException(status_code=400, detail=f"Invalid base64 image or prediction error: {e}")

    decision = postprocess_probs(probs)

    preds_sorted = sorted(

        [

            {

                "index": i,

                "name": IDX2NAME[i],

                "ads_id": IDX2ADS[i],

                "prob": float(probs[i]),

            }

            for i in range(N_CLASSES)

        ],

        key=lambda x: x["prob"],

        reverse=True,

    )

    return JSONResponse({

        "top": preds_sorted[0],

        "decision": decision,

        "predictions": preds_sorted,

        "version": "1.0.0",

    })


# ----------------------- Pair endpoint ----------------------------

@app.post("/predict-pair")
async def predict_pair(

        left_eye_file: UploadFile = File(...),

        right_eye_file: UploadFile = File(...),

):
    """

    Run pathology detection on BOTH eyes.

    Apply bilateral logic after per-eye demotion logic.

    """

    try:

        # read both

        left_img = Image.open(io.BytesIO(await left_eye_file.read())).convert("RGB")

        right_img = Image.open(io.BytesIO(await right_eye_file.read())).convert("RGB")

        # per-eye probs

        left_probs = service.predict_from_pil(left_img)

        right_probs = service.predict_from_pil(right_img)

        # per-eye decision (already demotes weak pathology to No Suspect)

        left_dec = postprocess_probs(left_probs)

        right_dec = postprocess_probs(right_probs)

        # bilateral rule:

        # if either is still pathological after demotion → call pathological

        pathological_set = {

            "Minor Pathological",

            "Obvious Pathological",

            "Other Macular Pathology",

        }

        if (

                left_dec["pred_name"] in pathological_set

                or right_dec["pred_name"] in pathological_set

        ):

            final_label = "Obvious Pathological"

            final_ads = 20

            rule = "PAIR:PATHOLOGICAL_DETECTED"

        else:

            # otherwise everything is "No Suspect or Negative"

            final_label = "No Suspect or Negative"

            final_ads = 18

            rule = "PAIR:DEFAULT_NO"

        return {

            "left": {

                "file_name": left_eye_file.filename,

                "decision": left_dec,

                "probabilities": probs_to_confmap(left_probs),

            },

            "right": {

                "file_name": right_eye_file.filename,

                "decision": right_dec,

                "probabilities": probs_to_confmap(right_probs),

            },

            "pair_summary": {

                "before_override": [left_dec["pred_name"], right_dec["pred_name"]],

                "after_override": final_label,

                "ads_id": final_ads,

                "rule": rule,

            },

            "model_info": {

                "model": MODEL_NAME,

                "img_size": IMG_SIZE,

                "device": DEVICE,

            },

        }

    except Exception as e:

        logger.exception("predict-pair failed: %s", e)

        raise HTTPException(

            status_code=500,

            detail=f"predict-pair failed: {str(e)}"

        )


# ----------------------- Local dev runner -------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("pathology_service:app", host="0.0.0.0", port=8006, reload=True)
