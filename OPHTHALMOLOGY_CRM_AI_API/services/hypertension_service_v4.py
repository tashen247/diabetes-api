# hypertension_service_v4.py
# ------------------------------------------------------------
# Hypertension microservice v4.0
# - Uses CNN probs
# - Applies logistic calibrator (trained vs AirDoc)
# - AirDoc-aligned rules: strong bias towards "No"
# - Thresholds for HR1-2 / HR3-4
# ------------------------------------------------------------
import os
import io
import json
import inspect
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, ImageFile

# ==============================
# Label / ADS maps
# ==============================
ADS = {
    "NoSuspectOrNegative": 7,
    "DecreasedRetinalArteryElasticity": 8,
    "HypertensiveRetinopathyGrade1or2": 9,
    "HypertensiveRetinopathyGrade3or4": 10,
}

CANON = {
    "No": "NoSuspectOrNegative",
    "No Suspect or Negative": "NoSuspectOrNegative",
    "NoSuspectOrNegative": "NoSuspectOrNegative",
    "Decreased Retinal Artery Elasticity": "DecreasedRetinalArteryElasticity",
    "DecreasedRetinalArteryElasticity": "DecreasedRetinalArteryElasticity",
    "Hypertensive Retinopathy Grade 1 or 2": "HypertensiveRetinopathyGrade1or2",
    "HypertensiveRetinopathyGrade1or2": "HypertensiveRetinopathyGrade1or2",
    "Hypertensive Retinopathy Grade 3 or 4": "HypertensiveRetinopathyGrade3or4",
    "HypertensiveRetinopathyGrade3or4": "HypertensiveRetinopathyGrade3or4",
}

ImageFile.LOAD_TRUNCATED_IMAGES = True

SERVICE_VERSION = "4.0.0-airdoc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = int(os.getenv("HTN_IMG_SIZE", "512"))

BYPASS_RULES = bool(int(os.getenv("BYPASS_RULES", "0")))
TEMPERATURE  = float(os.getenv("HTN_TEMP", "1.0"))

WEIGHTS_PATH = Path(os.getenv(
    "HTN_MC_WEIGHTS",
    r"C:\Source_Controls\development_sanele\weights\hypertension_multiclass_v3_11_best.pth"
))
CALIB_PATH = Path(os.getenv(
    "HTN_CALIB_JSON",
    r"C:\AI_Images\Hypertension\htn_calibrator.json"
))

# ---- thresholds (still tunable) ----
HR34_MIN    = float(os.getenv("HTN_HR34_MIN", "0.95"))
HR34_MARGIN = float(os.getenv("HTN_HR34_MARGIN", "0.15"))
HR12_MIN    = float(os.getenv("HTN_HR12_MIN", "0.60"))
NO_MIN      = float(os.getenv("HTN_NO_MIN",   "0.18"))

# AirDoc-style "No" bias
AIRDOC_NO_FORCE_MIN = float(os.getenv("AIRDOC_NO_FORCE_MIN", "0.86"))
AIRDOC_NO_SOFT_MIN  = float(os.getenv("AIRDOC_NO_SOFT_MIN", "0.40"))

# ==============================
# Transforms
# ==============================
to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ==============================
# Calibrator
# ==============================
class HypertensionCalibrator:
    """
    Multiclass logistic regression (4-way) trained vs AirDoc labels.
    JSON schema:
    {
      "classes": ["NoSuspectOrNegative", "DecreasedRetinalArteryElasticity", "HypertensiveRetinopathyGrade1or2", "HypertensiveRetinopathyGrade3or4"],
      "coef": [[... 4 ...], ... 4 rows ...],
      "intercept": [4 values]
    }
    """
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        self.classes: List[str] = d["classes"]
        self.coef = np.array(d["coef"], dtype=float)      # (4,4)
        self.intercept = np.array(d["intercept"], float)  # (4,)
        if self.coef.shape != (4, 4):
            raise ValueError(f"Expected coef (4,4), got {self.coef.shape}")
        if self.intercept.shape != (4,):
            raise ValueError(f"Expected intercept (4,), got {self.intercept.shape}")

    def predict_proba(self, prob_map: Dict[str, float]) -> Dict[str, float]:
        x = np.array([
            prob_map.get("NoSuspectOrNegative", 0.0),
            prob_map.get("DecreasedRetinalArteryElasticity", 0.0),
            prob_map.get("HypertensiveRetinopathyGrade1or2", 0.0),
            prob_map.get("HypertensiveRetinopathyGrade3or4", 0.0),
        ], dtype=float)
        logits = self.coef @ x + self.intercept
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        out = {self.classes[i]: float(p[i]) for i in range(4)}
        for k in ADS.keys():
            out.setdefault(k, 0.0)
        return out


CALIBRATOR = HypertensionCalibrator(CALIB_PATH)

# ==============================
# Model loading
# ==============================
def _supports_weights_only() -> bool:
    try:
        return "weights_only" in inspect.signature(torch.load).parameters
    except Exception:
        return False


def _load_ckpt(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing weights: {path}")
    if _supports_weights_only():
        try:
            from torch.serialization import add_safe_globals, safe_globals
            import numpy as _np
            add_safe_globals([_np.core.multiarray.scalar])
            with safe_globals([_np.core.multiarray.scalar]):
                return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(path, map_location="cpu")
    return torch.load(path, map_location="cpu")


def _build_model(nc: int):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, nc)
    return m


CK = _load_ckpt(WEIGHTS_PATH)
CLASSES: List[str] = CK.get("classes") if isinstance(CK, dict) else None
if not CLASSES:
    CLASSES = [
        "NoSuspectOrNegative",
        "DecreasedRetinalArteryElasticity",
        "HypertensiveRetinopathyGrade1or2",
        "HypertensiveRetinopathyGrade3or4",
    ]

MODEL = _build_model(len(CLASSES))
if isinstance(CK, dict) and any(k in CK for k in ("model", "state_dict")):
    state = CK.get("model") or CK.get("state_dict")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
elif isinstance(CK, dict):
    state = CK
else:
    state = CK.state_dict()

if any(k.startswith("module.") for k in state):
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

MODEL.load_state_dict(state, strict=False)
MODEL.eval().to(DEVICE)

# ==============================
# Core helpers
# ==============================
def _load_img_bytes(b: bytes) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    return to_tensor(img).unsqueeze(0).to(DEVICE)


@torch.inference_mode()
def _infer_raw_probs(x: torch.Tensor) -> Dict[str, float]:
    logits = MODEL(x) / max(TEMPERATURE, 1e-4)
    raw = torch.softmax(logits, dim=1)[0].detach().float().cpu().numpy()
    probs_raw = {CLASSES[i]: float(raw[i]) for i in range(len(CLASSES))}
    # Canonicalise
    probs: Dict[str, float] = {}
    for k, v in probs_raw.items():
        canon = CANON.get(k, k)
        probs[canon] = probs.get(canon, 0.0) + v
    for k in ADS.keys():
        probs.setdefault(k, 0.0)
    return probs


def _calibrated_probs(cnn_probs: Dict[str, float]) -> Dict[str, float]:
    return CALIBRATOR.predict_proba(cnn_probs)


def choose_airdoc_aligned(prob_map: Dict[str, float]) -> Dict[str, object]:
    """

    Decision engine on calibrated probs.

    """

    p_no = float(prob_map.get("NoSuspectOrNegative", 0.0))

    p_drae = float(prob_map.get("DecreasedRetinalArteryElasticity", 0.0))

    p_hr12 = float(prob_map.get("HypertensiveRetinopathyGrade1or2", 0.0))

    p_hr34 = float(prob_map.get("HypertensiveRetinopathyGrade3or4", 0.0))

    items = [

        ("NoSuspectOrNegative", p_no),

        ("DecreasedRetinalArteryElasticity", p_drae),

        ("HypertensiveRetinopathyGrade1or2", p_hr12),

        ("HypertensiveRetinopathyGrade3or4", p_hr34),

    ]

    items_sorted = sorted(items, key=lambda kv: kv[1], reverse=True)

    top_label, top_prob = items_sorted[0]

    second_prob = items_sorted[1][1] if len(items_sorted) > 1 else 0.0

    # ---- 0a) HARD NO: No is VERY high and DRAE is small ----

    # tighten this so we don't crush cases like 21641 left

    DRAE_HARD_MAX = 0.25  # if DRAE above this, don't allow HARD NO

    if (

            (top_label == "NoSuspectOrNegative")

            and (p_no >= AIRDOC_NO_FORCE_MIN)  # e.g. 0.75 or 0.8

            and (p_drae <= DRAE_HARD_MAX)

    ):
        lab = "NoSuspectOrNegative"

        return {"label": lab, "ads_id": ADS[lab], "prob": p_no, "engine": "AIRDOC_FORCE_NO"}

    # ---- 0b) BORDERLINE NO vs DRAE (when DRAE is top) ----

    NO_BORDER_MIN = 0.42

    NO_DRAE_MARGIN = 0.04

    if top_label == "DecreasedRetinalArteryElasticity":

        if (p_no >= NO_BORDER_MIN) and ((p_drae - p_no) <= NO_DRAE_MARGIN):
            lab = "NoSuspectOrNegative"

            return {"label": lab, "ads_id": ADS[lab], "prob": p_no, "engine": "AIRDOC_BORDER_NO"}

    # ---- 1) HR3-4 tier ----

    others = {

        "NoSuspectOrNegative": p_no,

        "DecreasedRetinalArteryElasticity": p_drae,

        "HypertensiveRetinopathyGrade1or2": p_hr12,

    }

    _, best_other = max(others.items(), key=lambda kv: kv[1])

    if (p_hr34 >= HR34_MIN) and ((p_hr34 - best_other) >= HR34_MARGIN):
        lab = "HypertensiveRetinopathyGrade3or4"

        return {"label": lab, "ads_id": ADS[lab], "prob": p_hr34, "engine": "THR_HR34"}

    # ---- 2) HR1-2 tier ----

    if p_hr12 >= HR12_MIN:
        lab = "HypertensiveRetinopathyGrade1or2"

        return {"label": lab, "ads_id": ADS[lab], "prob": p_hr12, "engine": "THR_HR12"}

    # ---- 3) DRAE rescue: No is top, but DRAE is non-trivially high ----

    DRAE_RESCUE_MIN = 0.32  # slightly below your 0.3289 case

    DRAE_RESCUE_GAPMAX = 0.30  # max allowed gap No - DRAE

    if top_label == "NoSuspectOrNegative":

        if (p_drae >= DRAE_RESCUE_MIN) and ((p_no - p_drae) <= DRAE_RESCUE_GAPMAX):
            lab = "DecreasedRetinalArteryElasticity"

            return {"label": lab, "ads_id": ADS[lab], "prob": p_drae, "engine": "DRAE_RESCUE_NO_TOP"}

    # ---- 4) NO floor *only if No is actually top* ----

    if (p_no >= NO_MIN) and (top_label == "NoSuspectOrNegative"):
        lab = "NoSuspectOrNegative"

        return {"label": lab, "ads_id": ADS[lab], "prob": p_no, "engine": "THR_NO"}

    # ---- 5) Fallback: pure argmax ----

    lab = top_label

    return {"label": lab, "ads_id": ADS[lab], "prob": float(prob_map[lab]), "engine": "ARGMAX_FALLBACK"}


def _pair_priority(lab_l: str, lab_r: str) -> Dict[str, object]:
    order = [
        "HypertensiveRetinopathyGrade3or4",
        "HypertensiveRetinopathyGrade1or2",
        "DecreasedRetinalArteryElasticity",
        "NoSuspectOrNegative",
    ]
    s = {lab_l, lab_r}
    for lab in order:
        if lab in s:
            return {
                "after_override": lab.replace("HypertensiveRetinopathy", "Hypertensive Retinopathy"),
                "ads_id": ADS[lab],
                "rule": "PAIR:PRIORITY",
            }
    return {
        "after_override": "No Suspect or Negative",
        "ads_id": ADS["NoSuspectOrNegative"],
        "rule": "PAIR:FALLBACK",
    }

# ==============================
# FastAPI app
# ==============================
app = FastAPI(title="Hypertension v4.0 (AirDoc aligned)", version=SERVICE_VERSION)


@app.get("/healthz")
def healthz():
    return {
        "service_version": SERVICE_VERSION,
        "device": DEVICE,
        "img_size": IMG_SIZE,
        "weights": str(WEIGHTS_PATH),
        "classes": CLASSES,
        "bypass_rules": BYPASS_RULES,
        "temperature": TEMPERATURE,
        "thresholds": {
            "HR34_MIN": HR34_MIN,
            "HR34_MARGIN": HR34_MARGIN,
            "HR12_MIN": HR12_MIN,
            "NO_MIN": NO_MIN,
        },
        "airdoc_alignment": {
            "AIRDOC_NO_FORCE_MIN": AIRDOC_NO_FORCE_MIN,
            "AIRDOC_NO_SOFT_MIN": AIRDOC_NO_SOFT_MIN,
        },
        "calibration_json": str(CALIB_PATH),
    }


@app.post("/predict-hypertension")
async def predict_single(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    x = _load_img_bytes(raw_bytes)

    cnn_probs = _infer_raw_probs(x)
    calib_probs = _calibrated_probs(cnn_probs)

    if BYPASS_RULES:
        best = max(calib_probs, key=calib_probs.get)
        best_dict = {"label": best, "ads_id": ADS[best], "prob": float(calib_probs[best]), "engine": "ARGMAX_ONLY"}
    else:
        best_dict = choose_airdoc_aligned(calib_probs)

    return {
        "file": file.filename,
        "probabilities": calib_probs,     # calibrated
        "probabilities_raw": cnn_probs,   # raw CNN
        "best_ads_overridden": {
            "ads_id": best_dict["ads_id"],
            "label": best_dict["label"],
        },
        "debug": {
            "engine": best_dict["engine"],
            "thresholds": {
                "HR34_MIN": HR34_MIN,
                "HR34_MARGIN": HR34_MARGIN,
                "HR12_MIN": HR12_MIN,
                "NO_MIN": NO_MIN,
            },
            "airdoc_rules": {
                "AIRDOC_NO_FORCE_MIN": AIRDOC_NO_FORCE_MIN,
                "AIRDOC_NO_SOFT_MIN": AIRDOC_NO_SOFT_MIN,
            },
        },
    }


@app.post("/predict-hypertension/pair")
async def predict_pair(
    left_eye_file: UploadFile = File(...),
    right_eye_file: UploadFile = File(...)
):
    l_bytes = await left_eye_file.read()
    r_bytes = await right_eye_file.read()

    l_tensor = _load_img_bytes(l_bytes)
    r_tensor = _load_img_bytes(r_bytes)

    l_cnn = _infer_raw_probs(l_tensor)
    r_cnn = _infer_raw_probs(r_tensor)

    l_probs = _calibrated_probs(l_cnn)
    r_probs = _calibrated_probs(r_cnn)

    if BYPASS_RULES:
        l_label = max(l_probs, key=l_probs.get)
        r_label = max(r_probs, key=r_probs.get)
        l_best = {"label": l_label, "ads_id": ADS[l_label], "prob": float(l_probs[l_label]), "engine": "ARGMAX_ONLY"}
        r_best = {"label": r_label, "ads_id": ADS[r_label], "prob": float(r_probs[r_label]), "engine": "ARGMAX_ONLY"}
    else:
        l_best = choose_airdoc_aligned(l_probs)
        r_best = choose_airdoc_aligned(r_probs)

    pair = _pair_priority(l_best["label"], r_best["label"])

    return {
        "left": {
            "file_name": left_eye_file.filename,
            "probabilities": l_probs,
            "probabilities_raw": l_cnn,
            "best_ads_overridden": {
                "ads_id": l_best["ads_id"],
                "label": l_best["label"],
            },
            "debug": {"engine": l_best["engine"]},
        },
        "right": {
            "file_name": right_eye_file.filename,
            "probabilities": r_probs,
            "probabilities_raw": r_cnn,
            "best_ads_overridden": {
                "ads_id": r_best["ads_id"],
                "label": r_best["label"],
            },
            "debug": {"engine": r_best["engine"]},
        },
        "pair_debug": {
            "before_override": [l_best["label"], r_best["label"]],
            **pair,
        },
        "model": {
            "version": "DRAE_TUNED_AIRDOC_V4",
            "service_version": SERVICE_VERSION,
            "device": DEVICE,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("HTN_PORT", "8003")))
