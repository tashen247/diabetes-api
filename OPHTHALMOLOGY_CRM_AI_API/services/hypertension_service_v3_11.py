# hypertension_service_v3_7_hotfix.py
import os, io, json, inspect
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, ImageFile

ADS = {
    "NoSuspectOrNegative": 7,
    "DecreasedRetinalArteryElasticity": 8,
    "HypertensiveRetinopathyGrade1or2": 9,
    "HypertensiveRetinopathyGrade3or4": 10,
}

ImageFile.LOAD_TRUNCATED_IMAGES = True

SERVICE_VERSION = "3.7.0-hotfix"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = int(os.getenv("HTN_IMG_SIZE", "512"))

# --- env flags ---
BYPASS_RULES = bool(int(os.getenv("BYPASS_RULES", "0")))  # set to 1 to force argmax
TEMPERATURE = float(os.getenv("HTN_TEMP", "1.0"))         # optional sharpening

# --- weights (use your existing v3 weights if you want) ---
WEIGHTS_PATH = Path(os.getenv(
    "HTN_MC_WEIGHTS",
    r"C:\Source_Controls\development_sanele\weights\hypertension_multiclass_v3_11_best.pth"
))

ADS_ID = {
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



# ---- decision thresholds (tune via env vars) ----
HR34_MIN = float(os.getenv("HTN_HR34_MIN", "0.80"))   # min prob to call HR3-4
HR12_MIN = float(os.getenv("HTN_HR12_MIN", "0.60"))   # min prob to call HR1-2
NO_MIN   = float(os.getenv("HTN_NO_MIN",   "0.18"))   # min prob to call No


to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def choose_with_thresholds(prob_map: dict) -> dict:
    """Decide label using calibrated thresholds instead of raw argmax."""
    p_no   = float(prob_map.get("NoSuspectOrNegative", 0.0))
    p_drae = float(prob_map.get("DecreasedRetinalArteryElasticity", 0.0))
    p_hr12 = float(prob_map.get("HypertensiveRetinopathyGrade1or2", 0.0))
    p_hr34 = float(prob_map.get("HypertensiveRetinopathyGrade3or4", 0.0))

    # 1) strong positives first
    if p_hr34 >= HR34_MIN:
        lab = "HypertensiveRetinopathyGrade3or4"
    elif p_hr12 >= HR12_MIN:
        lab = "HypertensiveRetinopathyGrade1or2"
    # 2) otherwise allow "No" if it clears a safety floor
    elif p_no >= NO_MIN:
        lab = "NoSuspectOrNegative"
    # 3) otherwise: fallback to argmax
    else:
        lab = max(prob_map, key=prob_map.get)

    return {"label": lab, "ads_id": ADS[lab], "prob": float(prob_map[lab])}

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

def _argmax_from_probs(prob_map: Dict[str, float]) -> Dict[str, object]:
    canon = {CANON.get(k, k): float(v) for k, v in prob_map.items()}
    label = max(canon, key=canon.get)
    return {"label": label, "ads_id": ADS[label], "prob": canon[label]}

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
            return {"after_override": lab.replace("HypertensiveRetinopathy", "Hypertensive Retinopathy"),
                    "ads_id": ADS[lab], "rule": "PAIR:PRIORITY"}
    return {"after_override": "No Suspect or Negative", "ads_id": ADS["NoSuspectOrNegative"], "rule": "PAIR:FALLBACK"}

# ---- Load model + classes ----
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
if isinstance(CK, dict) and any(k in CK for k in ("model","state_dict")):
    state = CK.get("model") or CK.get("state_dict")
    if hasattr(state, "state_dict"): state = state.state_dict()
elif isinstance(CK, dict):
    state = CK
else:
    state = CK.state_dict()
if any(k.startswith("module.") for k in state):
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
MODEL.load_state_dict(state, strict=False)
MODEL.eval().to(DEVICE)

app = FastAPI(title="Hypertension v3.7.0 (hotfix)", version=SERVICE_VERSION)

# add (or replace) this helper
@torch.inference_mode()
def _infer_probs(x: torch.Tensor) -> dict:
    # logits with optional temperature
    logits = MODEL(x) / max(TEMPERATURE, 1e-4)
    # detach before numpy()
    probs = torch.softmax(logits, dim=1)[0].detach().float().cpu().numpy()
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


def _load_img_bytes(b: bytes) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    return to_tensor(img).unsqueeze(0).to(DEVICE)

@app.get("/healthz")
def healthz():
    return {
        "device": DEVICE,
        "img_size": IMG_SIZE,
        "weights": str(WEIGHTS_PATH),
        "classes": CLASSES,
        "service_version": SERVICE_VERSION,
        "bypass_rules": BYPASS_RULES,
        "temperature": TEMPERATURE,
    }

@app.post("/predict-hypertension")
async def predict_single(file: UploadFile = File(...)):
    x = _load_img_bytes(await file.read())
    probs = _infer_probs(x)

    if BYPASS_RULES:
        best = _argmax_from_probs(probs)
        return {"file": file.filename, "probabilities": probs,
                "best_ads_overridden": {"ads_id": best["ads_id"], "label": best["label"]},
                "debug": {"logic_path": "BYPASS_RULES_ARGMAX"}}

    # Legacy rules (kept minimal)
    # If you really need them, implement here; otherwise default to argmax.
    #best = _argmax_from_probs(probs)
    # best = choose_with_thresholds(probs)
    # return {"file": file.filename, "probabilities": probs,
    #         "best_ads_overridden": {"ads_id": best["ads_id"], "label": best["label"]},
    #         "debug": {"logic_path": "ARGMAX_DEFAULT"}}
    best = choose_with_thresholds(probs)
    return {
        "file": file.filename,
        "probabilities": probs,
        "best_ads_overridden": {"ads_id": best["ads_id"], "label": best["label"]},
        "debug": {
            "logic_path": "THRESHOLDS_DECISION",
            "thresholds": {"HR34_MIN": HR34_MIN, "HR12_MIN": HR12_MIN, "NO_MIN": NO_MIN}
        }
    }


@app.post("/predict-hypertension/pair")
async def predict_pair(left_eye_file: UploadFile = File(...), right_eye_file: UploadFile = File(...)):
    l = _load_img_bytes(await left_eye_file.read())
    r = _load_img_bytes(await right_eye_file.read())
    l_probs = _infer_probs(l)
    r_probs = _infer_probs(r)

    if BYPASS_RULES:
        # l_best = _argmax_from_probs(l_probs)
        # r_best = _argmax_from_probs(r_probs)
        l_best = choose_with_thresholds(l_probs)
        r_best = choose_with_thresholds(r_probs)
        pair = _pair_priority(l_best["label"], r_best["label"])
        return {
            "left":  {"file_name": left_eye_file.filename,  "probabilities": l_probs,
                      "best_ads_overridden": {"ads_id": l_best["ads_id"], "label": l_best["label"]},
                      "debug": {"logic_path": "BYPASS_RULES_ARGMAX"}},
            "right": {"file_name": right_eye_file.filename, "probabilities": r_probs,
                      "best_ads_overridden": {"ads_id": r_best["ads_id"], "label": r_best["label"]},
                      "debug": {"logic_path": "BYPASS_RULES_ARGMAX"}},
            "pair_debug": {"before_override": [l_best["label"].replace("SuspectOr"," Suspect or "),
                                               r_best["label"].replace("SuspectOr"," Suspect or ")],
                           **pair},
            "model": {"version": "DRAE_TUNED_CALIBRATABLE", "service_version": SERVICE_VERSION,
                      "device": DEVICE, "thresholds": {"bypass_rules": True}}
        }

    # Legacy path → default to argmax anyway (safe fallback)
    # l_best = _argmax_from_probs(l_probs)
    # r_best = _argmax_from_probs(r_probs)
    # pair = _pair_priority(l_best["label"], r_best["label"])
    # return {
    #     "left":  {"file_name": left_eye_file.filename,  "probabilities": l_probs,
    #               "best_ads_overridden": {"ads_id": l_best["ads_id"], "label": l_best["label"]},
    #               "debug": {"logic_path": "ARGMAX_DEFAULT"}},
    #     "right": {"file_name": right_eye_file.filename, "probabilities": r_probs,
    #               "best_ads_overridden": {"ads_id": r_best["ads_id"], "label": r_best["label"]},
    #               "debug": {"logic_path": "ARGMAX_DEFAULT"}},
    #     "pair_debug": {"before_override": [l_best["label"].replace("SuspectOr"," Suspect or "),
    #                                        r_best["label"].replace("SuspectOr"," Suspect or ")],
    #                    **pair},
    #     "model": {"version": "DRAE_TUNED_CALIBRATABLE", "service_version": SERVICE_VERSION,
    #               "device": DEVICE, "thresholds": {"bypass_rules": False}}
    # }
    # Thresholded per-eye decisions
    l_best = choose_with_thresholds(l_probs)
    r_best = choose_with_thresholds(r_probs)
    pair = _pair_priority(l_best["label"], r_best["label"])

    return {
        "left":  {"file_name": left_eye_file.filename,  "probabilities": l_probs,
                  "best_ads_overridden": {"ads_id": l_best["ads_id"], "label": l_best["label"]},
                  "debug": {"logic_path": "THRESHOLDS_DECISION",
                            "thresholds": {"HR34_MIN": HR34_MIN, "HR12_MIN": HR12_MIN, "NO_MIN": NO_MIN}}},
        "right": {"file_name": right_eye_file.filename, "probabilities": r_probs,
                  "best_ads_overridden": {"ads_id": r_best["ads_id"], "label": r_best["label"]},
                  "debug": {"logic_path": "THRESHOLDS_DECISION",
                            "thresholds": {"HR34_MIN": HR34_MIN, "HR12_MIN": HR12_MIN, "NO_MIN": NO_MIN}}},
        "pair_debug": {"before_override": [
                           l_best["label"].replace("SuspectOr"," Suspect or "),
                           r_best["label"].replace("SuspectOr"," Suspect or ")
                       ], **pair},
        "model": {"version": "DRAE_TUNED_CALIBRATABLE", "service_version": SERVICE_VERSION,
                  "device": DEVICE,
                  "thresholds": {"bypass_rules": False,
                                 "HR34_MIN": HR34_MIN, "HR12_MIN": HR12_MIN, "NO_MIN": NO_MIN}}
}

