from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import os

# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="Upload a fundus image and get DR severity (0–4) with healthy-favoring rules.",
    version="1.1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)

# =========================
# Model / classes
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = [
    "No DR",           # 0
    "Mild DR",         # 1
    "Moderate DR",     # 2
    "Severe DR",       # 3
    "Proliferative DR" # 4
]

# ADS/label mapping
class_map = {
    0: {"label": "No Suspect or Negative", "ads_id": 2},
    1: {"label": "Mild",                   "ads_id": 3},
    2: {"label": "Moderate",               "ads_id": 4},
    3: {"label": "Severe",                 "ads_id": 5},
    4: {"label": "Proliferative",          "ads_id": 6}
}

# ----- Thresholds (ladder) -----
THRESHOLDS = {
    4: 0.80,  # PDR
    3: 0.85,  # Severe
    2: 0.77,  # Moderate
    1: 0.80,  # Mild
    0: 0.00   # Healthy fallback
}

# ----- Healthy dominance (global force rule) -----
HEALTHY_FORCE_MIN   = 0.60  # Healthy prob must be at least this
HEALTHY_FORCE_MARGIN = 0.25 # and within this of the top class prob

# ----- Favor Healthy over PDR/Mild unless clearly ahead -----
PDR_MIN_PROB       = 0.92
PDR_MARGIN_AHEAD   = 0.20
MILD_MIN_PROB      = 0.75
MILD_MARGIN_AHEAD  = 0.15

# ----- Referable guard (Moderate/Severe/PDR must have evidence) -----
REFERABLE_CLASSES   = {2, 3, 4}
REFERABLE_MIN_PROB  = 0.60      # max(prob among 2/3/4) must beat this
REFERABLE_SUM_MIN   = 0.70      # combined prob for 2+3+4 must beat this

# =========================
# Model setup
# =========================
def build_model(num_classes=5):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    nn.init.kaiming_normal_(m.fc.weight, mode='fan_out', nonlinearity='relu')
    nn.init.constant_(m.fc.bias, 0)
    m.to(device)
    m.eval()
    return m

def load_weights_if_any(model):
    path = os.environ.get("DR_WEIGHTS", r"C:\AI_Images\diabetic-retinopathy-detection\model\resnet50_diabetic.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"✅ Loaded weights: {path}")

model = build_model()
load_weights_if_any(model)

# =========================
# Preprocess
# =========================
img_size = 224
preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# =========================
# Helpers (selection logic)
# =========================
def _softmax_probs(img: Image.Image) -> torch.Tensor:
    """Return 1-D softmax probs over 5 classes."""
    x = preprocess(img.convert("RGB")).unsqueeze(0).to(device)  # [1,3,H,W]
    with torch.no_grad():
        logits = model(x)                                       # [1,5]
        probs  = F.softmax(logits, dim=1).squeeze(0)           # [5]
    return probs

def _healthy_dominance(probs: torch.Tensor) -> bool:
    """Force Healthy if it's strong and close to the winner."""
    p = [float(probs[i].item()) for i in range(5)]
    p0 = p[0]
    top = max(p)
    return (p0 >= HEALTHY_FORCE_MIN) and ((top - p0) <= HEALTHY_FORCE_MARGIN)

def _ladder_choice(probs: torch.Tensor) -> int:
    """Pick highest-severity class that meets its threshold (4→3→2→1→0)."""
    for cls in (4, 3, 2, 1, 0):
        if float(probs[cls].item()) >= THRESHOLDS[cls]:
            return cls
    return 0

def _favor_healthy_demotions(probs: torch.Tensor, class_id: int) -> int:
    """Demote PDR/Mild to Healthy if not clearly ahead."""
    p = [float(probs[i].item()) for i in range(5)]
    # PDR → Healthy unless clearly ahead
    if class_id == 4:
        if (p[4] < PDR_MIN_PROB) or ((p[4] - p[0]) < PDR_MARGIN_AHEAD):
            return 0
    # Mild → Healthy unless clearly ahead
    if class_id == 1:
        if (p[1] < MILD_MIN_PROB) or ((p[1] - p[0]) < MILD_MARGIN_AHEAD):
            return 0
    return class_id

def _referable_guard(probs: torch.Tensor, class_id: int) -> int:
    """If choosing referable class, require combined evidence; else demote (prefer Mild then Healthy)."""
    p = [float(probs[i].item()) for i in range(5)]
    if class_id in REFERABLE_CLASSES:
        referable_sum = p[2] + p[3] + p[4]
        top_ref = max(p[2], p[3], p[4])
        if (referable_sum < REFERABLE_SUM_MIN) or (top_ref < REFERABLE_MIN_PROB):
            # Prefer Mild if it passes its threshold; otherwise Healthy
            if p[1] >= THRESHOLDS[1]:
                return 1
            return 0
    return class_id

def _finalize_prediction(probs: torch.Tensor) -> dict:
    """
    Apply: Healthy dominance -> Ladder -> Healthy demotions -> Referable guard.
    Return dict with class/ads + probabilities.
    """
    # 0) Healthy dominance (global override)
    if _healthy_dominance(probs):
        class_id = 0
    else:
        # 1) Ladder
        class_id = _ladder_choice(probs)
        # 2) Demotions (favor Healthy)
        class_id = _favor_healthy_demotions(probs, class_id)
        # 3) Referable guard
        class_id = _referable_guard(probs, class_id)

    confidence = float(probs[class_id].item())
    class_name = CLASS_NAMES[class_id]
    ads_id     = class_map[class_id]["ads_id"]

    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "ads_id": ads_id,
        "referable": class_id in REFERABLE_CLASSES,
        "all_probabilities": {CLASS_NAMES[i]: float(probs[i].item()) for i in range(5)}
    }

# =========================
# Schemas
# =========================
class Prediction(BaseModel):
    class_id: int
    confidence: float

# =========================
# Endpoints
# =========================
@app.post("/predict", response_model=Prediction)
def predict(file: UploadFile = File(...), referral_id: Optional[str] = None):
    """
    Single-image prediction endpoint.
    Post with: files={'file': (...)} to /predict?referral_id=...
    Always returns ads_id + probabilities (unless 4xx/5xx).
    """
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Invalid image type. Only JPEG and PNG are supported.")
    try:
        contents = file.file.read()
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image file.")

    try:
        probs  = _softmax_probs(img)
        result = _finalize_prediction(probs)
        if referral_id is not None:
            result["referral_id"] = referral_id
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_detailed")
def predict_detailed(file: UploadFile = File(...), referral_id: Optional[str] = None):
    """Same as /predict but without response_model; returns the same JSON fields."""
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Invalid image type. Only JPEG and PNG are supported.")
    try:
        contents = file.file.read()
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image file.")

    try:
        probs  = _softmax_probs(img)
        result = _finalize_prediction(probs)
        if referral_id is not None:
            result["referral_id"] = referral_id
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detailed prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "framework": "PyTorch",
        "gpu": torch.cuda.is_available(),
        "device": str(device)
    }

@app.get("/")
def root():
    return {
        "message": "Diabetic Retinopathy Detection API",
        "model": "ResNet50 (PyTorch)",
        "classes": CLASS_NAMES,
        "class_mapping": class_map,
        "thresholds": THRESHOLDS,
        "healthy_dominance": {"min": HEALTHY_FORCE_MIN, "margin": HEALTHY_FORCE_MARGIN},
        "pdr_bias": {"min_prob": PDR_MIN_PROB, "margin_ahead": PDR_MARGIN_AHEAD},
        "mild_bias": {"min_prob": MILD_MIN_PROB, "margin_ahead": MILD_MARGIN_AHEAD},
        "referable_requirements": {"sum_min": REFERABLE_SUM_MIN, "max_min": REFERABLE_MIN_PROB},
        "usage": "POST /predict with fundus image file"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
