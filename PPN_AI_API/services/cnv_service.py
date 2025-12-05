import os
import io
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image

# ---------------------------
# Config
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MANIFEST_PATH = Path(os.getenv("MANIFEST_PATH", r"C:\AI_Images\CNV\cnv_kfold_out\ensemble.txt"))

app = FastAPI(title="CNV Ensemble Classifier", version="1.0.0")

MODELS: List[nn.Module] = []
CLASSES: List[str] = []
IMG_SIZE: int = 384

# ---------------------------
# Utils
# ---------------------------
def _load_single_model(ckpt_path: Path) -> nn.Module:
    data = torch.load(str(ckpt_path), map_location=DEVICE)
    classes = data["classes"]
    img_size = data["img_size"]

    global CLASSES, IMG_SIZE
    if not CLASSES:
        CLASSES = classes
        IMG_SIZE = img_size
    else:
        if classes != CLASSES:
            raise RuntimeError(f"Classes mismatch across folds. {classes} != {CLASSES}")
        if img_size != IMG_SIZE:
            raise RuntimeError(f"IMG_SIZE mismatch across folds. {img_size} != {IMG_SIZE}")

    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(data["model_state"])
    m.eval().to(DEVICE)
    return m

def _load_ensemble():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    paths = [Path(p.strip()) for p in MANIFEST_PATH.read_text().strip().splitlines() if p.strip()]
    if not paths:
        raise RuntimeError("Manifest is empty.")
    models_list = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        models_list.append(_load_single_model(p))
    return models_list

def _build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

def _predict_pil(img: Image.Image) -> Dict[str, Any]:
    tfm = _build_transform()
    x = tfm(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits_sum = None
        for m in MODELS:
            logits = m(x)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits_mean = logits_sum / len(MODELS)
        probs = torch.softmax(logits_mean, dim=1).squeeze(0).tolist()

    cnv_prob = probs[CLASSES.index("CNV")]
    normal_prob = probs[CLASSES.index("normal")]

    # Apply threshold
    if cnv_prob > 0.80:
        pred_label = "CNV"
        ads_id = 23
    else:
        pred_label = "normal"
        ads_id = 22

    return {
        "classes": CLASSES,
        "probabilities": {
            "CNV": float(cnv_prob),
            "normal": float(normal_prob)
        },
        "prediction": pred_label,
        "ads_id": ads_id
    }


# ---------------------------
# Schemas
# ---------------------------
class PredictResponse(BaseModel):
    classes: List[str]
    probabilities: Dict[str, float]
    prediction: str
    ads_id: int   # ← add this

class BatchPredictResponseItem(PredictResponse):
    filename: str

# ---------------------------
# Startup
# ---------------------------
@app.on_event("startup")
def _startup():
    global MODELS
    torch.backends.cudnn.benchmark = True
    MODELS = _load_ensemble()

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "num_models": len(MODELS),
        "img_size": IMG_SIZE,
        "classes": CLASSES
    }

@app.get("/labels")
def labels():
    return {"classes": CLASSES}

@app.post("/predict-cnv")
async def predict_cnv(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = _predict_pil(img)

    # Adapt result into condition-keyed format
    response = {
        "cnv": {
            "ads_id": result["ads_id"],
            "confidence_scores": result["probabilities"],
            "prediction": result["prediction"]
        }
    }

    return response

@app.post("/predict-cnv-batch", response_model=List[BatchPredictResponseItem])
async def predict_cnv_batch(files: List[UploadFile] = File(...)):
    results: List[BatchPredictResponseItem] = []
    for f in files:
        try:
            img = _read_image(await f.read())
            out = _predict_pil(img)
            results.append(BatchPredictResponseItem(filename=f.filename, **out))
        except HTTPException as he:
            # Still return a record for this file with an error marker
            results.append(BatchPredictResponseItem(
                filename=f.filename,
                classes=CLASSES or [],
                probabilities={},
                prediction=f"error: {he.detail}"
            ))
    return results
