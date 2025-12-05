import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFile
from pydantic import BaseModel
from torchvision import models, transforms

# allow truncated images just in case
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------- Settings -----------------
# You can override these with environment variables or change the defaults here.
MODEL_PATH = Path(os.getenv("AMD_MODEL_PATH", r"C:\AI_Images\amd_runs\run_ads\amd_best_model.pt"))
CLASS_TO_IDX_PATH = Path(os.getenv("AMD_CLASS_TO_IDX", r"C:\AI_Images\amd_runs\run_ads\class_to_idx.json"))
CLASS_META_PATH = Path(os.getenv("AMD_CLASS_META",    r"C:\AI_Images\amd_runs\run_ads\class_meta.json"))
MODEL_NAME = os.getenv("AMD_MODEL_NAME", "")           # if empty we’ll read it from the checkpoint
IMG_SIZE = int(os.getenv("AMD_IMG_SIZE", "224"))       # if 0/empty we’ll read from checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optional CORS
ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")

# ----------------- App -----------------
app = FastAPI(title="AMD Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Schemas -----------------
class ProbItem(BaseModel):
    index: int
    label: str
    probability: float

class PredictionResponse(BaseModel):
    top_label: str
    top_index: int
    top_probability: float
    probs: List[ProbItem]
    ads: Dict[str, Optional[str]]  # ADS_ID, ADS_AD_ID, ADS_SEVERITY
    model_name: str
    img_size: int

# ----------------- Model utils -----------------
def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = (name or "").lower()
    if name in ("", "resnet18"):
        m = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == "resnet50":
        m = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == "inception_v3":
        m = models.inception_v3(weights=None if not pretrained else models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported model name: {name}")

def load_class_maps(class_to_idx_path: Path, class_meta_path: Path):
    if not class_to_idx_path.exists():
        raise FileNotFoundError(f"Missing class_to_idx.json at {class_to_idx_path}")
    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    # invert mapping
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    ads_meta = {}
    if class_meta_path.exists():
        with open(class_meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # keys might be strings; normalize to int keys
        ads_meta = {int(k): v for k, v in raw.items()}
    return class_to_idx, idx_to_class, ads_meta

def load_model(model_path: Path):
    # Determine if this is a checkpoint dict or raw state_dict
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model_name = ckpt.get("model_name", MODEL_NAME or "resnet18")
        img_size = int(ckpt.get("img_size", IMG_SIZE or 224))
        # we need class_to_idx from file to know num_classes
        class_to_idx, idx_to_class, ads_meta = load_class_maps(CLASS_TO_IDX_PATH, CLASS_META_PATH)
        num_classes = len(class_to_idx)
        model = build_model(model_name, num_classes=num_classes, pretrained=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        return model, model_name, img_size, class_to_idx, idx_to_class, ads_meta
    else:
        # state_dict only
        class_to_idx, idx_to_class, ads_meta = load_class_maps(CLASS_TO_IDX_PATH, CLASS_META_PATH)
        num_classes = len(class_to_idx)
        model_name = MODEL_NAME or "resnet18"
        img_size = IMG_SIZE or 224
        model = build_model(model_name, num_classes=num_classes, pretrained=False)
        model.load_state_dict(ckpt, strict=True)
        return model, model_name, img_size, class_to_idx, idx_to_class, ads_meta

# Load once at startup
MODEL, MODEL_NAME_EFF, IMG_SIZE_EFF, CLASS_TO_IDX, IDX_TO_CLASS, ADS_META = load_model(MODEL_PATH)
MODEL.eval()
MODEL.to(DEVICE)

# transforms (match training normalization)
PREPROC = transforms.Compose([
    transforms.Resize((IMG_SIZE_EFF, IMG_SIZE_EFF)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ----------------- Endpoints -----------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH),
        "model_name": MODEL_NAME_EFF,
        "img_size": IMG_SIZE_EFF,
        "num_classes": len(CLASS_TO_IDX),
        "labels": [IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))],
    }

@app.post("/predict-macular-degeneration", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    x = PREPROC(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_index = int(probs.argmax())
    top_label = IDX_TO_CLASS[top_index]
    top_probability = float(probs[top_index])

    probs_list = [
        ProbItem(index=i, label=IDX_TO_CLASS[i], probability=float(probs[i]))
        for i in range(len(probs))
    ]

    ads = ADS_META.get(top_index, {})
    # ensure consistent string keys in response
    ads_out = {
        "ADS_ID": str(ads.get("ADS_ID")) if "ADS_ID" in ads else None,
        "ADS_AD_ID": str(ads.get("ADS_AD_ID")) if "ADS_AD_ID" in ads else None,
        "ADS_SEVERITY": ads.get("ADS_SEVERITY"),
    }

    return PredictionResponse(
        top_label=top_label,
        top_index=top_index,
        top_probability=top_probability,
        probs=probs_list,
        ads=ads_out,
        model_name=MODEL_NAME_EFF,
        img_size=IMG_SIZE_EFF,
    )
