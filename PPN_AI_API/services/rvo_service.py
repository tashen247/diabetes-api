# rvo_api.py — FastAPI prediction service for Retinal Vein Occlusion (RVO vs Normal)
# Run:
#   uvicorn rvo_api:app --host 0.0.0.0 --port 8012 --reload
#
# Env knobs (override as needed):
#   RVO_MODEL_PATH=./rvo_best_model.pth     # or best_model.pth
#   RVO_BACKBONE=tf_efficientnet_b0_ns      # must match training backbone
#   RVO_IMAGE_SIZE=512
#   RVO_THRESHOLD=0.5                        # decision threshold on p(RVO)
#   RVO_UNCERTAIN_BAND=0.04                  # +/- band around threshold
#   RVO_QC_BLUR_MIN=40                       # Laplacian variance min
#   RVO_QC_DARK_MEAN=15                      # grayscale mean min

import io, os, time, json
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageOps
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------------------- Config -------------------------
IMG_SIZE        = int(os.getenv("RVO_IMAGE_SIZE", "512"))
THRESHOLD       = float(os.getenv("RVO_THRESHOLD", "0.7"))
UNCERTAIN_BAND  = float(os.getenv("RVO_UNCERTAIN_BAND", "0.04"))
BLUR_MIN        = float(os.getenv("RVO_QC_BLUR_MIN", "40"))
DARK_MEAN       = float(os.getenv("RVO_QC_DARK_MEAN", "15"))
MODEL_PATH      = os.getenv("RVO_MODEL_PATH", r"C:\source_controls\PPN_AI_API\weights\rvo\rvo_best_model.pth")
MODEL_BACKBONE  = os.getenv("RVO_BACKBONE", "tf_efficientnet_b0_ns")

CLASS_NAMES     = ["Normal", "RVO"]  # index 0=Normal, 1=RVO
ADS_ID_MAP      = {"Normal": 24, "RVO": 25}  # <-- add this

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- Model -------------------------
try:
    import timm
except Exception as e:
    raise RuntimeError("timm is required: pip install timm") from e

class RVOModel(nn.Module):
    """Binary head returning a single logit (match trainer)."""
    def __init__(self, backbone: str = MODEL_BACKBONE):
        super().__init__()
        self.net = timm.create_model(backbone, pretrained=False, num_classes=1)

    def forward(self, x):
        return self.net(x).squeeze(1)  # [B] single logit

def load_model(path: str) -> nn.Module:
    model = RVOModel(MODEL_BACKBONE)
    ckpt = torch.load(path, map_location="cpu")

    # Flexible loader: raw state_dict or dict with 'state_dict'
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Strip common prefixes
    new_state = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)
    model.to(device).eval()
    return model

model = None
load_error = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print(f"[RVO] Loaded {MODEL_PATH} on {device}")
except Exception as e:
    load_error = repr(e)
    print(f"[RVO] Load failed: {load_error}")

# ------------------------- Preprocessing & QC -------------------------
def _read_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return ImageOps.exif_transpose(img)

def _transform() -> T.Compose:
    return T.Compose([
        T.Resize(int(IMG_SIZE * 1.05)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def _qc_metrics(pil: Image.Image) -> Dict[str, float]:
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "blur_var_laplacian": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean_brightness": float(gray.mean()),
    }

def _qc_flags(m: Dict[str, float]) -> Dict[str, bool]:
    return {
        "too_blurry": m["blur_var_laplacian"] < BLUR_MIN,
        "too_dark": m["mean_brightness"] < DARK_MEAN,
    }

# ------------------------- Inference -------------------------
@torch.inference_mode()
def infer_rvo(pil: Image.Image) -> Dict[str, Any]:
    x = _transform()(pil).unsqueeze(0).to(device)
    logit = model(x)  # [1]
    p_rvo = float(torch.sigmoid(logit)[0].item())
    p_normal = 1.0 - p_rvo

    lower = THRESHOLD - UNCERTAIN_BAND
    upper = THRESHOLD + UNCERTAIN_BAND
    uncertain = lower <= p_rvo <= upper

    label_idx = 1 if p_rvo >= THRESHOLD else 0
    label = CLASS_NAMES[label_idx]
    return {
        "prediction": label,
        "probabilities": {"Normal": round(p_normal, 6), "RVO": round(p_rvo, 6)},
        "threshold": THRESHOLD,
        "uncertain": bool(uncertain),
        "uncertainty_band": [round(lower, 4), round(upper, 4)],
        "ads_id": ADS_ID_MAP[label],   # <-- add this
    }


# ------------------------- API -------------------------
class Health(BaseModel):
    status: str
    device: str
    model_loaded: bool
    error: Optional[str] = None
    config: Dict[str, Any]

app = FastAPI(title="RVO Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health", response_model=Health)
def health():
    return Health(
        status="ok" if (model is not None and load_error is None) else "error",
        device=str(device),
        model_loaded=(model is not None and load_error is None),
        error=load_error,
        config={
            "IMG_SIZE": IMG_SIZE,
            "THRESHOLD": THRESHOLD,
            "UNCERTAIN_BAND": UNCERTAIN_BAND,
            "BLUR_MIN": BLUR_MIN,
            "DARK_MEAN": DARK_MEAN,
            "MODEL_PATH": MODEL_PATH,
            "BACKBONE": MODEL_BACKBONE,
        }
    )

@app.post("/predict-rvo")
async def predict_rvo(
    image: UploadFile = File(..., description="Fundus image (JPG/PNG)"),
    referral_id: Optional[str] = Query(None),
    side: Optional[str] = Query(None, description="'left' or 'right'"),
    filename: Optional[str] = Query(None)
):
    if model is None or load_error:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {image.content_type}")

    try:
        raw = await image.read()
        pil = _read_image(raw)
        qc = _qc_metrics(pil)
        qc_flags = _qc_flags(qc)

        t0 = time.time()
        result = infer_rvo(pil)
        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "condition": "Retinal Vein Occlusion",
            **result,
            "qc": {
                "blur_var_laplacian": round(qc["blur_var_laplacian"], 2),
                "mean_brightness": round(qc["mean_brightness"], 2),
                "too_blurry": qc_flags["too_blurry"],
                "too_dark": qc_flags["too_dark"],
            },
            "meta": {
                "referral_id": referral_id,
                "side": side,
                "filename": filename or image.filename,
                "model_path": MODEL_PATH,
                "device": str(device),
                "elapsed_ms": elapsed_ms,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {repr(e)}")
