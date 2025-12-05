# myopia_service.py
# ------------------------------------------------------------
# Myopia Prediction Microservice (FastAPI)
# ------------------------------------------------------------

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import timm
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

# ============================== Paths / IDs ==============================
OVERLAY_DIR = Path(os.getenv("MYOPIA_OVERLAY_DIR", "overlays"))
OVERLAY_DIR.mkdir(exist_ok=True)

MODEL_PATH = os.getenv(
    "MYOPIA_MODEL_PATH",
    #r"C:\AI_Images\Myopia\outputs_bin\myopia_binary_best.pth",
    r"C:\AI_Images\Myopia\outputs_bin\myopia_binary_finetuned.pth",
)

ADS_MAP = {"Normal": 26, "Myopia": 42}

# ============================== Inference toggles ==============================
# These fall back to safe defaults; you DON'T need an env file.
APPLY_CLAHE = os.getenv("MYOPIA_CLAHE", "1") == "1"
USE_FOVCROP = os.getenv("MYOPIA_FOVCROP", "1") == "1"
USE_TTA = os.getenv("MYOPIA_TTA", "1") == "1"
USE_ENSEMBLE = os.getenv("MYOPIA_ENSEMBLE", "1") == "1"
USE_PREPROC = os.getenv("MYOPIA_PREPROCESS", "1") == "1"
SWAP_LOGITS = os.getenv("MYOPIA_SWAP_LOGITS", "0") == "1"
IMG_SIZE = int(os.getenv("MYOPIA_IMG_SIZE", "224"))

# ============================== QC thresholds ==============================
# Reasonable defaults; not strict (reduces false "bad quality")
BLUR_MIN = float(os.getenv("MYOPIA_QC_BLUR_MIN", "10"))
DARK_MEAN = float(os.getenv("MYOPIA_QC_DARK_MEAN", "15"))
BRIGHT_MEAN = float(os.getenv("MYOPIA_QC_BRIGHT_MEAN", "245"))

# ============================== Precision anti-FP knobs ==============================
# <<< If you need fewer false positives, tweak ONLY these 3 >>>
TUNED_THR = 0.50   # model p(Myopia) must exceed this
VETO_COMP = 0.30   # if morphology composite < this, flip to Normal
HI_CONF_CAP = 0.995  # ignore veto only if p(Myopia) is essentially certain

# Optional moderation band around the threshold
GRAY_LOW, GRAY_HIGH = 0.55, 0.80


# ============================== Utils: QC ==============================
def _qc_metrics_from_rgb(rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "blur_lapvar": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "illum_mean": float(gray.mean()),
        "illum_std": float(gray.std()),
    }


def quality_check_from_metrics(m: dict) -> dict:
    return {
        **m,
        "is_blurry": m["blur_lapvar"] < BLUR_MIN,
        "is_too_dark": m["illum_mean"] < DARK_MEAN,
        "is_too_bright": m["illum_mean"] > BRIGHT_MEAN,
        "quality_ok": not (
            m["blur_lapvar"] < BLUR_MIN
            or m["illum_mean"] < DARK_MEAN
            or m["illum_mean"] > BRIGHT_MEAN
        ),
    }


# ============================== OpenCV preprocessing ==============================
def preprocess_fundus_opencv(rgb: np.ndarray) -> np.ndarray:
    """Sharpen + CLAHE + circular mask. Input/Output are RGB."""
    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (512, 512))

        blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=15)
        bgr = cv2.addWeighted(bgr, 1.5, blur, -0.5, 0)

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        mask = np.zeros(bgr.shape[:2], np.uint8)
        h, w = mask.shape
        center, radius = (w // 2, h // 2), int(0.47 * min(w, h))
        cv2.circle(mask, center, radius, 255, -1)
        bgr = cv2.bitwise_and(bgr, bgr, mask=mask)

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return rgb


# ============================== Feature proxies ==============================
def compute_myopia_features(image_bytes: bytes) -> dict:
    """Heuristic morphology proxies (non-diagnostic)."""
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        # corrupt / unreadable
        return {
            "disc_aspect_ratio": 1.0,
            "macula_darkness": 0.0,
            "vessel_contrast": 0.0,
            "ppa_ring_strength": 0.0,
            "disc_temporal_offset": 0.0,
            "composite_myopia_score": 0.0,
        }

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]

    # Disc (bright zone)
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        disc = max(cnts, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(disc)
        disc_center = (x + ww / 2, y + hh / 2)
        disc_aspect = float(ww) / float(hh) if hh > 0 else 1.0
    else:
        disc_center, disc_aspect = (w / 2, h / 2), 1.0

    # Macula darkness (rough)
    roi = gray[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    macula_darkness = float(255 - np.mean(roi)) if roi.size else 0.0

    # Tessellation proxy
    vessel_map = cv2.equalizeHist(gray)
    vessel_contrast = float(np.std(vessel_map))

    # PPA proxy
    big = cv2.GaussianBlur(gray, (35, 35), 0)
    ring_strength = float(np.mean(cv2.subtract(big, gray)))

    # Disc offset
    img_center = (w / 2, h / 2)
    disc_offset = float(abs(disc_center[0] - img_center[0]) / (w / 2))

    # Composite
    comp = (
        0.3 * (macula_darkness / 255.0)
        + 0.25 * disc_offset
        + 0.25 * abs(1.0 - disc_aspect)
        + 0.2 * (ring_strength / 50.0)
    )

    return {
        "disc_aspect_ratio": round(disc_aspect, 3),
        "macula_darkness": round(macula_darkness, 2),
        "vessel_contrast": round(vessel_contrast, 2),
        "ppa_ring_strength": round(ring_strength, 2),
        "disc_temporal_offset": round(disc_offset, 3),
        "composite_myopia_score": round(comp, 3),
    }


# ============================== Geometry overlay ==============================
def analyze_fundus_geometry(
    image_bytes: bytes,
    filename: Optional[str],
) -> dict:
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    _, _, _, maxLoc = cv2.minMaxLoc(blur)
    disc_center = maxLoc

    patch = gray[
        max(0, disc_center[1] - 60) : min(h, disc_center[1] + 60),
        max(0, disc_center[0] - 60) : min(w, disc_center[0] + 60),
    ]
    _, mask = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(cnt)
        disc_aspect = round(ww / hh, 3) if hh > 0 else 1.0
    else:
        disc_aspect = 1.0

    # Macula opposite disc horizontally
    macula_center = (
        int(disc_center[0] + (w * 0.35 if disc_center[0] < w // 2 else -w * 0.35)),
        h // 2,
    )

    mac_roi = gray[
        max(0, macula_center[1] - 50) : min(h, macula_center[1] + 50),
        max(0, macula_center[0] - 50) : min(w, macula_center[0] + 50),
    ]
    macula_darkness = float(round(255 - np.mean(mac_roi), 2)) if mac_roi.size else 0.0
    vessel_contrast = float(round(np.std(cv2.equalizeHist(gray)), 2))
    disc_offset = float(round(abs(disc_center[0] - w / 2) / (w / 2), 3))

    vis = bgr.copy()
    cv2.circle(vis, disc_center, 25, (0, 255, 255), 2)
    cv2.circle(vis, macula_center, 25, (0, 128, 255), 2)
    cv2.line(vis, disc_center, macula_center, (255, 0, 0), 2)
    cv2.putText(
        vis,
        f"Disc Aspect: {disc_aspect}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        vis,
        f"Macula Dark: {macula_darkness}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        vis,
        f"Disc Offset: {disc_offset}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    stem = (
        Path(filename).stem
        if filename
        else f"overlay_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    overlay_path = OVERLAY_DIR / f"{stem}_overlay.jpg"
    cv2.imwrite(str(overlay_path), vis)

    return {
        "disc_aspect_ratio": disc_aspect,
        "macula_darkness": macula_darkness,
        "vessel_contrast": vessel_contrast,
        "disc_temporal_offset": disc_offset,
        "overlay_path": str(overlay_path),
    }


# ============================== Model wrapper ==============================
class MyopiaModelService:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.class_names = self._load_model(model_path)
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _build_model(self, arch: str = "efficientnet_b0", num_classes: int = 2) -> torch.nn.Module:
        return timm.create_model(arch, pretrained=False, num_classes=num_classes)

    def _load_model(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            classes = ckpt.get("classes", ["Normal", "Myopia"])
        else:
            sd = ckpt
            classes = ["Normal", "Myopia"]

        model = self._build_model(num_classes=len(classes))
        model.load_state_dict(sd, strict=False)
        model.to(self.device).eval()

        # Normalize class names
        classes = [
            "Normal"
            if str(c).lower().startswith("norm")
            else "Myopia"
            if str(c).lower().startswith("myop")
            else str(c)
            for c in classes
        ]
        if set(classes) != {"Normal", "Myopia"}:
            classes = ["Normal", "Myopia"]

        return model, classes

    def _read_rgb(self, image_bytes: bytes) -> np.ndarray:
        """Try PIL first, fallback to OpenCV."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.load()
            return np.array(img)
        except Exception:
            arr = np.frombuffer(image_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Could not decode image bytes.")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _apply_clahe(self, rgb: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    def _crop_fov(self, rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v = cv2.GaussianBlur(hsv[:, :, 2], (0, 0), 3)
        mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return rgb
        c = max(cnts, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        x, y, r = int(x), int(y), int(r)
        crop = rgb[max(0, y - r) : y + r, max(0, x - r) : x + r]
        return crop if crop.size else rgb

    def _tta_views(self, pil_img: Image.Image):
        if not USE_TTA:
            return [pil_img]
        return [
            pil_img,
            pil_img.transpose(Image.FLIP_LEFT_RIGHT),
            pil_img.transpose(Image.FLIP_TOP_BOTTOM),
            pil_img.transpose(Image.ROTATE_90),
            pil_img.transpose(Image.ROTATE_270),
        ]

    def _pipeline_probs(self, rgb: np.ndarray, use_clahe: bool = True, use_fov: bool = True) -> np.ndarray:
        img = rgb.copy()
        if use_clahe:
            img = self._apply_clahe(img)
        if use_fov:
            img = self._crop_fov(img)

        pil = Image.fromarray(img)
        probs = []
        for v in self._tta_views(pil):
            x = self.transform(v).unsqueeze(0).to(self.device)
            p = torch.softmax(self.model(x), dim=1).cpu().numpy()[0]
            probs.append(p)

        return np.mean(probs, axis=0)

    @torch.no_grad()
    def predict(self, image_bytes: bytes, filename: Optional[str] = None) -> dict:
        """
        End-to-end inference (conservative / precision-biased).
        """
        # -------- Decode to RGB --------
        rgb = self._read_rgb(image_bytes)

        # -------- Preprocess (OpenCV) and QC on RAW + PROCESSED --------
        rgb_proc = preprocess_fundus_opencv(rgb) if USE_PREPROC else rgb

        def _qc_from_metrics(m: dict) -> dict:
            return {
                **m,
                "is_blurry": m["blur_lapvar"] < BLUR_MIN,
                "is_too_dark": m["illum_mean"] < DARK_MEAN,
                "is_too_bright": m["illum_mean"] > BRIGHT_MEAN,
                "quality_ok": not (
                    m["blur_lapvar"] < BLUR_MIN
                    or m["illum_mean"] < DARK_MEAN
                    or m["illum_mean"] > BRIGHT_MEAN
                ),
            }

        qc_raw_m = _qc_metrics_from_rgb(rgb)
        qc_pp_m = _qc_metrics_from_rgb(rgb_proc)
        qc_raw = _qc_from_metrics(qc_raw_m)
        qc_pp = _qc_from_metrics(qc_pp_m)

        # Accept if either (raw or processed) passes QC
        qc = {
            "blur_lapvar": qc_pp["blur_lapvar"],
            "illum_mean": qc_pp["illum_mean"],
            "illum_std": qc_pp["illum_std"],
            "is_blurry": qc_pp["is_blurry"],
            "is_too_dark": qc_pp["is_too_dark"],
            "is_too_bright": qc_pp["is_too_bright"],
            "quality_ok": qc_pp["quality_ok"] or qc_raw["quality_ok"],
        }

        # -------- Ensemble on processed RGB --------
        pA = self._pipeline_probs(rgb_proc, use_clahe=APPLY_CLAHE, use_fov=USE_FOVCROP)
        if USE_ENSEMBLE:
            pB = self._pipeline_probs(rgb_proc, use_clahe=not APPLY_CLAHE, use_fov=not USE_FOVCROP)
            p = np.maximum(pA, pB)
            s = p.sum()
            if s > 0:
                p = p / s
        else:
            p = pA

        if SWAP_LOGITS:
            p = p[::-1]

        # Map to class indices robustly
        try:
            idx_n = self.class_names.index("Normal")
            idx_m = self.class_names.index("Myopia")
        except Exception:
            idx_n, idx_m = 0, 1

        p_normal = float(p[idx_n])
        p_myopia = float(p[idx_m])

        # -------- Morphology proxies (OpenCV) --------
        features = compute_myopia_features(image_bytes)
        comp = float(features.get("composite_myopia_score", 0.0))

        # -------- Conservative decision (prob ∧ morphology ∧ QC) --------
        decision_trace: list[str] = []
        is_high_prob = p_myopia >= TUNED_THR
        has_morph = comp >= VETO_COMP
        qc_ok = bool(qc.get("quality_ok", True))

        if is_high_prob and has_morph and qc_ok:
            pred_label = "Myopia"
            decision_trace.append(
                f"base:Myopia p={p_myopia:.3f}≥{TUNED_THR:.2f} ∧ "
                f"comp={comp:.3f}≥{VETO_COMP:.2f} ∧ qc_ok"
            )
        else:
            reasons = []
            if not is_high_prob:
                reasons.append(f"p<{TUNED_THR:.2f}")
            if not has_morph:
                reasons.append(f"comp<{VETO_COMP:.2f}")
            if not qc_ok:
                reasons.append("qc_fail")
            pred_label = "Normal"
            decision_trace.append(f"base:Normal ({' & '.join(reasons) or 'guard'})")

        # -------- Optional gray-zone upgrade (only with strong morphology) --------
        if (
            pred_label == "Normal"
            and (GRAY_LOW <= p_myopia < GRAY_HIGH)
            and (comp >= 0.60)
            and qc_ok
        ):
            pred_label = "Myopia (Morphology Supported)"
            decision_trace.append(
                f"grayzone: upgrade p={p_myopia:.3f}∈[{GRAY_LOW:.2f},{GRAY_HIGH:.2f}) ∧ comp≥0.60 ∧ qc_ok"
            )

        # -------- ADS mapping --------
        ads_id = 42 if "Myopia" in pred_label else 26
        decision = "final"

        # -------- Optional geometry overlay --------
        geometry_metrics: dict = {}
        try:
            safe_name = filename if (filename and isinstance(filename, str)) else "image.jpg"
            geometry_metrics = analyze_fundus_geometry(image_bytes, filename=safe_name)
        except Exception as e:
            # keep service resilient; just log server-side
            print(f"[WARN] geometry overlay failed: {e}")
            geometry_metrics = {}

        # -------- Compose JSON --------
        return {
            "myopia": {
                "prediction": pred_label,
                "ads_id": ads_id,
                "decision": decision,
                "confidence_scores": {"Normal": p_normal, "Myopia": p_myopia},
                "quality": qc,
                "thresholds": {"tuned": TUNED_THR, "gray_zone": [GRAY_LOW, GRAY_HIGH]},
                "metrics": features,
                "geometry": geometry_metrics,
                "settings": {
                    "MODE": "precision",
                    "THRESHOLD": TUNED_THR,
                    "CLAHE": APPLY_CLAHE,
                    "FOVCROP": USE_FOVCROP,
                    "TTA": USE_TTA,
                    "ENSEMBLE": USE_ENSEMBLE,
                    "PREPROCESS": USE_PREPROC,
                    "SWAP_LOGITS": SWAP_LOGITS,
                },
                "classes": self.class_names,
                "decision_trace": decision_trace,
            }
        }


# ============================== FastAPI ==============================
app = FastAPI(title="Myopia Prediction API", version="3.0")

# Load model once
MODEL = MyopiaModelService(MODEL_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "device": str(MODEL.device),
        "classes": MODEL.class_names,
        "toggles": {
            "CLAHE": APPLY_CLAHE,
            "FOVCROP": USE_FOVCROP,
            "TTA": USE_TTA,
            "ENSEMBLE": USE_ENSEMBLE,
            "PREPROCESS": USE_PREPROC,
            "SWAP_LOGITS": SWAP_LOGITS,
        },
        "thresholds": {"tuned": TUNED_THR, "gray_zone": [GRAY_LOW, GRAY_HIGH]},
    }


@app.post("/predict-myopia")
async def predict_myopia(file: UploadFile = File(...)):
    try:
        if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
            raise HTTPException(status_code=415, detail="Unsupported image type.")

        data = await file.read()
        result = MODEL.predict(data, filename=file.filename)

        return JSONResponse(
            content=result,
            headers={
                "X-Model": "MyopiaBinary-Precision",
                "X-Threshold": str(TUNED_THR),
                "X-GrayZone": f"{GRAY_LOW}-{GRAY_HIGH}",
                "X-Morphology-Veto": "enabled",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Myopia microservice failed: {str(e)}"
        )
