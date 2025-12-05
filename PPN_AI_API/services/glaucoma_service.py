from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import torchvision
from torch.serialization import safe_globals
import io
import logging
from typing import Dict,Union, Any, Tuple, Optional
import uvicorn
from pydantic import BaseModel
import base64
import traceback
import os
import uuid

AI_DB_CONN = os.getenv("LIVE_CONNECTION_STRING", "")

def _db_get_g_folder(ai_ar_id: int):
    """Return (g_root_folder, db_image_name); create G if needed. Works with/without GFolder column."""
    if not AI_DB_CONN:
        return (None, None)
    try:
        import pyodbc, os
        conn = pyodbc.connect(AI_DB_CONN)
        cur  = conn.cursor()
        cur.execute("EXEC dbo.STP_GET_AI_IMAGE_PATH @AI_AR_ID = ?", ai_ar_id)
        row  = cur.fetchone()
        cols = [d[0] for d in cur.description] if cur.description else []
        cur.close(); conn.close()
        if not row:
            return (None, None)

        # map row -> dict by column name (case-insensitive)
        r = { cols[i].upper(): row[i] for i in range(len(cols)) }

        img_name = r.get("AI_IMAGE_NAME")
        img_loc  = r.get("AI_IMAGE_LOCATION")
        g_root   = r.get("GFOLDER")

        # If proc doesn't return GFolder, derive it
        if not g_root and img_loc:
            sep = "" if img_loc.endswith(("\\", "/")) else "\\"
            g_root = f"{img_loc}{sep}G"

        if g_root:
            os.makedirs(g_root, exist_ok=True)

        return (g_root, str(img_name) if img_name else None)
    except Exception as e:
        logging.error(f"_db_get_g_folder failed for {ai_ar_id}: {e}")
        return (None, None)

#from services.cdr_service import _db_get_g_folder

# Try to import numpy safely - if it fails, we'll work without it
try:
    import numpy as np
    HAS_NUMPY = True
    logger = logging.getLogger(__name__)
    logger.info("Numpy imported successfully")
except ImportError as e:
    HAS_NUMPY = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Numpy not available: {e}")

# Try to import opencv safely - if it fails, we'll work without it  
try:
    import cv2
    HAS_OPENCV = True
    logger.info("OpenCV imported successfully")
except ImportError as e:
    HAS_OPENCV = False
    logger.warning(f"OpenCV not available: {e}")

# Try to import TensorFlow/Keras for U-Net segmentation
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
    logger.info(f"TensorFlow imported successfully: {tf.__version__}")
except ImportError as e:
    HAS_TENSORFLOW = False
    logger.warning(f"TensorFlow not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_CONFIG = {
    'img_size': 224,
    'model_path': r'C:\AI_Images\Glaucoma\combined_dataset/resnet50_glaucoma.pt',
    'unet_model_path': r'C:\AI_Images\trained_models/MINIMAL_DUAL_224.h5',
    'class_names': ['Normal', 'Glaucoma'],
    'save_crops': True,
    'crops_folder': 'saved_crops',
    'use_sensitivity_adjustment': True,  # DISABLED BY DEFAULT
    'glaucoma_sensitivity': 0.01,  # Conservative boost
    'minimum_glaucoma_threshold': 0.5,  # Only boost very high confidence
    'use_segmentation': True,
    'segmentation_input_size': 224,
    'disc_thresh': 0.45,
    'cup_thresh':  0.40,
    'disc_min_frac': 0.003,   # 0.1% of fundus area
    'disc_max_frac': 0.08   ,    # up to 30%
    'min_cup_frac_of_disc': 0.03,  # 1% of disc area; below this we consider cup "missing"
}


# Response models
class GlaucomaDetectionResponse(BaseModel):
    glaucoma_probability: float
    normal_probability: float
    prediction: str
    ads_id: int
    confidence: float
    cdr_analysis: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    device: str

# Model architectures
class HybridGlaucomaModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(HybridGlaucomaModel, self).__init__()
        
        # ResNet50 branch
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # VGG16 branch  
        self.vgg = models.vgg16(pretrained=pretrained)
        self.vgg_features = self.vgg.features
        
        # Adaptive pooling for VGG features
        self.vgg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature fusion
        self.resnet_fc = nn.Linear(2048, 512)
        self.vgg_fc = nn.Linear(512, 512)
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # ResNet branch
        resnet_features = self.resnet_features(x)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        resnet_out = self.resnet_fc(resnet_features)
        
        # VGG branch
        vgg_features = self.vgg_features(x)
        vgg_features = self.vgg_pool(vgg_features)
        vgg_features = vgg_features.view(vgg_features.size(0), -1)
        vgg_out = self.vgg_fc(vgg_features)
        
        # Combine features
        combined = torch.cat([resnet_out, vgg_out], dim=1)
        output = self.classifier(combined)
        
        return output

class OptimizedResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(OptimizedResNet50, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final layers for glaucoma-specific features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class LightweightGlaucomaModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(LightweightGlaucomaModel, self).__init__()
        
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Custom classifier for glaucoma
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# Initialize FastAPI app
app = FastAPI(
    title="Glaucoma Detection API",
    description="AI-powered glaucoma detection from fundus images using hybrid deep learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None
transform = None
model_type = "resnet50"
segmentation_model = None

# ADD THIS FUNCTION TO YOUR glaucoma_service.py

from typing import Optional, Dict, Tuple
import numpy as np, cv2

def estimate_cdr_from_disc_brightness(
    disc_mask: np.ndarray,
    original_image: np.ndarray
) -> Dict:
    """
    Estimate cup inside the given disc using brightness.
    Returns CDR + masks for overlay.
    """
    try:
        # binarize disc
        disc_bin = (disc_mask > 0).astype(np.uint8)

        # grayscale image
        if original_image.ndim == 3:
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = original_image.copy()

        # only pixels inside disc
        yy, xx = np.where(disc_bin > 0)
        if yy.size == 0:
            return {"cdr": None, "method": "no_disc_found"}

        disc_pixels = gray[yy, xx]

        # choose a high percentile inside disc as threshold
        thr = np.percentile(disc_pixels, 75)

        # === this is the mask you asked about ===
        cup_mask_estimated = np.zeros_like(disc_bin, dtype=np.uint8)
        cup_mask_estimated[(gray >= thr) & (disc_bin > 0)] = 1

        # clean up cup
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cup_mask_estimated = cv2.morphologyEx(cup_mask_estimated, cv2.MORPH_CLOSE, k)
        cup_mask_estimated = cv2.morphologyEx(cup_mask_estimated, cv2.MORPH_OPEN,  k)

        # areas & diameters
        disc_area = int(disc_bin.sum())
        cup_area  = int(cup_mask_estimated.sum())
        if disc_area == 0:
            return {"cdr": None, "method": "no_disc_area"}

        disc_diam = 2.0 * np.sqrt(disc_area / np.pi)
        cup_diam  = 2.0 * np.sqrt(max(cup_area, 0) / np.pi)

        cdr_vert  = float(cup_diam / disc_diam) if disc_diam > 0 else None
        cdr_vert  = None if cdr_vert is None else min(0.90, max(0.05, cdr_vert))
        cdr_area  = float(cup_area / disc_area)

        return {
            "disc_area": float(disc_area),
            "cup_area":  float(cup_area),
            "disc_diameter": float(disc_diam),
            "cup_diameter":  float(cup_diam),
            "cdr_vertical": cdr_vert,
            "cdr_area": cdr_area,
            "cdr_primary": cdr_vert,
            "method": "brightness_estimation_fallback",
            "confidence": "medium",
            # expose masks for overlay (0/255 uint8)
            "overlay_masks": {
                "disc": (disc_bin * 255).astype(np.uint8),
                "cup":  (cup_mask_estimated * 255).astype(np.uint8),
            },
        }
    except Exception as e:
        return {"cdr": None, "method": "estimation_failed", "error": str(e)}

# at top of file with other config:
# near the top
DEBUG_DIR = os.path.abspath(os.environ.get("CDR_DEBUG_DIR", "./debug"))
OVERLAY_DIR = os.path.abspath("overlays")
os.makedirs(DEBUG_DIR, exist_ok=True)
app.mount("/overlays", StaticFiles(directory=OVERLAY_DIR), name="overlays")

def segment_optic_disc_cup(image: Image.Image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Run 2-channel U-Net (disc=ch0, cup=ch1).
    - Threshold sweep to stabilize weak masks
    - Keep small discs (mark as weak in meta)
    - Force cup inside disc; if erased or too tiny, mark cup as None
    - Save debug heatmaps/masks
    """
    if segmentation_model is None or not HAS_NUMPY:
        logger.warning("Segmentation model not available")
        return None, None, {"method": "not_available"}

    try:
        img_rgb = np.array(image.convert("RGB"))
        H, W = img_rgb.shape[:2]
        sz = int(MODEL_CONFIG.get("segmentation_input_size", MODEL_CONFIG.get("img_size", 224)))

        # NHWC [0,1]
        x = cv2.resize(img_rgb, (sz, sz), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)

        # forward: (sz,sz,2) sigmoid probs
        y = segmentation_model.predict(x, verbose=0)
        if isinstance(y, (list, tuple)):
            y = y[0]
        probs_small = y[0]  # (sz,sz,2)

        # --- auto-detect channel mapping ---
        # --- auto-detect channel mapping ---
        chan0 = cv2.resize(probs_small[..., 0], (W, H), interpolation=cv2.INTER_LINEAR)
        chan1 = cv2.resize(probs_small[..., 1], (W, H), interpolation=cv2.INTER_LINEAR)

        base_disc_t = float(MODEL_CONFIG.get("disc_thresh", 0.50))

        def area_frac(prob, t=base_disc_t):
            m = (prob >= t).astype(np.uint8)
            return m.sum() / float(H * W)

        f0 = area_frac(chan0)  # fraction of image above threshold
        f1 = area_frac(chan1)

        # Heuristic: the true disc channel should cover a small fraction of the frame (<~8%)
        # while the "background/fundus" channel often covers most of it.
        if f0 < f1:
            disc_prob = chan0
            cup_prob = chan1
        else:
            disc_prob = chan1
            cup_prob = chan0

        logger.info(f"[seg] channel map chosen: disc from {'ch0' if disc_prob is chan0 else 'ch1'} "
                    f"(f={min(f0, f1):.3f}), cup from {'ch1' if disc_prob is chan0 else 'ch0'} (other f={max(f0, f1):.3f})")

        base_disc_t = float(MODEL_CONFIG.get("disc_thresh", 0.50))
        base_cup_t  = float(MODEL_CONFIG.get("cup_thresh",  0.50))
        disc_ts = [base_disc_t, 0.45, 0.40, 0.35, 0.30]
        cup_ts  = [base_cup_t,  0.45, 0.40, 0.35, 0.30]

        fundus_area = H * W * 0.75
        min_disc_frac = float(MODEL_CONFIG.get("disc_min_frac", 0.001))  # 0.1%
        max_disc_frac = float(MODEL_CONFIG.get("disc_max_frac", 0.30))   # 30%
        min_cup_frac  = float(MODEL_CONFIG.get("min_cup_frac_of_disc", 0.03))  # 3%

        # ---------- DISC ----------
        disc_mask = None
        disc_t_used = base_disc_t
        disc_weak = False

        for dt in disc_ts:
            dm = (disc_prob >= dt).astype(np.uint8) * 255
            dm = cv2.morphologyEx(dm, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            cnts, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            candidate = np.zeros_like(dm)
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(candidate, [c], -1, 255, -1)
            area = int((candidate > 0).sum())
            frac = area / max(fundus_area, 1.0)
            if (min_disc_frac <= frac <= max_disc_frac):
                disc_mask = candidate
                disc_t_used = dt
                break

        # If still None, keep largest component at loosest threshold (don’t reject)
        if disc_mask is None:
            dm = (disc_prob >= disc_ts[-1]).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None, None, {"method": "no_disc"}
            disc_mask = np.zeros_like(dm)
            cv2.drawContours(disc_mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
            disc_weak = True  # definitely weak
        else:
            # flag as weak if outside preferred bounds (shouldn't happen here, but be explicit)
            disc_pixels = int((disc_mask > 0).sum())
            frac = disc_pixels / max(fundus_area, 1.0)
            disc_weak = not (min_disc_frac <= frac <= max_disc_frac)

        # ---------- CUP ----------
        cup_mask = None
        cup_t_used = None
        for ct in cup_ts:
            cm = (cup_prob >= ct).astype(np.uint8) * 255
            cm = cv2.morphologyEx(cm, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
            cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            candidate = np.zeros_like(cm)
            cv2.drawContours(candidate, [max(cnts, key=cv2.contourArea)], -1, 255, -1)

            # keep cup inside disc
            candidate = cv2.bitwise_and(candidate, disc_mask)
            if int((candidate > 0).sum()) > 0:
                cup_mask = candidate
                cup_t_used = ct
                break

        # If cup exists but is too tiny compared to disc, mark it missing so fallback can run
        disc_pixels = int((disc_mask > 0).sum())
        cup_pixels  = int((cup_mask > 0).sum()) if cup_mask is not None else 0
        cup_frac    = (cup_pixels / disc_pixels) if disc_pixels > 0 else 0.0
        if cup_mask is not None and cup_frac < min_cup_frac:
            logger.info(f"[seg] cup too small ({cup_frac:.3%} of disc) -> mark as missing")
            cup_mask = None
            cup_pixels = 0
            cup_frac = 0.0

        # ---------- DEBUG DUMPS ----------
        uid = uuid.uuid4().hex[:8]
        try:
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{uid}_disc_prob.png"), (disc_prob*255).astype(np.uint8))
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{uid}_cup_prob.png"),  (cup_prob*255).astype(np.uint8))
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{uid}_disc_mask.png"), disc_mask)
            if cup_mask is not None:
                cv2.imwrite(os.path.join(DEBUG_DIR, f"{uid}_cup_mask.png"),  cup_mask)
            else:
                cv2.imwrite(os.path.join(DEBUG_DIR, f"{uid}_cup_mask.png"),  np.zeros_like(disc_mask))
        except Exception:
            pass

        return disc_mask, cup_mask, {
            "method": "unet_segmentation",
            "uid": uid,
            "disc_pixels": disc_pixels,
            "cup_pixels":  cup_pixels,
            "cup_frac_of_disc": round(float(cup_frac), 4),
            "disc_t_used": float(disc_t_used),
            "cup_t_used":  float(cup_t_used) if cup_t_used is not None else None,
            "disc_min_frac": min_disc_frac,
            "disc_max_frac": max_disc_frac,
            "disc_weak": bool(disc_weak),
        }

    except Exception as e:
        logger.error(f"U-Net segmentation failed: {e}")
        logger.error(traceback.format_exc())
        return None, None, {"method": "failed", "error": str(e)}


# REPLACE your calculate_cdr_from_masks function with this enhanced version:
def _rescue_disc_from_brightness(rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Build a disc mask from very bright pixels (optic nerve head is typically brightest).
    Returns 0/255 uint8 mask or None.
    """
    try:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        thr = np.percentile(v, 99.0)  # top 1% brightest
        m = np.zeros_like(v, dtype=np.uint8)
        m[v >= thr] = 255

        # clean up and keep largest blob
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        out = np.zeros_like(m)
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
        return out
    except Exception:
        return None

def _to_rgb_array(img) -> np.ndarray:
    """
    Accepts PIL.Image or np.ndarray and returns an RGB uint8 np.ndarray (H,W,3).
    """
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:  # grayscale
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.ndim == 3:
            if arr.shape[2] == 3:
                return arr.astype(np.uint8)
            if arr.shape[2] == 4:
                # drop alpha
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        raise ValueError(f"Unsupported ndarray shape for image: {arr.shape}")
    elif isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def calculate_cdr_from_masks(
    disc_mask: np.ndarray,
    cup_mask: Optional[np.ndarray],
    original_image: Optional[np.ndarray] = None
) -> Dict:
    """Calculate vertical CDR. If cup is missing/tiny, use brightness fallback on the ORIGINAL image."""
    try:

        if not HAS_NUMPY or not HAS_OPENCV:
            return {"cdr": None, "method": "libraries_unavailable"}

        # ---- DISC ----
        disc_cnts, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not disc_cnts:
            return {"cdr": None, "method": "no_disc_found"}

        disc = max(disc_cnts, key=cv2.contourArea)
        disc_area = float(cv2.contourArea(disc))
        if disc_area < 5000:
            return {"cdr": None, "method": "invalid_disc_area"}

        # vertical diameter via ellipse (minor axis), fallback to area-diameter
        if len(disc) >= 5:
            (_, _), (a, b), _ = cv2.fitEllipse(disc)  # a=major, b=minor (width,height)
            disc_vert = min(a, b) / 2.0
        else:
            disc_vert = np.sqrt(disc_area / np.pi)

        result = {
            "disc_area": disc_area,
            "disc_diameter": float(2 * disc_vert),
            "method": "disc_detected"
        }

        # ---- CUP ----
        min_cup_frac = float(MODEL_CONFIG.get("min_cup_frac_of_disc", 0.03))  # treat tiny cups as missing
        have_valid_cup = False
        cup_area = 0.0
        cup_vert = 0.0

        if cup_mask is not None:
            cup_cnts, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cup_cnts:
                cup = max(cup_cnts, key=cv2.contourArea)
                cup_area = float(cv2.contourArea(cup))
                if len(cup) >= 5:
                    (_, _), (a, b), _ = cv2.fitEllipse(cup)
                    cup_vert = min(a, b) / 2.0
                else:
                    cup_vert = np.sqrt(max(cup_area, 0.0) / np.pi)

                if disc_area > 0 and (cup_area / disc_area) >= min_cup_frac and cup_vert > 0:
                    have_valid_cup = True

        if have_valid_cup:
            # --- compute core metrics ---
            cdr = cup_vert / disc_vert if disc_vert > 0 else None
            cup_frac = (cup_area / disc_area) if disc_area > 0 else 0.0

            # centers for sanity check
            def _center_from_contour(cnt):
                if len(cnt) >= 5:
                    (cx, cy), _, _ = cv2.fitEllipse(cnt)
                    return float(cx), float(cy)
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    return None
                return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])

            disc_ctr = _center_from_contour(disc)
            cup_ctr = _center_from_contour(cup) if 'cup' in locals() else None
            center_dist = 0.0
            if disc_ctr is not None and cup_ctr is not None:
                center_dist = float(np.hypot(disc_ctr[0] - cup_ctr[0], disc_ctr[1] - cup_ctr[1]))

            # plausibility gate
            # build suspect flag (keep whatever you already have or this)
            suspect = (
                    cdr is None or
                    cdr > 0.90 or
                    cup_frac > 0.70 or
                    center_dist > 0.5 * disc_vert
            )

            # REPLACE your existing suspect block with this:
            if suspect:
                logger.warning(
                    f"[cdr] implausible dual-seg (cdr={cdr if cdr is not None else 'NA':>5}, "
                    f"cup_frac={cup_frac:.2f}, center_dist={center_dist:.1f}) → fallback"
                )
                if original_image is not None:
                    bres = estimate_cdr_from_disc_brightness((disc_mask > 0).astype(np.uint8), original_image)
                    if bres.get("cdr_primary") is not None:
                        result.update(bres)
                        result["method"] = "brightness_estimation_fallback"
                        return result

                result.update({
                    "cup_area": cup_area,
                    "cup_diameter": float(2 * cup_vert) if cup_vert > 0 else None,
                    "cdr_vertical": None,
                    "cdr_area": round(cup_frac, 3) if disc_area > 0 else None,
                    "cdr_primary": None,
                    "method": "dual_segmentation_suspect",
                    "confidence": "low",
                    "note": "Suspicious segmentation; not reporting CDR."
                })
                return result

            # normal, plausible dual-seg path
            if cdr is not None:
                cdr = float(max(0.05, min(0.95, cdr)))  # clamp to a sane display range
                result.update({
                    "cup_area": cup_area,
                    "cup_diameter": float(2 * cup_vert),
                    "cdr_vertical": round(cdr, 2),
                    "cdr_area": round(cup_frac, 3),
                    "cdr_primary": round(cdr, 2),
                    "method": "dual_segmentation",
                    "confidence": "high",
                    "glaucoma_risk": "high" if cdr > 0.7 else "moderate" if cdr > 0.6 else "low"
                })
                return result

        # ---- Brightness fallback when cup missing/tiny ----
        if original_image is not None:
            logger.info("Cup missing/too small → using brightness fallback")
            bres = estimate_cdr_from_disc_brightness((disc_mask > 0).astype(np.uint8), original_image)
            if bres.get("cdr_primary") is not None:
                result.update(bres)
                return result

        # Couldn’t compute reliably
        result.update({
            "cdr_primary": None,
            "method": "disc_only_no_cup",
            "note": "Cup not detected; CDR not computed"
        })
        return result

    except Exception as e:
        logger.error(f"CDR calculation failed: {e}")
        return {"cdr": None, "method": "calculation_failed", "error": str(e)}



def detect_and_crop_optic_disc_with_unet(image: Image.Image, save_crops: bool = False, filename_prefix: str = "api_request"):
    logger.info(f"Starting U-Net-based optic disc detection for CDR analysis: {filename_prefix}")
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)  # keep original for brightness fallback
        disc_mask, cup_mask, seg_metadata = segment_optic_disc_cup(image)

        if disc_mask is not None:
            logger.info("U-Net segmentation successful, calculating CDR")
            cdr_results = calculate_cdr_from_masks(disc_mask, cup_mask, img_array)
            cdr_results.setdefault("overlay_masks", {})
            cdr_results["overlay_masks"]["disc"] = disc_mask
            cdr_results["overlay_masks"]["cup"] = cup_mask

            if HAS_NUMPY and HAS_OPENCV:
                coords = np.where(disc_mask > 0)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    h, w = img_array.shape[:2]
                    padding = max(y_max - y_min, x_max - x_min) // 4
                    y_min = max(0, y_min - padding); y_max = min(h, y_max + padding)
                    x_min = max(0, x_min - padding); x_max = min(w, x_max + padding)
                    cropped_pil = Image.fromarray(img_array[y_min:y_max, x_min:x_max])
                    logger.info(f"U-Net based crop: ({x_min},{y_min}) to ({x_max},{y_max})")
                    return cropped_pil, cdr_results
            # fallback crop if mask exists but bbox failed
            return center_crop_fallback(image), cdr_results

        logger.warning("U-Net segmentation failed, falling back to traditional detection")
        return detect_and_crop_optic_disc(image, save_crops, filename_prefix), {"method": "fallback"}

    except Exception as e:
        logger.error(f"U-Net optic disc detection failed: {e}")
        logger.error(traceback.format_exc())
        return center_crop_fallback(image), {"method": "error", "error": str(e)}

# CDR Clinical Assessment Function
def get_cdr_clinical_assessment(cdr_value: Optional[float]) -> Dict:
    """Get clinical assessment based on CDR value"""
    if cdr_value is None:
        return {
            "cdr_value": None,
            "clinical_label": "Unable to Calculate",
            "risk_level": "Unknown",
            "description": "CDR could not be calculated from the image",
            "recommendation": "Manual assessment by ophthalmologist recommended"
        }
    
    if cdr_value < 0.3:
        return {
            "cdr_value": round(cdr_value, 3),
            "clinical_label": "Normal",
            "risk_level": "Low Risk",
            "description": "Normal optic disc appearance",
            "recommendation": "Continue routine annual eye exams"
        }
    elif cdr_value < 0.6:
        return {
            "cdr_value": round(cdr_value, 3),
            "clinical_label": "Normal",
            "risk_level": "Low Risk", 
            "description": "Normal range for cup-to-disc ratio",
            "recommendation": "Continue routine annual eye exams"
        }
    elif cdr_value < 0.7:
        return {
            "cdr_value": round(cdr_value, 3),
            "clinical_label": "Glaucoma Suspect",
            "risk_level": "Medium Risk",
            "description": "CDR in borderline range - monitoring recommended",
            "recommendation": "Follow-up with ophthalmologist within 6 months"
        }
    else:
        return {
            "cdr_value": round(cdr_value, 3),
            "clinical_label": "High Risk",
            "risk_level": "High Risk",
            "description": "CDR suggests possible glaucomatous changes",
            "recommendation": "URGENT: Ophthalmologist evaluation within 2 weeks"
        }

# U-Net segmentation functions
def load_segmentation_model():
    """Load TF/Keras U-Net (2-channel: disc,cup)"""
    global segmentation_model

    if not HAS_TENSORFLOW:
        logger.warning("TensorFlow not available, segmentation disabled")
        return False

    try:
        unet_path = MODEL_CONFIG.get('unet_model_path')  # <- use your config
        if not unet_path or not os.path.exists(unet_path):
            logger.warning(f"U-Net model not found at {unet_path!r}, segmentation disabled")
            return False

        segmentation_model = keras.models.load_model(unet_path, compile=False)
        logger.info(f"U-Net loaded: {unet_path}")
        logger.info(f"Input shape: {segmentation_model.input_shape}, Output shape: {segmentation_model.output_shape}")
        return True

    except Exception as e:
        logger.error(f"Failed to load U-Net: {e}")
        segmentation_model = None
        return False

# at top of file with other config:
DEBUG_DIR = os.path.abspath(os.environ.get("CDR_DEBUG_DIR", "./debug"))
os.makedirs(DEBUG_DIR, exist_ok=True)

def apply_glaucoma_sensitivity_adjustment(probs: list) -> list:
    """Optionally adjust probabilities to increase sensitivity for glaucoma detection"""
    
    if not MODEL_CONFIG.get('use_sensitivity_adjustment', False):
        logger.info("Sensitivity adjustment disabled - using raw model predictions")
        return probs
    
    normal_prob = float(probs[0])
    glaucoma_prob = float(probs[1])
    
    sensitivity_boost = MODEL_CONFIG.get('glaucoma_sensitivity', 0.05)
    min_threshold = MODEL_CONFIG.get('minimum_glaucoma_threshold', 0.85)
    
    logger.info(f"Original probabilities: Normal={normal_prob:.4f}, Glaucoma={glaucoma_prob:.4f}")
    
    if glaucoma_prob > min_threshold:
        adjusted_glaucoma = min(glaucoma_prob + sensitivity_boost, 0.99)
        adjusted_normal = 1.0 - adjusted_glaucoma
        logger.info(f"High confidence detected - applied {sensitivity_boost} boost")
        logger.info(f"Adjusted probabilities: Normal={adjusted_normal:.4f}, Glaucoma={adjusted_glaucoma:.4f}")
        return [adjusted_normal, adjusted_glaucoma]
    else:
        logger.info(f"Confidence below threshold ({min_threshold}) - no adjustment applied")
        return probs


def map_glaucoma_ads_id(probs: list) -> dict:
    """Convert probabilities into final label + ADS_ID mapping.
       Defaults to Normal (16) if uncertain."""
    try:
        normal_prob, glaucoma_prob = probs
        if glaucoma_prob >= normal_prob:
            return {
                "label": "Glaucoma",
                "confidence": glaucoma_prob,
                "ads_id": 17
            }
        else:
            return {
                "label": "Normal",
                "confidence": normal_prob,
                "ads_id": 16
            }
    except Exception as e:
        logger.warning(f"ADS_ID mapping failed, defaulting to Normal: {e}")
        return {
            "label": "Normal",
            "confidence": 0.0,
            "ads_id": 16
        }


def detect_and_crop_optic_disc(image: Image.Image, save_crops: bool = False, filename_prefix: str = "api_request"):
    """Traditional computer vision method for optic disc detection"""
    logger.info(f"Using traditional optic disc detection for {filename_prefix}")
    
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if HAS_OPENCV and HAS_NUMPY:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale for circle detection
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Use HoughCircles to detect optic disc
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=int(min(image.size) * 0.5),
                param1=50,
                param2=30,
                minRadius=int(min(image.size) * 0.05),
                maxRadius=int(min(image.size) * 0.25)
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # Take the first (most prominent) circle
                x, y, r = circles[0]
                
                # Expand crop region around detected circle
                crop_radius = int(r * 2.5)
                x1 = max(0, x - crop_radius)
                y1 = max(0, y - crop_radius)
                x2 = min(image.size[0], x + crop_radius)
                y2 = min(image.size[1], y + crop_radius)
                
                # Crop the image
                cropped = image.crop((x1, y1, x2, y2))
                
                logger.info(f"Detected optic disc at ({x}, {y}) with radius {r}")
                return cropped
            else:
                logger.warning("No optic disc detected, using center crop")
                return center_crop_fallback(image)
        else:
            logger.warning("OpenCV not available, using center crop")
            return center_crop_fallback(image)
            
    except Exception as e:
        logger.error(f"Traditional optic disc detection failed: {e}")
        return center_crop_fallback(image)

def center_crop_fallback(image: Image.Image):
    """Fallback center crop when optic disc detection fails"""
    width, height = image.size
    
    # Make square crop from center
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    
    cropped = image.crop((left, top, right, bottom))
    logger.info(f"Fallback center crop: {size}x{size}")
    return cropped

def preprocess_fundus_image_safe(image: Image.Image, save_crops: bool = False, filename_prefix: str = "api_request") -> Tuple[torch.Tensor, Dict]:
    """Enhanced preprocessing with U-Net segmentation for precise CDR analysis"""
    
    logger.info(f"Starting preprocessing for {filename_prefix}")
    logger.info(f"Input image mode: {image.mode}, size: {image.size}")
    logger.info(f"Available: numpy={HAS_NUMPY}, opencv={HAS_OPENCV}, tensorflow={HAS_TENSORFLOW}")
    logger.info(f"U-Net segmentation: {'enabled' if segmentation_model is not None else 'disabled'}")
    
    try:
        cdr_results = {}
        
        # Step 1: Choose detection method based on available tools
        if segmentation_model is not None and MODEL_CONFIG.get('use_segmentation', True):
            logger.info("Using U-Net segmentation for optic disc detection")
            optic_disc_crop, cdr_results = detect_and_crop_optic_disc_with_unet(image, save_crops, filename_prefix)
        else:
            logger.info("Using traditional computer vision for optic disc detection")
            optic_disc_crop = detect_and_crop_optic_disc(image, save_crops, filename_prefix)
            cdr_results = {"method": "traditional_cv", "cdr_available": False}
        
        # Step 2: Enhance contrast for better features
        try:
            enhancer = ImageEnhance.Contrast(optic_disc_crop)
            enhanced_crop = enhancer.enhance(1.4)
            logger.info("Applied contrast enhancement")
        except Exception as enhance_error:
            logger.warning(f"Contrast enhancement failed: {enhance_error}")
            enhanced_crop = optic_disc_crop
        
        # Step 3: Convert to tensor for model inference
        try:
            transform_pipeline = transforms.Compose([
                transforms.Resize((MODEL_CONFIG['img_size'], MODEL_CONFIG['img_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            tensor = transform_pipeline(enhanced_crop)
            logger.info(f"Transform successful, tensor shape: {tensor.shape}")
            
        except Exception as transform_error:
            logger.warning(f"Standard transform failed: {transform_error}, using fallback")
            
            # Fallback: manual tensor creation
            resized_crop = enhanced_crop.resize((MODEL_CONFIG['img_size'], MODEL_CONFIG['img_size']), Image.Resampling.LANCZOS)
            
            simple_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            tensor = simple_transform(resized_crop)
            logger.info(f"Fallback transform successful, tensor shape: {tensor.shape}")
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        logger.info(f"Final tensor shape: {tensor.shape}")
        
        # Process CDR results for clinical assessment
        cdr_value = cdr_results.get('cdr_primary', None)
        cdr_assessment = get_cdr_clinical_assessment(cdr_value)
        
        # Add technical details to assessment
        if cdr_results:
            cdr_assessment.update({
                "technical_details": {
                    "segmentation_method": cdr_results.get('method', 'unknown'),
                    "disc_area": cdr_results.get('disc_area', None),
                    "cup_area": cdr_results.get('cup_area', None),
                    "disc_diameter": cdr_results.get('disc_diameter', None),
                    "cup_diameter": cdr_results.get('cup_diameter', None)
                }
            })
        
        # Log CDR results if available
        if cdr_value is not None:
            logger.info(f"CDR Analysis: {cdr_value:.3f} - {cdr_assessment['clinical_label']} ({cdr_assessment['risk_level']})")
        else:
            logger.info("CDR calculation not available")
        
        return tensor, cdr_assessment
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Preprocessing failed: {str(e)}"
        )

def preprocess_fundus_image(image: Image.Image, save_crops: bool = False, filename_prefix: str = "api_request") -> Tuple[torch.Tensor, Dict]:
    """Main preprocessing function - returns tensor and CDR analysis"""
    return preprocess_fundus_image_safe(image, save_crops, filename_prefix)

def load_model():
    """Load the trained glaucoma detection model and U-Net segmentation model"""
    global model, device, transform, model_type
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load U-Net segmentation model first
        unet_loaded = load_segmentation_model()
        if unet_loaded:
            logger.info("U-Net segmentation model ready for precise CDR analysis")
        else:
            logger.info("U-Net not available, using traditional computer vision methods")
        
        # Initialize classification model based on type
        if model_type == 'hybrid':
            model = HybridGlaucomaModel(num_classes=2, pretrained=False)
        elif model_type == 'resnet50':
            model = OptimizedResNet50(num_classes=2, pretrained=False)
        elif model_type == 'lightweight':
            model = LightweightGlaucomaModel(num_classes=2, pretrained=False)
        else:
            # Fallback to simple ResNet50
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
        
        # Load trained weights if available
        try:
            checkpoint = torch.load(MODEL_CONFIG['model_path'], map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
            else:
                # Handle direct state dict
                if isinstance(checkpoint, dict):
                    checkpoint = {k: v for k, v in checkpoint.items() 
                                if not (k.startswith('fc.') and hasattr(model, 'fc'))}
                model.load_state_dict(checkpoint, strict=False)
            logger.info("Classification model weights loaded successfully")
        except FileNotFoundError:
            logger.warning("No trained weights found. Using pretrained ImageNet weights.")
            if model_type == 'hybrid':
                model = HybridGlaucomaModel(num_classes=2, pretrained=True)
            elif model_type == 'resnet50':
                model = OptimizedResNet50(num_classes=2, pretrained=False)
            elif model_type == 'lightweight':
                model = LightweightGlaucomaModel(num_classes=2, pretrained=True)
        except Exception as e:
            logger.warning(f"Error loading weights: {e}. Using pretrained ImageNet weights.")
            if model_type == 'hybrid':
                model = HybridGlaucomaModel(num_classes=2, pretrained=True)
            elif model_type == 'resnet50':
                model = OptimizedResNet50(num_classes=2, pretrained=False)
            elif model_type == 'lightweight':
                model = LightweightGlaucomaModel(num_classes=2, pretrained=True)
        
        model.to(device)
        model.eval()
        
        # Define transforms
        try:
            transform = transforms.Compose([
                transforms.Resize((MODEL_CONFIG['img_size'], MODEL_CONFIG['img_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            logger.info("Transform pipeline created successfully")
        except Exception as transform_error:
            logger.error(f"Error creating transform pipeline: {transform_error}")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            logger.info("Using minimal transform pipeline")
        
        logger.info("Models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.error(traceback.format_exc())
        return False

def _estimate_cup_mask_from_brightness(disc_mask: np.ndarray, original_rgb: np.ndarray) -> np.ndarray:
    """Return a binary cup mask (uint8 0/255) estimated from bright pixels inside the disc."""
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY) if original_rgb.ndim == 3 else original_rgb
    yy, xx = np.where(disc_mask > 0)
    if yy.size == 0:
        return np.zeros_like(disc_mask, dtype=np.uint8)
    thr = np.percentile(gray[yy, xx], 75)
    cm = np.zeros_like(disc_mask, dtype=np.uint8)
    cm[(gray >= thr) & (disc_mask > 0)] = 255
    cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    cm = cv2.morphologyEx(cm, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    # keep largest blob
    cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(cm)
    cm2 = np.zeros_like(cm)
    cv2.drawContours(cm2, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return cm2

def _fit_minor_axis(mask: np.ndarray) -> Tuple[Optional[Tuple[int,int]], Optional[Tuple[int,int]], Optional[float]]:
    """Return (center_xy, axes_wh, angle_deg) from fitEllipse if possible."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None, None, None
    center, axes, angle = cv2.fitEllipse(c)  # center(x,y), axes(width,height)
    return (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), float(angle)

def _create_overlay(image_pil: Image.Image,
                    disc_mask: Optional[np.ndarray],
                    cup_mask: Optional[np.ndarray],
                    cdr_value: Optional[float],
                    method: str) -> str:
    """Draw disc (blue) and cup (green) with CDR label; save PNG and return relative URL."""
    rgb = np.array(image_pil.convert("RGB"))
    canvas = rgb.copy()  # writable

    # If no cup mask but we have a disc, try brightness estimate so we can draw something
    if cup_mask is None and disc_mask is not None:
        cup_mask = _estimate_cup_mask_from_brightness(disc_mask, rgb)

    # Draw ellipses if possible
    if disc_mask is not None:
        dc = _fit_minor_axis(disc_mask)
        if dc[0] and dc[1]:
            cv2.ellipse(canvas, dc, (255, 0, 0), 2, lineType=cv2.LINE_AA)  # blue

    if cup_mask is not None and (cup_mask > 0).any():
        cc = _fit_minor_axis(cup_mask)
        if cc[0] and cc[1]:
            cv2.ellipse(canvas, cc, (0, 255, 0), 2, lineType=cv2.LINE_AA)  # green

    # Label (top-left)
    label = f"Vertical CDR: {cdr_value:.2f}" if isinstance(cdr_value, (float, int)) else "Vertical CDR: N/A"
    cv2.rectangle(canvas, (8, 8), (8+310, 40), (20,20,20), -1)
    cv2.putText(canvas, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(canvas, method, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

    # Save & return URL
    fname = f"{uuid.uuid4().hex}.png"
    fpath = os.path.join(OVERLAY_DIR, fname)
    Image.fromarray(canvas).save(fpath)
    return f"/overlays/{fname}"

def _fit_ellipse(mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None
    return cv2.fitEllipse(c)  # ((cx,cy),(major,minor),angle)

def draw_cup_disc_overlay(rgb: np.ndarray,
                          disc_mask: Optional[np.ndarray],
                          cup_mask: Optional[np.ndarray],
                          cdr: Optional[float],
                          method: str) -> np.ndarray:

    canvas = rgb.copy()
    # disc
    if disc_mask is not None:
        e = _fit_ellipse(disc_mask)
        if e is not None:
            (cx,cy), (a,b), ang = e
            cv2.ellipse(canvas, (int(cx),int(cy)), (int(a/2),int(b/2)), ang, 0, 360, (255,0,0), 2, cv2.LINE_AA)
    # cup
    if cup_mask is not None:
        e = _fit_ellipse(cup_mask)
        if e is not None:
            (cx,cy), (a,b), ang = e
            cv2.ellipse(canvas, (int(cx),int(cy)), (int(a/2),int(b/2)), ang, 0, 360, (0,255,0), 2, cv2.LINE_AA)

    # label
    label = f"Vertical CDR: {cdr:.2f}" if cdr is not None else "Vertical CDR: N/A"
    cv2.putText(canvas, label, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(canvas, method, (12,46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
    return canvas


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Glaucoma Detection API...")
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_type=model_type,
        device=str(device) if device else "not_set"
    )

@app.get("/health", response_model=HealthResponse) 
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_type=model_type,
        device=str(device) if device else "not_set"
    )
#predict-glaucoma
def _stem_and_ext(db_image_name: Optional[str], upload_filename: Optional[str]) -> Tuple[str, str]:
    base = db_image_name or upload_filename or "image.jpg"
    name, ext = os.path.splitext(os.path.basename(base))
    if not ext:
        ext = ".jpg"
    return name, ext


def _disc_crop_box(disc_mask: Optional[np.ndarray], pad_ratio: float = 0.45) -> Optional[Tuple[int,int,int,int]]:
    """
    Returns (x1,y1,x2,y2) padded around disc mask. None if mask invalid.
    """
    if disc_mask is None:
        return None
    m = (disc_mask > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    cy = (y1 + y2) // 2
    cx = (x1 + x2) // 2
    pad_y = int(h * pad_ratio)
    pad_x = int(w * pad_ratio)
    y1p, y2p = max(0, cy - (h // 2 + pad_y)), cy + (h // 2 + pad_y)
    x1p, x2p = max(0, cx - (w // 2 + pad_x)), cx + (w // 2 + pad_x)
    return (x1p, y1p, x2p, y2p)

# thresholds (tune as needed)
FUSE_CONF_MAX       = 0.60   # upgrade if Normal with low conf
FUSE_CDR_THRESH     = 0.75
DOWNGRADE_CONF_MAX  = 0.55   # downgrade if Glaucoma with low conf
DOWNGRADE_CDR_MAX   = 0.40   # and CDR is clearly normal

def fuse_results(m: Dict[str, Any], c: Dict[str, Any], image_file: UploadFile, eye_side: str):
    """Fuse model + CDR with upgrade/downgrade safeguards"""
    model_ads   = int(m.get("ads_id", 16))
    g_prob      = float(m.get("glaucoma_probability", 0.0))
    n_prob      = float(m.get("normal_probability",   0.0))
    model_conf  = max(g_prob, n_prob)

    model_pred = "Glaucoma" if g_prob > n_prob else "Normal"
    model_ads_from_probs = 17 if model_pred == "Glaucoma" else 16

    if model_ads != model_ads_from_probs:
        logging.warning(
            f"Model ads_id={model_ads} disagrees with probs "
            f"(g={g_prob:.3f}, n={n_prob:.3f}) -> forcing {model_ads_from_probs}"
        )
        model_ads = model_ads_from_probs

    # CDR
    v_cdr = c.get("vertical_cdr")
    v_cdr = float(v_cdr) if v_cdr not in (None, "", "NA") else None

    final_ads     = model_ads
    final_pred    = "Glaucoma" if final_ads == 17 else "Normal"
    fuse_applied  = False
    fuse_reason   = None

    if v_cdr is not None:
        # Upgrade safeguard
        #if (model_ads == 16 and model_conf < FUSE_CONF_MAX and v_cdr >= FUSE_CDR_MIN_UP):
        if model_ads == 16 and model_conf < FUSE_CONF_MAX and v_cdr >= FUSE_CDR_THRESH:
            final_ads, final_pred = 17, "Glaucoma"
            fuse_applied = True
            #fuse_reason  = f"Upgraded: weak Normal (conf={model_conf:.2f}) but CDR={v_cdr:.2f} ≥ {FUSE_CDR_MIN_UP}"
            fuse_reason = f"Upgraded by CDR safeguard (CDR={v_cdr:.2f} ≥ {FUSE_CDR_THRESH}, conf={model_conf:.2f} < {FUSE_CONF_MAX})"
        # Downgrade safeguard
        elif (model_ads == 17 and model_conf < DOWNGRADE_CONF_MAX and v_cdr < DOWNGRADE_CDR_MAX):
            final_ads, final_pred = 16, "Normal"
            fuse_applied = True
            fuse_reason  = f"Downgraded: weak Glaucoma (conf={model_conf:.2f}) but CDR={v_cdr:.2f} < {DOWNGRADE_CDR_MAX}"

    logging.info(
        f"[Fuse] file={image_file.filename} eye={eye_side} "
        f"model:g={g_prob:.3f} n={n_prob:.3f} conf={model_conf:.3f} "
        f"cdr={v_cdr if v_cdr is not None else 'NA'} "
        f"final={final_pred}({final_ads}) "
        f"{'fused:'+fuse_reason if fuse_applied else 'no_fuse'}"
    )

    return final_ads, final_pred, model_conf, fuse_applied, fuse_reason

@app.post("/predict-glaucoma")
async def predict_glaucoma(file: UploadFile = File(...), referral_id: str = None):
    """Predict glaucoma from fundus image with CDR analysis"""

    # Resolve G folder if ai_ar_id provided
    g_root, db_image_name = (None, None)
    if referral_id is not None:
        g_root, db_image_name = _db_get_g_folder(int(referral_id))

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty image upload")
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Create deterministic names
        stem, ext = _stem_and_ext(db_image_name, file.filename)
        full_name = f"{stem}_overlay{ext}"
        crop_name = f"{stem}_overlay_crop{ext}"

        # Choose output directory
        out_dir = g_root if g_root else os.path.abspath("overlays")
        os.makedirs(out_dir, exist_ok=True)
        
        # Unique request ID
        request_id = str(uuid.uuid4())[:8]
        filename_prefix = f"upload_{file.filename}_{request_id}"
        
        # Preprocess & CDR
        processed_tensor, cdr_analysis = preprocess_fundus_image(
            image, 
            save_crops=MODEL_CONFIG.get('save_crops', False),
            filename_prefix=filename_prefix
        )
        processed_tensor = processed_tensor.to(device)
        
        # Model prediction
        with torch.no_grad():
            outputs = model(processed_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs = probabilities[0].cpu().numpy()
        
        # Apply sensitivity adjustment
        adjusted_probs = apply_glaucoma_sensitivity_adjustment(probs)

        # Determine prediction + ads_id (with safe fallback to 16)
        if adjusted_probs[1] > adjusted_probs[0]:
            final_pred = "Glaucoma"
            final_ads_id = 17
            final_conf = float(adjusted_probs[1])
        else:
            final_pred = "Normal"
            final_ads_id = 16
            final_conf = float(adjusted_probs[0])

        logger.info(
            f"Prediction for {file.filename}: "
            f"{final_pred} (ads_id={final_ads_id}, conf={final_conf:.4f})"
        )

        # Original RGB
        orig_rgb = np.array(image.convert("RGB"))

        # Get masks & values
        om = cdr_analysis.get("overlay_masks", {}) if isinstance(cdr_analysis, dict) else {}
        disc_m = om.get("disc")
        cup_m = om.get("cup")
        cdr = cdr_analysis.get("cdr_primary")
        method = cdr_analysis.get("method", "")

        # Draw overlay
        overlay_img = draw_cup_disc_overlay(orig_rgb, disc_m, cup_m, cdr, method)

        # Save overlay + crop
        full_path = os.path.join(out_dir, full_name)
        cv2.imwrite(full_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        overlay_path = full_path if g_root else f"/overlays/{full_name}"

        crop_path = None
        box = _disc_crop_box(disc_m, pad_ratio=0.45)
        if box is not None:
            x1, y1, x2, y2 = box
            H, W = overlay_img.shape[:2]
            x1, x2 = max(0, min(W, x1)), max(0, min(W, x2))
            y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                crop = overlay_img[y1:y2, x1:x2]
                crop_path = os.path.join(out_dir, crop_name)
                cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        overlay_url_full = full_path if g_root else f"/overlays/{full_name}"
        overlay_url_crop = (crop_path if g_root else (f"/overlays/{crop_name}" if crop_path else None))

        # Build final response
        # Apply decision threshold
        glaucoma_prob = float(adjusted_probs[1])
        normal_prob   = float(adjusted_probs[0])

        THRESHOLD = 0.65  # tune based on validation set

        if glaucoma_prob >= THRESHOLD:
            final_pred = "Glaucoma"
            final_ads_id = 17
        else:
            final_pred = "Normal"
            final_ads_id = 16

        resp = {
            "glaucoma_probability": round(float(adjusted_probs[1]), 4),
            "normal_probability": round(float(adjusted_probs[0]), 4),
            "prediction": final_pred,
            "ads_id": final_ads_id,   # <-- never None now
            "cdr_analysis": cdr_analysis,
            "overlay_url": f"/{overlay_path.replace(os.sep, '/')}",
            "overlay_url_full": overlay_url_full,
            "overlay_url_crop": overlay_url_crop,
            "referral_id": referral_id
        }

        return JSONResponse(content=resp)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_base64")
async def predict_from_base64(image_data: dict):
    """Predict from base64 encoded image with CDR analysis"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image_base64 = image_data.get("image", "")
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image data provided")
            
        # Remove data URL prefix if present
        if image_base64.startswith('data:image'):
            image_base64 = image_base64.split(',')[1]
            
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate unique filename prefix for this request
        request_id = str(uuid.uuid4())[:8]
        filename_prefix = f"base64_{request_id}"
        
        # Preprocess and get CDR analysis
        processed_tensor, cdr_analysis = preprocess_fundus_image(
            image,
            save_crops=MODEL_CONFIG.get('save_crops', False),
            filename_prefix=filename_prefix
        )
        processed_tensor = processed_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(processed_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probs = probabilities[0].cpu().numpy()
        
        # Apply sensitivity adjustment (only if enabled)
        adjusted_probs = apply_glaucoma_sensitivity_adjustment(probs)
        
        # Log prediction for debugging
        original_prediction = "Glaucoma" if probs[1] > probs[0] else "Normal"
        adjusted_prediction = "Glaucoma" if adjusted_probs[1] > adjusted_probs[0] else "Normal"
        
        logger.info(f"Base64 raw prediction: {original_prediction} (confidence: {max(probs):.4f})")
        if MODEL_CONFIG.get('use_sensitivity_adjustment', False):
            logger.info(f"Base64 adjusted prediction: {adjusted_prediction} (confidence: {max(adjusted_probs):.4f})")
        
        # Return response with CDR analysis
        return JSONResponse(content={
            "glaucoma_probability": round(float(adjusted_probs[1]), 4),
            "normal_probability": round(float(adjusted_probs[0]), 4),
            "prediction": "Glaucoma" if adjusted_probs[1] > adjusted_probs[0] else "Normal",
            "ads_id": 17 if adjusted_probs[1] > adjusted_probs[0] else 16,
            "confidence": float(adjusted_probs[1] if adjusted_probs[1] > adjusted_probs[0] else adjusted_probs[0]),
            "cdr_analysis": cdr_analysis
        })
        
    except Exception as e:
        logger.error(f"Base64 prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# CDR analysis endpoints
@app.get("/cdr/info")
async def get_cdr_info():
    """Get information about CDR analysis and clinical thresholds"""
    return {
        "cdr_thresholds": {
            "normal": "< 0.6",
            "glaucoma_suspect": "0.6 - 0.7", 
            "high_risk": "> 0.7"
        },
        "clinical_labels": {
            "normal": "Normal optic disc appearance",
            "glaucoma_suspect": "Borderline CDR - monitoring recommended",
            "high_risk": "CDR suggests possible glaucomatous changes"
        },
        "calculation_methods": {
            "unet_segmentation": "Uses AI segmentation to identify optic disc and cup regions",
            "traditional_cv": "Uses computer vision techniques like Hough circles",
            "area_based": "CDR calculated from area ratio (cup area / disc area)",
            "diameter_based": "CDR calculated from diameter ratio (cup diameter / disc diameter)"
        },
        "segmentation_status": {
            "unet_available": segmentation_model is not None,
            "opencv_available": HAS_OPENCV,
            "numpy_available": HAS_NUMPY,
            "tensorflow_available": HAS_TENSORFLOW
        },
        "recommendations": {
            "normal": "Continue routine annual eye exams",
            "glaucoma_suspect": "Follow-up with ophthalmologist within 6 months",
            "high_risk": "URGENT: Ophthalmologist evaluation within 2 weeks"
        }
    }

@app.post("/cdr/analyze")
async def analyze_cdr_only(file: UploadFile = File(...)):
    """Analyze only CDR without glaucoma prediction"""
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Generate unique filename prefix for this request
        request_id = str(uuid.uuid4())[:8]
        filename_prefix = f"cdr_only_{file.filename}_{request_id}"
        
        # Get CDR analysis only
        _, cdr_analysis = preprocess_fundus_image(
            image, 
            save_crops=MODEL_CONFIG.get('save_crops', False),
            filename_prefix=filename_prefix
        )
        
        logger.info(f"CDR-only analysis for {file.filename}: {cdr_analysis.get('clinical_label', 'Unknown')}")
        # original RGB
        orig_rgb = np.array(image.convert("RGB"))

        # get masks & values
        om = cdr_analysis.get("overlay_masks", {}) if isinstance(cdr_analysis, dict) else {}
        disc_m = om.get("disc")
        cup_m = om.get("cup")
        cdr = cdr_analysis.get("cdr_primary")
        method = cdr_analysis.get("method", "")

        # draw
        overlay_img = draw_cup_disc_overlay(orig_rgb, disc_m, cup_m, cdr, method)

        # save & add URL
        os.makedirs("overlays", exist_ok=True)
        overlay_path = os.path.join("overlays", f"{uuid.uuid4().hex[:8]}.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

        resp = {
            "filename": file.filename,
            "timestamp": str(uuid.uuid4()),
            "analysis_type": "cdr_only",
            "cdr_analysis": cdr_analysis,
            "overlay_url": f"/{overlay_path.replace(os.sep, '/')}"
        }
        return JSONResponse(content=resp)
        
    except Exception as e:
        logger.error(f"CDR analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"CDR analysis failed: {str(e)}")

# Sensitivity adjustment endpoints
@app.get("/sensitivity/settings")
async def get_sensitivity_settings():
    """Get current sensitivity adjustment settings"""
    return {
        "use_sensitivity_adjustment": MODEL_CONFIG.get('use_sensitivity_adjustment', False),
        "glaucoma_sensitivity": MODEL_CONFIG.get('glaucoma_sensitivity', 0.05),
        "minimum_glaucoma_threshold": MODEL_CONFIG.get('minimum_glaucoma_threshold', 0.85),
        "status": "enabled" if MODEL_CONFIG.get('use_sensitivity_adjustment', False) else "disabled"
    }

@app.post("/sensitivity/settings")
async def update_sensitivity_settings(settings: dict):
    """Update sensitivity adjustment settings"""
    if 'use_sensitivity_adjustment' in settings:
        MODEL_CONFIG['use_sensitivity_adjustment'] = bool(settings['use_sensitivity_adjustment'])
    
    if 'glaucoma_sensitivity' in settings:
        # Clamp to reasonable range
        sensitivity = float(settings['glaucoma_sensitivity'])
        MODEL_CONFIG['glaucoma_sensitivity'] = max(0.0, min(0.2, sensitivity))  # Max 20% boost
    
    if 'minimum_glaucoma_threshold' in settings:
        # Clamp to reasonable range
        threshold = float(settings['minimum_glaucoma_threshold'])
        MODEL_CONFIG['minimum_glaucoma_threshold'] = max(0.5, min(0.95, threshold))
    
    logger.info(f"Updated sensitivity settings: {MODEL_CONFIG['use_sensitivity_adjustment']}, boost={MODEL_CONFIG['glaucoma_sensitivity']}, threshold={MODEL_CONFIG['minimum_glaucoma_threshold']}")
    
    return {
        "status": "updated",
        "use_sensitivity_adjustment": MODEL_CONFIG['use_sensitivity_adjustment'],
        "glaucoma_sensitivity": MODEL_CONFIG['glaucoma_sensitivity'],
        "minimum_glaucoma_threshold": MODEL_CONFIG['minimum_glaucoma_threshold']
    }

@app.post("/sensitivity/disable")
async def disable_sensitivity_adjustment():
    """Quickly disable sensitivity adjustment to get raw model predictions"""
    MODEL_CONFIG['use_sensitivity_adjustment'] = False
    logger.info("Sensitivity adjustment disabled - will use raw model predictions")
    
    return {
        "status": "disabled",
        "message": "Sensitivity adjustment turned off. API will now return raw model predictions."
    }

@app.post("/sensitivity/enable")
async def enable_sensitivity_adjustment():
    """Enable conservative sensitivity adjustment"""
    MODEL_CONFIG['use_sensitivity_adjustment'] = True
    logger.info("Sensitivity adjustment enabled with conservative settings")
    
    return {
        "status": "enabled",
        "message": "Conservative sensitivity adjustment enabled.",
        "settings": {
            "glaucoma_sensitivity": MODEL_CONFIG['glaucoma_sensitivity'],
            "minimum_glaucoma_threshold": MODEL_CONFIG['minimum_glaucoma_threshold']
        }
    }

# Crops management endpoints
@app.get("/crops/settings")
async def get_crop_settings():
    """Get current crop saving settings"""
    return {
        "save_crops": MODEL_CONFIG.get('save_crops', False),
        "crops_folder": MODEL_CONFIG.get('crops_folder', 'saved_crops'),
        "folder_exists": os.path.exists(MODEL_CONFIG.get('crops_folder', 'saved_crops'))
    }

@app.post("/crops/settings")
async def update_crop_settings(settings: dict):
    """Update crop saving settings"""
    if 'save_crops' in settings:
        MODEL_CONFIG['save_crops'] = bool(settings['save_crops'])
    
    if 'crops_folder' in settings:
        MODEL_CONFIG['crops_folder'] = str(settings['crops_folder'])
        # Create folder if it doesn't exist
        os.makedirs(MODEL_CONFIG['crops_folder'], exist_ok=True)
    
    logger.info(f"Updated crop settings: save_crops={MODEL_CONFIG['save_crops']}, folder={MODEL_CONFIG['crops_folder']}")
    
    return {
        "status": "updated",
        "save_crops": MODEL_CONFIG['save_crops'],
        "crops_folder": MODEL_CONFIG['crops_folder']
    }

@app.get("/crops/list")
async def list_saved_crops():
    """List all saved crop files"""
    crops_folder = MODEL_CONFIG.get('crops_folder', 'saved_crops')
    
    if not os.path.exists(crops_folder):
        return {"error": "Crops folder does not exist", "folder": crops_folder}
    
    files = []
    for filename in os.listdir(crops_folder):
        filepath = os.path.join(crops_folder, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            files.append({
                "filename": filename,
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
                "type": filename.split('_')[1] if '_' in filename else "unknown"
            })
    
    return {
        "crops_folder": crops_folder,
        "total_files": len(files),
        "files": sorted(files, key=lambda x: x['modified'], reverse=True)
    }

@app.delete("/crops/clear")
async def clear_saved_crops():
    """Clear all saved crop files"""
    crops_folder = MODEL_CONFIG.get('crops_folder', 'saved_crops')
    
    if not os.path.exists(crops_folder):
        return {"error": "Crops folder does not exist"}
    
    deleted_count = 0
    for filename in os.listdir(crops_folder):
        filepath = os.path.join(crops_folder, filename)
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting {filename}: {e}")
    
    return {
        "status": "cleared",
        "deleted_files": deleted_count,
        "folder": crops_folder
    }

@app.post("/cdr/overlay")
async def cdr_overlay(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run segmentation
    disc_mask, cup_mask, _ = segment_optic_disc_cup(image)

    # Compute CDR (with brightness fallback)
    img_np = np.array(image)
    cdr_info = calculate_cdr_from_masks(disc_mask, cup_mask, img_np)

    cdr_val = cdr_info.get("cdr_primary") or cdr_info.get("cdr_vertical")
    method  = cdr_info.get("method", "unknown")

    # Build overlay
    overlay_url = _create_overlay(image, disc_mask, cup_mask, cdr_val, method)

    return {
        "vertical_cdr": cdr_val,
        "method": method,
        "overlay_url": overlay_url
    }


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    # Run the API
    uvicorn.run(
        "main:app",  # Update with your filename
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )