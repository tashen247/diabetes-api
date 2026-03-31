import asyncio, io, os, re, logging, torch
from typing import Optional
import torch.nn.functional as F
import numpy as np
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from transformers import (
    AutoImageProcessor, SegformerForSemanticSegmentation,
    AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("medgem_local")

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEG_ID = "pamixsun/segformer_for_optic_disc_cup_segmentation"
VLM_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

# --- Model Store ---
_seg_model, _seg_proc = None, None
_vlm_model, _vlm_proc = None, None

def load_models():
    global _seg_model, _seg_proc, _vlm_model, _vlm_proc

    log.info("[*] Loading Local SegFormer...")
    _seg_proc  = AutoImageProcessor.from_pretrained(SEG_ID)
    _seg_model = SegformerForSemanticSegmentation.from_pretrained(SEG_ID).to(DEVICE).eval()

    log.info("[*] Loading Local LLaVA (4-bit)...")
    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    _vlm_proc  = AutoProcessor.from_pretrained(VLM_ID)
    _vlm_model = AutoModelForImageTextToText.from_pretrained(
        VLM_ID, quantization_config=q_config, device_map="auto"
    ).eval()


class SuperAnalysisReport(BaseModel):
    referral_id:      str
    real_cdr:         float
    geometric_cdr:    Optional[float]
    vlm_estimate:     float
    prediction:       str
    ads_id:           int
    decision_basis:   str
    thinning_detected: bool
    thinning_location: Optional[str] = "None"
    analysis:         str


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: VLM CDR extraction — broader regex covering prose patterns
# Original only matched: "(?:estimated_cdr|cdr)\D*(\d\.\d+)"
# LLaVA writes things like:
#   "CDR appears to be around 0.85"
#   "cup-to-disc ratio of approximately 0.7"
#   "estimated CDR: 0.65"
#   "vertical CDR is 0.6"
# ─────────────────────────────────────────────────────────────────────────────
def extract_cdr_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.lower()

    # Pattern A: explicit key=value  (cdr: 0.7, estimated_cdr=0.85)
    m = re.search(r"(?:estimated_cdr|vertical.cdr|cdr)\s*[=:]\s*(\d+\.?\d*)", t)
    if m:
        val = float(m.group(1))
        return round(val, 3) if 0.0 < val <= 1.0 else None

    # Pattern B: "around / approximately / about / of 0.X"
    m = re.search(
        r"(?:around|approximately|about|of|ratio\s+of|ratio\s+is|appears?\s+to\s+be)\s+(\d+\.?\d*)",
        t
    )
    if m:
        val = float(m.group(1))
        return round(val, 3) if 0.0 < val <= 1.0 else None

    # Pattern C: bare decimal after any CDR-related phrase
    m = re.search(
        r"(?:cup.to.disc|disc.to.cup|cdr|cup.disc)\s+(?:ratio\s+)?(?:of\s+)?(?:is\s+)?(\d+\.?\d*)",
        t
    )
    if m:
        val = float(m.group(1))
        return round(val, 3) if 0.0 < val <= 1.0 else None

    # Pattern D: any standalone decimal in [0.3, 0.95] near disc/cup/glaucoma words
    context_words = ["disc", "cup", "glaucoma", "optic", "cdr", "ratio"]
    if any(w in t for w in context_words):
        for m in re.finditer(r"\b(0\.\d{1,2})\b", t):
            val = float(m.group(1))
            if 0.3 <= val <= 0.95:
                return round(val, 3)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: Thinning detection — semantic matching on prose, not exact string
# Original: "rim_thinning: yes" in vlm_res.lower()  ← LLaVA never writes this
# ─────────────────────────────────────────────────────────────────────────────
_THINNING_PHRASES = [
    r"rim\s+thinning",
    r"thinning\s+of\s+the\s+(?:rim|neuroretinal|disc)",
    r"neuroretinal\s+rim\s+(?:loss|thinning|defect)",
    r"inferior\s+(?:rim\s+)?thinning",
    r"superior\s+(?:rim\s+)?thinning",
    r"temporal\s+(?:rim\s+)?thinning",
    r"nasal\s+(?:rim\s+)?thinning",
    r"notching",
    r"disc\s+notch",
    r"focal\s+notch",
    r"rim\s+defect",
    r"loss\s+of\s+(?:the\s+)?rim",
]
_THINNING_RE = re.compile("|".join(_THINNING_PHRASES), re.IGNORECASE)

_QUADRANT_MAP = {
    "superior":  re.compile(r"\bsuperior\b",  re.IGNORECASE),
    "inferior":  re.compile(r"\binferior\b",  re.IGNORECASE),
    "temporal":  re.compile(r"\btemporal\b",  re.IGNORECASE),
    "nasal":     re.compile(r"\bnasal\b",     re.IGNORECASE),
}

# Negation words that indicate absence of thinning
_NEGATION_PRE  = re.compile(
    r"\b(no|not|none|without|absent|absence|negative|normal|clear|intact)\b",
    re.IGNORECASE
)
_NEGATION_POST = re.compile(
    r"^\s*(?:is\s+)?(?:[:]\s*)?(no|not|none|absent|absence|"
    r"not\s+present|not\s+detected|not\s+observed|normal|clear|intact)\b",
    re.IGNORECASE
)

def detect_thinning(text: str):
    """
    Returns (thinning_detected: bool, location: str).
    Searches prose for rim-thinning language with negation guards.
    Handles: "No rim thinning detected", "Rim thinning is absent",
             "rim thinning: no", "not present" etc.
    """
    if not text:
        return False, "None"

    m = _THINNING_RE.search(text)
    if not m:
        return False, "None"

    # Check negation in 60 chars BEFORE the match
    pre_window  = text[max(0, m.start()-60): m.start()]
    # Check negation in 50 chars AFTER the match (catches "is absent", ": no")
    post_window = text[m.end(): m.end()+50]

    if _NEGATION_PRE.search(pre_window) or _NEGATION_POST.search(post_window):
        log.info(f"[thinning] Negated — pre={repr(pre_window[-30:])}, "
                 f"post={repr(post_window[:30])}")
        return False, "None"

    # Find quadrant within 120 chars of the thinning phrase
    window = text[max(0, m.start()-80): m.end()+80]
    for quadrant, pattern in _QUADRANT_MAP.items():
        if pattern.search(window):
            return True, quadrant.capitalize()

    return True, "Unknown"


def run_geometric_cdr(image_rgb: np.ndarray, referral_id: str = "0") -> Optional[float]:
    """SegFormer geometric CDR with sensitivity boost on cup channel."""
    inputs = _seg_proc(image_rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs  = _seg_model(**inputs)
        logits   = outputs.logits
        logits[:, 2, :, :] += 2.5          # cup sensitivity boost
        up_logits = F.interpolate(logits, size=image_rgb.shape[:2], mode="bilinear")
        pred = up_logits.argmax(dim=1)[0].cpu().numpy()

    d_y = np.where(pred == 1)[0]
    c_y = np.where(pred == 2)[0]

    geo_cdr = None
    if len(d_y) > 0 and len(c_y) > 0:
        disc_h = d_y.max() - d_y.min()
        cup_h  = c_y.max() - c_y.min()
        if disc_h > 0:
            raw = float(cup_h / disc_h)
            # Clamp to plausible range
            if 0.0 < raw <= 1.0:
                geo_cdr = round(raw, 3)
            else:
                log.warning(f"[geo_cdr] Implausible raw CDR {raw:.3f} — rejected")

    if geo_cdr is not None:
        debug_img = image_rgb.copy()
        debug_img[pred == 1] = [0, 255, 0]
        debug_img[pred == 2] = [0, 0, 255]
        cv2.imwrite(
            f"debug_{referral_id}.jpg",
            cv2.addWeighted(image_rgb, 0.7, debug_img, 0.3, 0)
        )

    return geo_cdr


# --- API ---
app = FastAPI(title="Super Glaucoma Analyzer", version="2.0.0")

@app.on_event("startup")
async def startup():
    load_models()


@app.post("/super-analyze", response_model=SuperAnalysisReport)
async def super_analyze(file: UploadFile = File(...), referral_id: str = Form("0")):
    data    = await file.read()
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    pil_img.save(f"incoming_image_{referral_id}.jpg")
    img_np  = np.array(pil_img)

    # ── 1. Geometric CDR (SegFormer) ──────────────────────────────────────────
    geo_cdr = run_geometric_cdr(img_np, referral_id)
    log.info(f"[geo] CDR={geo_cdr}")

    # ── 2. VLM Reasoning (LLaVA) ─────────────────────────────────────────────
    prompt = (
        "[INST] <image>\n"
        "You are an ophthalmologist reviewing a fundus photograph. "
        "Please do the following:\n"
        "1. Estimate the vertical cup-to-disc ratio (CDR) as a decimal between 0.1 and 1.0. "
        "State it clearly, e.g. 'The estimated CDR is 0.7'.\n"
        "2. State whether there is rim thinning and in which quadrant "
        "(Superior, Inferior, Nasal, Temporal). "
        "e.g. 'Rim thinning is present in the inferior quadrant.' or 'No rim thinning detected.'\n"
        "Be concise. [/INST]"
    )

    inputs = _vlm_proc(text=prompt, images=pil_img, return_tensors="pt").to(DEVICE)
    log.info(f"[vlm] pixel_values shape={inputs['pixel_values'].shape}, "
             f"mean={inputs['pixel_values'].mean().item():.4f}")

    with torch.no_grad():
        output = _vlm_model.generate(
            **inputs,
            max_new_tokens=300,        # FIX: was 100 — too short for LLaVA to finish
            do_sample=True,
            temperature=0.6,           # slightly tighter than 0.7
            top_p=0.92,
            repetition_penalty=1.2,
            pad_token_id=_vlm_proc.tokenizer.eos_token_id,
        )
        vlm_res = _vlm_proc.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

    log.info(f"[vlm] response: {vlm_res[:200]}")

    # ── 3. Extract CDR + thinning from VLM prose ─────────────────────────────
    vlm_cdr = extract_cdr_from_text(vlm_res) or 0.0
    thinning_detected, thinning_loc = detect_thinning(vlm_res)   # FIX 2
    log.info(f"[vlm] CDR={vlm_cdr}, thinning={thinning_detected} ({thinning_loc})")

    # ── 4. Fuse geometric + VLM ───────────────────────────────────────────────
    # Priority: geometric CDR (SegFormer) is more reliable than VLM estimate.
    # VLM is used ONLY as fallback when SegFormer fails (geo_cdr is None).
    # When both are available, use geo_cdr — VLM can hallucinate CDR values
    # on difficult images (e.g. off-centre disc, warm-tinted, low contrast).
    if geo_cdr is not None:
        real_cdr = geo_cdr   # geometric is primary
        log.info(f"[fuse] Using geometric CDR={geo_cdr:.3f} (VLM={vlm_cdr:.3f} ignored)")
    elif vlm_cdr > 0:
        real_cdr = vlm_cdr   # VLM fallback when SegFormer failed
        log.info(f"[fuse] geo_cdr unavailable — using VLM CDR={vlm_cdr:.3f}")
    else:
        real_cdr = 0.0
        log.warning("[fuse] Both geo and VLM CDR unavailable — defaulting to 0.0")

    # ── 5. Decision ladder ────────────────────────────────────────────────────
    LOW_SUSPECT  = 0.55
    HIGH_SUSPECT = 0.70

    if real_cdr >= HIGH_SUSPECT:
        final_pred = "Glaucoma Suspicious (High CDR)"
        final_ads  = 17
        basis      = "CDR_HIGH_RISK"
    elif real_cdr >= LOW_SUSPECT:
        final_pred = "Glaucoma Suspicious (Borderline CDR)"
        final_ads  = 17
        basis      = "CDR_BORDERLINE"
    elif thinning_detected:
        final_pred = f"Glaucoma Suspicious ({thinning_loc} Thinning)"
        final_ads  = 17
        basis      = "RIM_THINNING_OVERRIDE"
    else:
        final_pred = "Normal"
        final_ads  = 16
        basis      = "MODEL"

    log.info(f"[decision] real_cdr={real_cdr:.3f}, pred={final_pred}, ads_id={final_ads}")

    return SuperAnalysisReport(
        referral_id       = referral_id,
        real_cdr          = real_cdr,
        geometric_cdr     = geo_cdr,
        vlm_estimate      = vlm_cdr,
        prediction        = final_pred,
        ads_id            = final_ads,
        decision_basis    = basis,
        thinning_detected = thinning_detected,
        thinning_location = thinning_loc if thinning_detected else "None",
        analysis          = vlm_res.strip(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8025)
