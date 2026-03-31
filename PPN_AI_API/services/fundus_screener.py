"""
Fundus Multi-Condition Screener — Port 8027 (v2)
=================================================
Refactored from v1 with:
  - Per-condition focused prompts (LLaVA looks for specific clinical features)
  - Positive-only override (LLaVA can only upgrade, never confirm negative)
  - Separate VLM inference per condition (no more lazy all-clear JSON)
  - Confidence normalisation (rejects uniform 0.9 responses as unreliable)

Endpoints:
  POST /screen          — single eye, all conditions
  POST /screen-pair     — both eyes with bilateral promotion
  GET  /thresholds      — view/update override thresholds
  GET  /health
"""

import io, os, re, logging, json
from typing import Optional, Dict, Any, Tuple
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
log = logging.getLogger("fundus_screener")

# ─────────────────────────────────────────────────────────────────────────────
# ADS MAP
# ─────────────────────────────────────────────────────────────────────────────
ADS: Dict[str, Dict[str, int]] = {
    "diabetes": {
        "No Suspect or Negative":                   2,
        "Mild":                                     3,
        "Moderate":                                 4,
        "Severe":                                   5,
        "Proliferative Diabetic Retinopathy (PDR)": 6,
    },
    "hypertension": {
        "No Suspect or Negative":                   7,
        "Decreased Retinal Artery Elasticity":      8,
        "Hypertensive Retinopathy Grade 1 or 2":    9,
        "Hypertensive Retinopathy Grade 3 or 4":   10,
    },
    "macular_degeneration": {
        "No Suspect or Negative":                  11,
        "Drusen":                                  12,
        "AMD Early or Intermediate Stage":         13,
        "AMD Advanced Stage":                      14,
        "Large Drusen with Abnormal Pigmentation": 15,
    },
    "cnv": {
        "No Suspect or Negative":                  22,
        "Suspect or Positive":                     23,
    },
    "rvo": {
        "No Suspect or Negative":                  24,
        "Suspect or Positive":                     25,
    },
    "pathological": {
        "No Suspect or Negative":                  18,
        "Minor Pathological":                      19,
        "Obvious Pathological":                    20,
    },
}

# Negative label for each condition (first entry = negative)
NEG_LABEL = {c: list(labels.keys())[0] for c, labels in ADS.items()}
NEG_ADS   = {c: list(labels.values())[0] for c, labels in ADS.items()}

# ─────────────────────────────────────────────────────────────────────────────
# OVERRIDE THRESHOLDS
# Only applies when LLaVA returns a POSITIVE finding.
# LLaVA cannot override an existing positive result with a negative.
# ─────────────────────────────────────────────────────────────────────────────
OVERRIDE_THRESHOLDS: Dict[str, float] = {
    "diabetes":             float(os.getenv("THR_DIABETES", "0.70")),
    "hypertension":         float(os.getenv("THR_HTN",      "0.72")),
    "macular_degeneration": float(os.getenv("THR_AMD",      "0.72")),
    "cnv":                  float(os.getenv("THR_CNV",      "0.75")),
    "rvo":                  float(os.getenv("THR_RVO",      "0.72")),
    "pathological":         float(os.getenv("THR_PATHO",    "0.65")),
}

BILATERAL_THRESHOLD = float(os.getenv("THR_BILATERAL", "0.80"))

# ─────────────────────────────────────────────────────────────────────────────
# PER-CONDITION FOCUSED PROMPTS
# Each prompt asks LLaVA to look for the specific clinical signs of that
# condition. Returns a small focused JSON — not a 6-condition sweep.
# ─────────────────────────────────────────────────────────────────────────────
PROMPTS: Dict[str, str] = {

    "diabetes": """[INST] <image>
You are an expert ophthalmologist. Examine this fundus photograph specifically for signs of DIABETIC RETINOPATHY.

Look carefully for:
- Microaneurysms (tiny red dots)
- Dot and blot haemorrhages
- Hard exudates (bright yellow-white deposits)
- Cotton wool spots (fluffy white patches)
- Venous beading or loops
- Neovascularisation (new vessel growth on disc or elsewhere)
- Vitreous haemorrhage or fibrous proliferation

Based only on what you can see, respond with ONLY this JSON (no other text):
{
  "grade": "<No Suspect or Negative | Mild | Moderate | Severe | Proliferative Diabetic Retinopathy (PDR)>",
  "confidence": <0.0-1.0>,
  "findings": "<specific features you can see>"
}
[/INST]""",

    "hypertension": """[INST] <image>
You are an expert ophthalmologist. Examine this fundus photograph specifically for signs of HYPERTENSIVE RETINOPATHY.

Look carefully for:
- Arteriovenous nicking (AV nipping at crossings)
- Arteriolar narrowing or silver/copper wiring
- Flame-shaped haemorrhages
- Cotton wool spots
- Hard exudates in a macular star pattern
- Disc oedema (papilloedema)
- Generalised arteriolar attenuation

Based only on what you can see, respond with ONLY this JSON (no other text):
{
  "grade": "<No Suspect or Negative | Decreased Retinal Artery Elasticity | Hypertensive Retinopathy Grade 1 or 2 | Hypertensive Retinopathy Grade 3 or 4>",
  "confidence": <0.0-1.0>,
  "findings": "<specific features you can see>"
}
[/INST]""",

    "macular_degeneration": """[INST] <image>
You are an expert ophthalmologist. Examine this fundus photograph specifically for signs of AGE-RELATED MACULAR DEGENERATION (AMD).

Look carefully at the macular region (centre of image) for:
- Drusen (small yellow-white deposits under the retina)
- Pigment epithelial changes (hyper or hypopigmentation)
- Geographic atrophy (sharply defined areas of RPE loss)
- Subretinal fluid or haemorrhage (wet AMD)
- Choroidal neovascular membrane
- Disciform scar

Based only on what you can see, respond with ONLY this JSON (no other text):
{
  "grade": "<No Suspect or Negative | Drusen | AMD Early or Intermediate Stage | AMD Advanced Stage | Large Drusen with Abnormal Pigmentation>",
  "confidence": <0.0-1.0>,
  "findings": "<specific features you can see>"
}
[/INST]""",

    "cnv": """[INST] <image>
You are an expert ophthalmologist. Examine this fundus photograph specifically for signs of CHOROIDAL NEOVASCULARISATION (CNV).

Look carefully for:
- Subretinal or sub-RPE haemorrhage
- Grey-green membrane beneath the retina
- Subretinal fluid or exudate around the macula
- Pigment epithelial detachment (dome-shaped elevation)
- Disciform scar (white fibrous tissue at macula)
- Surrounding hard exudates

Based only on what you can see, respond with ONLY this JSON (no other text):
{
  "grade": "<No Suspect or Negative | Suspect or Positive>",
  "confidence": <0.0-1.0>,
  "findings": "<specific features you can see>"
}
[/INST]""",

    "rvo": """[INST] <image>
You are an expert ophthalmologist. Examine this fundus photograph specifically for signs of RETINAL VEIN OCCLUSION (RVO).

Look carefully for:
- Flame-shaped haemorrhages in one or more quadrants
- Dilated and tortuous retinal veins
- Disc swelling or hyperaemia
- Cotton wool spots
- Macular oedema
- For branch RVO: haemorrhages confined to one sector
- For central RVO: haemorrhages in all four quadrants

Based only on what you can see, respond with ONLY this JSON (no other text):
{
  "grade": "<No Suspect or Negative | Suspect or Positive>",
  "confidence": <0.0-1.0>,
  "findings": "<specific features you can see>"
}
[/INST]""",

    "pathological": """[INST] <image>
You are an expert ophthalmologist. Examine this fundus photograph for any PATHOLOGICAL or ABNORMAL findings not explained by a normal fundus.

Look carefully for:
- Subretinal fluid or exudates
- Retinal detachment or traction
- Choroidal lesions or masses
- Large areas of chorioretinal atrophy
- Significant retinal pigment changes
- White or grey patches suggesting ischaemia or infiltration
- Any finding that would not be present in a completely healthy eye

Based only on what you can see, respond with ONLY this JSON (no other text):
{
  "grade": "<No Suspect or Negative | Minor Pathological | Obvious Pathological>",
  "confidence": <0.0-1.0>,
  "findings": "<specific abnormal features you can see>"
}
[/INST]""",
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
VLM_ID     = "llava-hf/llava-v1.6-mistral-7b-hf"
_vlm_proc  = None
_vlm_model = None


def load_model():
    global _vlm_proc, _vlm_model
    log.info("[*] Loading LLaVA (4-bit)...")
    q = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    _vlm_proc  = AutoProcessor.from_pretrained(VLM_ID)
    _vlm_model = AutoModelForImageTextToText.from_pretrained(
        VLM_ID, quantization_config=q, device_map="auto"
    ).eval()
    log.info("[*] LLaVA loaded.")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _run_vlm_single(pil_img: Image.Image, prompt: str) -> str:
    """Run one focused prompt against the image."""
    inputs = _vlm_proc(
        text=prompt,
        images=pil_img,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = _vlm_model.generate(
            **inputs,
            max_new_tokens=200,       # focused prompt needs less tokens
            do_sample=False,          # greedy — consistent JSON
            pad_token_id=_vlm_proc.tokenizer.eos_token_id,
        )

    return _vlm_proc.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()


def _parse_single(raw: str, condition: str) -> Optional[Dict]:
    """Parse a focused single-condition JSON response."""
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    for attempt in (text, re.search(r"\{.*\}", text, re.DOTALL)):
        try:
            candidate = json.loads(attempt if isinstance(attempt, str) else attempt.group(0))
            if "grade" in candidate:
                return candidate
        except Exception:
            continue

    # Try repairing truncated JSON
    try:
        open_b  = text.count("{")
        close_b = text.count("}")
        if open_b > close_b:
            repaired = text + "}" * (open_b - close_b)
            candidate = json.loads(repaired)
            if "grade" in candidate:
                return candidate
    except Exception:
        pass

    log.warning(f"[{condition}] Could not parse: {raw[:200]}")
    return None


def _is_uniform_confidence(results: Dict[str, Dict]) -> bool:
    """
    Detect lazy uniform-confidence responses (e.g. all 0.9).
    Returns True if all conditions have the exact same confidence — sign of
    LLaVA not actually examining the image.
    """
    confs = [r.get("confidence", 0.0) for r in results.values() if r]
    if len(confs) < 2:
        return False
    return len(set(round(c, 2) for c in confs)) == 1


def _resolve_grade(condition: str, raw_grade: str) -> Tuple[str, int]:
    """
    Match raw VLM grade string to known label. Returns (label, ads_id).
    Falls back to negative if unrecognised.
    """
    ads_map = ADS[condition]

    # Exact match
    if raw_grade in ads_map:
        return raw_grade, ads_map[raw_grade]

    # Case-insensitive substring match
    grade_lower = raw_grade.lower()
    for label, aid in ads_map.items():
        if label.lower() in grade_lower or grade_lower in label.lower():
            return label, aid

    # Default to negative
    neg = NEG_LABEL[condition]
    log.warning(f"[{condition}] Unrecognised grade '{raw_grade}' → defaulting to '{neg}'")
    return neg, NEG_ADS[condition]


def screen_image(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Run all 6 focused prompts against one image.
    Returns per-condition results dict.
    """
    results = {}

    for condition, prompt in PROMPTS.items():
        raw    = _run_vlm_single(pil_img, prompt)
        parsed = _parse_single(raw, condition)
        log.info(f"[{condition}] raw: {raw[:120]}")

        if parsed is None:
            results[condition] = {
                "grade":      NEG_LABEL[condition],
                "ads_id":     NEG_ADS[condition],
                "confidence": 0.0,
                "findings":   "Parse failed",
                "raw":        raw[:200],
                "parse_failed": True,
            }
            continue

        grade, ads_id = _resolve_grade(condition, str(parsed.get("grade", "")))
        confidence    = min(1.0, max(0.0, float(parsed.get("confidence", 0.0))))
        findings      = str(parsed.get("findings", ""))

        results[condition] = {
            "grade":      grade,
            "ads_id":     ads_id,
            "confidence": round(confidence, 3),
            "findings":   findings,
            "raw":        raw[:300],
        }

    # Flag uniform confidence as unreliable
    if _is_uniform_confidence(results):
        log.warning("[screen] Uniform confidence detected — marking results as low reliability")
        for r in results.values():
            r["low_reliability"] = True

    return results


def apply_override(
    condition:    str,
    llava:        Dict,
    existing_ads: Optional[int],
) -> Dict[str, Any]:
    """
    POSITIVE-ONLY override:
    LLaVA can only upgrade (negative → positive).
    LLaVA can NEVER override an existing positive with a negative.
    LLaVA can NEVER confirm a negative with high confidence to suppress existing result.
    """
    threshold  = OVERRIDE_THRESHOLDS.get(condition, 0.75)
    llava_ads  = llava["ads_id"]
    llava_conf = llava["confidence"]
    neg_ads    = NEG_ADS[condition]

    # Reject low-reliability responses
    if llava.get("low_reliability") or llava.get("parse_failed"):
        return {
            "overridden":   False,
            "reason":       "low_reliability_or_parse_failed",
            "final_ads_id": existing_ads if existing_ads is not None else llava_ads,
            "final_grade":  llava["grade"],
        }

    # SAFETY: never downgrade — if existing is positive, keep it
    existing_is_positive = existing_ads is not None and existing_ads != neg_ads
    llava_is_negative    = llava_ads == neg_ads

    if existing_is_positive and llava_is_negative:
        return {
            "overridden":   False,
            "reason":       "safety_no_downgrade",
            "final_ads_id": existing_ads,
            "final_grade":  llava["grade"],
        }

    # POSITIVE OVERRIDE: LLaVA found something + confidence >= threshold
    if llava_ads != neg_ads and llava_conf >= threshold:
        return {
            "overridden":   True,
            "reason":       f"positive_finding: conf={llava_conf:.2f} >= {threshold}",
            "final_ads_id": llava_ads,
            "final_grade":  llava["grade"],
        }

    # No override
    final_ads = existing_ads if existing_ads is not None else llava_ads
    return {
        "overridden":   False,
        "reason":       "below_threshold" if llava_conf < threshold else "no_positive_finding",
        "final_ads_id": final_ads,
        "final_grade":  llava["grade"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fundus Multi-Condition Screener v2",
    description="Per-condition focused LLaVA prompts. Positive-only override.",
    version="2.0.0",
)


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": _vlm_model is not None,
        "conditions":   list(PROMPTS.keys()),
    }


@app.get("/thresholds")
async def get_thresholds():
    return {"thresholds": OVERRIDE_THRESHOLDS, "bilateral": BILATERAL_THRESHOLD}


@app.post("/thresholds")
async def set_thresholds(body: dict):
    global BILATERAL_THRESHOLD
    for k, v in body.items():
        if k == "bilateral":
            BILATERAL_THRESHOLD = float(v)
        elif k in OVERRIDE_THRESHOLDS:
            OVERRIDE_THRESHOLDS[k] = float(v)
    return {"thresholds": OVERRIDE_THRESHOLDS, "bilateral": BILATERAL_THRESHOLD}


@app.post("/screen")
async def screen(
    file:             UploadFile = File(...),
    referral_id:      str        = Form("0"),
    # Optional existing ads_ids per condition: {"diabetes": 2, "hypertension": 7, ...}
    existing_ads_ids: str        = Form("{}"),
):
    """
    Screen a single eye for all 6 conditions using focused per-condition prompts.
    Pass existing_ads_ids (JSON dict) to enable positive-only override logic.
    """
    if _vlm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data    = await file.read()
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")

    try:
        existing = json.loads(existing_ads_ids) if existing_ads_ids else {}
    except Exception:
        existing = {}

    log.info(f"[screen] referral={referral_id} file={file.filename}")
    llava_results = screen_image(pil_img)

    conditions = {}
    flags      = {}

    for condition, llava in llava_results.items():
        existing_ads = existing.get(condition)
        override     = apply_override(condition, llava, existing_ads)

        conditions[condition] = {
            "llava":           {"grade": llava["grade"], "ads_id": llava["ads_id"],
                                "confidence": llava["confidence"], "findings": llava["findings"]},
            "override":        override["overridden"],
            "override_reason": override["reason"],
            "final":           {"ads_id": override["final_ads_id"], "prediction": override["final_grade"]},
        }

        if override["final_ads_id"] != NEG_ADS[condition]:
            flags[condition] = conditions[condition]["final"]

    return JSONResponse(content={
        "referral_id": referral_id,
        "filename":    file.filename,
        "conditions":  conditions,
        "flags":       flags,
        "any_positive": len(flags) > 0,
    })


@app.post("/screen-pair")
async def screen_pair(
    left_eye_file:    UploadFile = File(...),
    right_eye_file:   UploadFile = File(...),
    referral_id:      str        = Form("0"),
    # Optional: {"left": {"diabetes": 2}, "right": {"diabetes": 3}}
    existing_ads_ids: str        = Form("{}"),
):
    """
    Screen both eyes with bilateral promotion.
    If either eye flags a condition at >= BILATERAL_THRESHOLD, promotes the other eye.
    """
    if _vlm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        existing = json.loads(existing_ads_ids) if existing_ads_ids else {}
    except Exception:
        existing = {}

    def _screen_one(upload_data: bytes, filename: str, eye: str) -> Dict:
        pil_img = Image.open(io.BytesIO(upload_data)).convert("RGB")
        log.info(f"[screen-pair] {eye} eye: referral={referral_id} file={filename}")
        llava_results = screen_image(pil_img)
        existing_eye  = existing.get(eye, {})
        out = {}
        for condition, llava in llava_results.items():
            existing_ads = existing_eye.get(condition)
            override     = apply_override(condition, llava, existing_ads)
            out[condition] = {
                "llava":           {"grade": llava["grade"], "ads_id": llava["ads_id"],
                                    "confidence": llava["confidence"], "findings": llava["findings"]},
                "override":        override["overridden"],
                "override_reason": override["reason"],
                "final":           {"ads_id": override["final_ads_id"], "prediction": override["final_grade"]},
            }
        return out

    left_data  = await left_eye_file.read()
    right_data = await right_eye_file.read()

    left_conditions  = _screen_one(left_data,  left_eye_file.filename,  "left")
    right_conditions = _screen_one(right_data, right_eye_file.filename, "right")

    # ── Bilateral promotion ───────────────────────────────────────────────────
    bilateral_changes = []
    for condition in ADS.keys():
        neg_ads = NEG_ADS[condition]
        l       = left_conditions[condition]
        r       = right_conditions[condition]
        l_conf  = l["llava"]["confidence"]
        r_conf  = r["llava"]["confidence"]
        l_ads   = l["final"]["ads_id"]
        r_ads   = r["final"]["ads_id"]

        # Left positive + high conf → promote right
        if l_ads != neg_ads and l_conf >= BILATERAL_THRESHOLD and r_ads == neg_ads:
            right_conditions[condition]["final"]               = l["final"]
            right_conditions[condition]["bilateral_promoted"]  = True
            bilateral_changes.append(
                f"{condition}: right promoted from left (conf={l_conf:.2f})"
            )

        # Right positive + high conf → promote left
        elif r_ads != neg_ads and r_conf >= BILATERAL_THRESHOLD and l_ads == neg_ads:
            left_conditions[condition]["final"]              = r["final"]
            left_conditions[condition]["bilateral_promoted"] = True
            bilateral_changes.append(
                f"{condition}: left promoted from right (conf={r_conf:.2f})"
            )

    left_flags  = {c: left_conditions[c]["final"]  for c in ADS if left_conditions[c]["final"]["ads_id"]  != NEG_ADS[c]}
    right_flags = {c: right_conditions[c]["final"] for c in ADS if right_conditions[c]["final"]["ads_id"] != NEG_ADS[c]}

    return JSONResponse(content={
        "referral_id":       referral_id,
        "left":              {"conditions": left_conditions},
        "right":             {"conditions": right_conditions},
        "bilateral_changes": bilateral_changes,
        "left_flags":        left_flags,
        "right_flags":       right_flags,
        "any_positive":      len(left_flags) > 0 or len(right_flags) > 0,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8027)
