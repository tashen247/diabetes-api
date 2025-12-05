# hypertension_service1.py
# ------------------------------------------------------------
# Hypertension microservice v3.6.1 (ultra-conservative DRAE)
# ------------------------------------------------------------
import os, io, json
from typing import Dict, Tuple
import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageFile
# ============================================================
# CONFIG
# ============================================================
SERVICE_VERSION = "3.6.1"
MODEL_VERSION = "DRAE_TUNED_0.45+0.38_MARGIN0.12"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
ImageFile.LOAD_TRUNCATED_IMAGES = True
SEVERITY_WEIGHTS = r"C:\source_controls\PPN_AI_API\weights\hypertension_multiclass_refined_DRAE.pt"
SEVERITY_CLASSES_JSON = r"C:\source_controls\PPN_AI_API\weights\hypertension_classes.json"
CALIB_JSON = r"C:\source_controls\PPN_AI_API\weights\drae_thresholds.json"
ADS_ID = {
   "No Suspect or Negative": 7,
   "Decreased Retinal Artery Elasticity": 8,
   "Hypertensive Retinopathy Grade 1 or 2": 9,
   "Hypertensive Retinopathy Grade 3 or 4": 10,
}
# ============================================================
# THRESHOLDS
# ============================================================
DRAE_FORCE = 0.45
BORDERLINE_DRAE_FORCE = {"min_drae": 0.38, "min_margin_over_no": 0.12}
NO_RULE = {"min_no": 0.65, "max_drae": 0.25, "max_hr12": 0.2, "max_hr34": 0.1}
HR34_SUPPRESS = {"min_hr34": 0.6, "max_no": 0.25, "max_drae": 0.05, "max_hr12": 0.1}
if os.path.exists(CALIB_JSON):
   try:
       with open(CALIB_JSON, "r") as f:
           c = json.load(f)
       DRAE_FORCE = float(c.get("DRAE_FORCE", DRAE_FORCE))
       if "BORDERLINE" in c:
           BORDERLINE_DRAE_FORCE.update(c["BORDERLINE"])
       if "NO_RULE" in c:
           NO_RULE.update(c["NO_RULE"])
       if "HR34_SUPPRESS" in c:
           HR34_SUPPRESS.update(c["HR34_SUPPRESS"])
       print("✅ Loaded calibrated thresholds:")
   except Exception as e:
       print(f"⚠️ Failed to load calibration JSON: {e}")
else:
   print("⚠️ No calibration JSON found, using defaults.")
# ============================================================
# MODEL
# ============================================================
def build_resnet_head(nc):
   m = models.resnet50(weights=None)
   m.fc = nn.Linear(m.fc.in_features, nc)
   return m
def load_class_map(path):
   with open(path, "r") as f:
       rows = sorted(json.load(f), key=lambda r: r["index"])
   labels = [r["folder"] for r in rows]
   ads_ids = [r["ads_id"] for r in rows]
   return labels, ads_ids
def load_severity_model(wpath, cjson):
   labels, ads_ids = load_class_map(cjson)
   model = build_resnet_head(len(labels))
   ckpt = torch.load(wpath, map_location="cpu")
   model.load_state_dict(ckpt["state_dict"], strict=False)
   model.to(DEVICE).eval()
   return model, labels, ads_ids
SEVERITY_MODEL, SEV_LABELS, SEV_ADS = load_severity_model(SEVERITY_WEIGHTS, SEVERITY_CLASSES_JSON)
# ============================================================
# TRANSFORMS & UTILS
# ============================================================
pre_tf = T.Compose([
   T.Resize(int(IMG_SIZE * 1.14)),
   T.CenterCrop(IMG_SIZE),
   T.ToTensor(),
   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
def load_image_from_bytes(data: bytes):
   try:
       img = Image.open(io.BytesIO(data)).convert("RGB")
       img.load()
       return img
   except Exception as e:
       raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
def _dominant_label(pm): return max(pm.items(), key=lambda kv: kv[1])[0]
def normalize_label(label: str) -> str:
   map_dict = {
       "No": "No Suspect or Negative",
       "No Suspect": "No Suspect or Negative",
       "NoSuspect": "No Suspect or Negative",
       "DecreasedRetinalArteryElasticity": "Decreased Retinal Artery Elasticity",
       "HypertensiveRetinopathyGrade1or2": "Hypertensive Retinopathy Grade 1 or 2",
       "HypertensiveRetinopathyGrade3or4": "Hypertensive Retinopathy Grade 3 or 4",
   }
   return map_dict.get(label.strip(), label.strip())
# ============================================================
# FINAL OVERRIDE LOGIC
# ============================================================
def apply_final_override(pm: Dict[str, float]) -> Tuple[Dict, Dict]:
    """

    Ultra-conservative override for hypertension grading.

    Only classifies DRAE if DRAE >= 0.90 probability.

    Otherwise defaults to 'No Suspect or Negative'.

    """

    pm = {normalize_label(k): v for k, v in pm.items()}

    p_no = pm.get("No Suspect or Negative", 0)

    p_drae = pm.get("Decreased Retinal Artery Elasticity", 0)

    p_hr12 = pm.get("Hypertensive Retinopathy Grade 1 or 2", 0)

    p_hr34 = pm.get("Hypertensive Retinopathy Grade 3 or 4", 0)

    debug = {"probabilities": pm, "logic_path": None}

    # 1️⃣ Strong Stage 3 suppression

    if (

            p_hr34 >= HR34_SUPPRESS["min_hr34"]

            and p_no <= HR34_SUPPRESS["max_no"]

            and p_drae <= HR34_SUPPRESS["max_drae"]

            and p_hr12 <= HR34_SUPPRESS["max_hr12"]

    ):
        debug["logic_path"] = "SUPPRESS_FALSE_STAGE3"

        return {"ads_id": ADS_ID["No Suspect or Negative"], "label": "No Suspect or Negative"}, debug

    # 2️⃣ Force No if clearly normal

    if p_no >= 0.40:
        debug["logic_path"] = "FORCE_NO_IF_ABOVE_0.40"

        return {"ads_id": ADS_ID["No Suspect or Negative"], "label": "No Suspect or Negative"}, debug

    # 3️⃣ No-rule satisfied

    if (

            p_no >= NO_RULE["min_no"]

            and p_drae < NO_RULE["max_drae"]

            and p_hr12 < NO_RULE["max_hr12"]

            and p_hr34 < NO_RULE["max_hr34"]

    ):
        debug["logic_path"] = "NO_RULE_SATISFIED"

        return {"ads_id": ADS_ID["No Suspect or Negative"], "label": "No Suspect or Negative"}, debug

    # 4️⃣ Hypertensive thresholds

    if p_hr34 >= 0.6:
        debug["logic_path"] = "HR34_THRESHOLD"

        return {"ads_id": ADS_ID["Hypertensive Retinopathy Grade 3 or 4"],
                "label": "Hypertensive Retinopathy Grade 3 or 4"}, debug

    if p_hr12 >= 0.55:
        debug["logic_path"] = "HR12_THRESHOLD"

        return {"ads_id": ADS_ID["Hypertensive Retinopathy Grade 1 or 2"],
                "label": "Hypertensive Retinopathy Grade 1 or 2"}, debug

    # 5️⃣ DRAE strict cutoff (≥ 0.9 = DRAE, otherwise No Suspect)

    if p_drae >= 0.9:

        debug["logic_path"] = "DRAE_FORCE_STRICT_0.9"

        return {"ads_id": ADS_ID["Decreased Retinal Artery Elasticity"],
                "label": "Decreased Retinal Artery Elasticity"}, debug

    else:

        debug["logic_path"] = "DRAE_BELOW_0.9_FORCE_NO"

        return {"ads_id": ADS_ID["No Suspect or Negative"], "label": "No Suspect or Negative"}, debug


# ============================================================
# FASTAPI ENDPOINTS
# ============================================================
app = FastAPI(title="Hypertension Microservice", version=SERVICE_VERSION)
@torch.no_grad()
def run_inference(img):
   x = pre_tf(img).unsqueeze(0).to(DEVICE)
   logits = SEVERITY_MODEL(x)
   probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
   return {SEV_LABELS[i]: float(probs[i]) for i in range(len(SEV_LABELS))}
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
   img = load_image_from_bytes(await file.read())
   prob_map = run_inference(img)
   best, debug = apply_final_override(prob_map)
   return {"file_name": file.filename, "probabilities": prob_map, "best_ads": best, "debug": debug}
@app.post("/predict-hypertension/pair")
async def predict_pair(left_eye_file: UploadFile = File(...), right_eye_file: UploadFile = File(...)):
   left_img = load_image_from_bytes(await left_eye_file.read())
   right_img = load_image_from_bytes(await right_eye_file.read())
   l_probs, r_probs = run_inference(left_img), run_inference(right_img)
   l_best, l_debug = apply_final_override(l_probs)
   r_best, r_debug = apply_final_override(r_probs)
   # Pair override
   p_no_left = l_debug["probabilities"].get("No Suspect or Negative", 0)
   p_no_right = r_debug["probabilities"].get("No Suspect or Negative", 0)
   if p_no_left >= 0.40 or p_no_right >= 0.40:
       pair_label = "No Suspect or Negative"
       ads_id = ADS_ID[pair_label]
       rule = "PAIR:NO_SUSPECT_DOMINANT"
   elif l_best["label"] == "Decreased Retinal Artery Elasticity" or r_best["label"] == "Decreased Retinal Artery Elasticity":
       pair_label = "Decreased Retinal Artery Elasticity"
       ads_id = ADS_ID[pair_label]
       rule = "PAIR:DRAE_DOMINANT"
   else:
       pair_label = "No Suspect or Negative"
       ads_id = ADS_ID[pair_label]
       rule = "PAIR:DEFAULT_NO"
   return {
       "left": {"file_name": left_eye_file.filename, "probabilities": l_probs, "best_ads_overridden": l_best, "debug": l_debug},
       "right": {"file_name": right_eye_file.filename, "probabilities": r_probs, "best_ads_overridden": r_best, "debug": r_debug},
       "pair_debug": {"before_override": [l_best["label"], r_best["label"]], "after_override": pair_label, "ads_id": ads_id, "rule": rule},
       "model": {"version": MODEL_VERSION, "service_version": SERVICE_VERSION, "device": DEVICE},
   }
# ============================================================
# LOCAL DEV RUNNER
# ============================================================
if __name__ == "__main__":
   import uvicorn
   uvicorn.run("hypertension_service1:app", host="0.0.0.0", port=8003, reload=True)