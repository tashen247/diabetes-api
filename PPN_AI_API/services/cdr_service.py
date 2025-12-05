import os, io, cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
app = FastAPI()
# -------------------------------
# Load SegFormer model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
model = SegformerForSemanticSegmentation.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation").to(device)
model.eval()
# -------------------------------
# Segmentation helper
# -------------------------------
def segment_disc_cup(image_rgb: np.ndarray):
   """Returns mask (0=background, 1=disc, 2=cup)."""
   inputs = processor(image_rgb, return_tensors="pt")
   inputs = {k: v.to(device) for k, v in inputs.items()}
   with torch.no_grad():
       outputs = model(**inputs)
       logits = outputs.logits
   up_logits = F.interpolate(logits, size=image_rgb.shape[:2], mode="bilinear", align_corners=False)
   pred = up_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
   return pred
# -------------------------------
# Overlay + optional crop
# -------------------------------
def draw_cup_disc_overlay(image_bgr, disc_mask, cup_mask, cdr_result):
    """
    Draw disc and cup contours on the image.

    Args:
        image_bgr: Original image in BGR format
        disc_mask: Binary mask for optic disc (0/1)
        cup_mask: Binary mask for optic cup (0/1)
        cdr_result: CDR computation result dict (optional, for future use)

    Returns:
        overlay image in BGR format with contours drawn
    """
    overlay = image_bgr.copy()

    # Find contours
    contours_disc, _ = cv2.findContours(
        disc_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours_cup, _ = cv2.findContours(
        cup_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw disc contour in green
    if contours_disc:
        cv2.drawContours(overlay, contours_disc, -1, (0, 255, 0), 2)

    # Draw cup contour in red
    if contours_cup:
        cv2.drawContours(overlay, contours_cup, -1, (0, 0, 255), 2)

    # Optionally add CDR text overlay
    if cdr_result and cdr_result.get("cdr_value") is not None:
        cdr_text = f"CDR: {cdr_result['cdr_value']:.3f}"
        cv2.putText(overlay, cdr_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return overlay
# -------------------------------
# Main API
# -------------------------------
import pyodbc
import os
import logging

# reuse your existing connection string

logger = logging.getLogger("cdr_service")
AI_DB_CONN = "DRIVER={SQL Server};SERVER=Psylocke;DATABASE=Ophthalmology;UID=www.eyepath.co.za;PWD=3y3p@th"
import re


def _make_gai_filename(orig_name: str, referral_id: str) -> str:
    """
    Preserve original eye code (130 or 131) from filename if present.
    """
    m = re.search(r"AI_(13[01])_", orig_name)
    if m:
        eye_code = m.group(1)  # "130" or "131"
    else:
        eye_code = "130"  # default fallback

    return f"GAI_{eye_code}_{referral_id}.JPG"


def _db_get_g_folder(ai_ar_id: int):
    """Return (g_root_folder, db_image_name); create G if needed. Works with/without GFolder column."""

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


import cv2
import numpy as np
import torch.nn.functional as F

def compute_cdr(disc_mask: np.ndarray, cup_mask: np.ndarray):
   """
   Compute vertical cup-to-disc ratio (CDR) and related metrics.
   Args:
       disc_mask: binary mask for optic disc (0/1)
       cup_mask:  binary mask for optic cup (0/1)
   Returns:
       dict with:
         - cdr_value: vertical cup-to-disc ratio
         - disc_area
         - cup_area
         - disc_diameter (vertical)
         - cup_diameter (vertical)
         - bounding_boxes: dict with 'disc' and 'cup' (x1,y1,x2,y2)
   """
   try:
       disc_coords = np.where(disc_mask > 0)
       cup_coords  = np.where(cup_mask > 0)
       if disc_coords[0].size == 0 or cup_coords[0].size == 0:
           logger.warning("[CDR] Empty disc or cup mask.")
           return None
       # --- Heights (vertical diameters)
       disc_top, disc_bottom = disc_coords[0].min(), disc_coords[0].max()
       cup_top, cup_bottom   = cup_coords[0].min(), cup_coords[0].max()
       disc_height = disc_bottom - disc_top
       cup_height  = cup_bottom - cup_top
       if disc_height <= 0:
           return None
       cdr = round(float(cup_height / disc_height), 3)
       # --- Areas
       disc_area = int(np.sum(disc_mask > 0))
       cup_area  = int(np.sum(cup_mask > 0))
       # --- Bounding boxes
       x1_d, x2_d = disc_coords[1].min(), disc_coords[1].max()
       y1_d, y2_d = disc_top, disc_bottom
       x1_c, x2_c = cup_coords[1].min(), cup_coords[1].max()
       y1_c, y2_c = cup_top, cup_bottom
       return {
           "cdr_value": cdr,
           "disc_area": disc_area,
           "cup_area": cup_area,
           "disc_diameter": float(disc_height),
           "cup_diameter": float(cup_height),
           "bounding_boxes": {
               "disc": (int(x1_d), int(y1_d), int(x2_d), int(y2_d)),
               "cup":  (int(x1_c), int(y1_c), int(x2_c), int(y2_c)),
           }
       }
   except Exception as e:
       logger.error(f"[CDR] Failed to compute: {e}")
       return None


import cv2
import numpy as np
def _disc_crop_box(disc_mask: np.ndarray, pad_ratio: float = 0.3):
   """
   Compute a bounding box for the optic disc mask with extra padding.
   Uses heuristics to reject false contours near image border.
   """
   if disc_mask is None or not np.any(disc_mask):
       return None
   H, W = disc_mask.shape[:2]
   # Find contours
   contours, _ = cv2.findContours(
       disc_mask.astype(np.uint8),
       cv2.RETR_EXTERNAL,
       cv2.CHAIN_APPROX_SIMPLE
   )
   if not contours:
       return None
   # Pick candidate discs with heuristics
   candidates = []
   for cnt in contours:
       x, y, w, h = cv2.boundingRect(cnt)
       cx, cy = x + w // 2, y + h // 2
       area = cv2.contourArea(cnt)
       # --- Reject tiny contours ---
       if area < 100:
           continue
       # --- Reject eccentric positions (too close to border) ---
       if cx < 0.15 * W or cx > 0.85 * W or cy < 0.15 * H or cy > 0.85 * H:
           continue
       # --- Aspect ratio check ---
       aspect = w / float(h + 1e-5)
       if aspect < 0.5 or aspect > 2.0:
           continue
       candidates.append((area, (x, y, w, h)))
   if not candidates:
       # fallback: pick largest contour
       largest = max(contours, key=cv2.contourArea)
       x, y, w, h = cv2.boundingRect(largest)
   else:
       # pick largest valid candidate
       _, (x, y, w, h) = max(candidates, key=lambda t: t[0])
   # Add padding
   pad_x = int(w * pad_ratio)
   pad_y = int(h * pad_ratio)
   x1 = max(0, x - pad_x)
   y1 = max(0, y - pad_y)
   x2 = min(W, x + w + pad_x)
   y2 = min(H, y + h + pad_y)
   return x1, y1, x2, y2

def validate_fundus_image(img: np.ndarray, debug: bool = False) -> bool:
    """
    Return True iff the image looks like a fundus photograph.
    Uses: circular border (with fallback), vessel edges in circle ROI,
    bright disc percentile, and optional straight-line rejection.
    """
    if img is None or img.size == 0:
        return False

    # --- grayscale (handle OpenCV BGR) ---
    if img.ndim == 3 and img.shape[2] == 3:
        # Try BGR first (cv2.imdecode gives BGR); fall back to RGB if needed
        gray_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # pick the one with more edge energy to be robust
        e_bgr = cv2.Canny(gray_bgr, 30, 100).sum()
        e_rgb = cv2.Canny(gray_rgb, 30, 100).sum()
        gray = gray_bgr if e_bgr >= e_rgb else gray_rgb
    else:
        gray = img.copy()

    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # --- circular border: primary pass ---
    mn, mx = int(min(h, w) * 0.25), int(min(h, w) * 0.75)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) // 2,
        param1=60, param2=25, minRadius=mn, maxRadius=mx
    )
    # fallback, more permissive
    if circles is None:
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=min(h, w) // 2,
            param1=40, param2=18, minRadius=int(min(h, w) * 0.20),
            maxRadius=int(min(h, w) * 0.80)
        )
    if circles is None:
        if debug: print("❌ No circular fundus border detected (both passes)")
        return False

    c = np.uint16(np.around(circles))[0][0]
    cx, cy, r = int(c[0]), int(c[1]), int(c[2])

    # reject if circle occupies too little area (likely not a fundus)
    circle_area = np.pi * r * r
    if circle_area < 0.30 * (h * w):
        if debug: print("❌ Fundus circle too small")
        return False

    # --- ROI mask (inside circle only) ---
    mask = np.zeros_like(gray, np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    n_roi = max(1, int(mask.sum() // 255))

    # --- vessel presence (edges in ROI) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi)
    edges = cv2.Canny(enhanced, 30, 100)
    vessel_fraction = float((edges > 0).sum()) / float(n_roi)
    if vessel_fraction < 0.002:
        if debug: print("❌ Not enough vessel-like structures")
        return False

    # --- bright disc presence (use percentile in ROI) ---
    roi_vals = roi[mask > 0]
    if roi_vals.size == 0:
        return False
    p95 = np.percentile(roi_vals, 95)
    if p95 < 130:  # adaptive-ish versus absolute 100
        if debug: print("❌ No sufficiently bright disc region")
        return False

    # --- reject report pages: many long straight lines in ROI ---
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=60,
                            minLineLength=int(0.20 * min(h, w)),
                            maxLineGap=10)
    if lines is not None and len(lines) > 40:
        if debug: print("❌ Too many straight lines (likely report/OCT sheet)")
        return False

    if debug:
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(overlay, (cx, cy), r, (0, 255, 0), 2)
        cv2.imwrite("fundus_debug.jpg", overlay)
        print("✔ Fundus validated, debug saved as fundus_debug.jpg")

    return True


@app.post("/cdr/analyze")
async def analyze_cdr(
        file: UploadFile = File(...),
        referral_id: str = Query(...),
        bias: str = Form("auto"),
        db_image_name_q: str = Query(...),
        g_folder_q: str = Query(...)
):
    """
    Analyze CDR using SegFormer optic disc/cup segmentation.
    Returns overlay and crop paths with metadata.
    """
    try:
        # --- read image ---
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image upload")

        # if not validate_fundus_image(image):
        #     return {"error": "Rejected", "detail": "Invalid fundus image"}

        # --- segment disc + cup ---
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = segment_disc_cup(img_rgb)  # segformer model call
        disc_mask = (mask == 1).astype(np.uint8)
        cup_mask = (mask == 2).astype(np.uint8)

        # --- compute CDR (returns dict) ---
        cdr_result = compute_cdr(disc_mask, cup_mask)

        # --- construct filename with G prefix ---
        stem, ext = os.path.splitext(db_image_name_q or file.filename)
        gai_filename = f"G{stem}{ext}"
        os.makedirs(g_folder_q, exist_ok=True)

        # --- draw overlay ---
        overlay_img = draw_cup_disc_overlay(
            image,
            disc_mask,
            cup_mask,
            cdr_result  # Pass the full result dict
        )

        overlay_path = os.path.join(g_folder_q, db_image_name_q)
        logger.info(f"Saving overlay to: {overlay_path}")
       # cv2.imwrite(overlay_path, overlay_img)

        # --- crop disc region WITH overlay ---
        crop_path = None
        box = _disc_crop_box(disc_mask, pad_ratio=0.45)
        if box:
            x1, y1, x2, y2 = box
            H, W = overlay_img.shape[:2]
            x1, x2 = max(0, min(W, x1)), max(0, min(W, x2))
            y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                crop_overlay = overlay_img[y1:y2, x1:x2]
                crop_path = os.path.join(g_folder_q, f"{gai_filename}")
                cv2.imwrite(crop_path, crop_overlay)

        # --- response ---
        response_data = {
            "filename": gai_filename,
            "overlay_path": overlay_path,
            "crop_path": crop_path,
            "referral_id": referral_id,
            "bias": bias,
            "db_image_name": db_image_name_q,
            "g_folder": g_folder_q,
        }

        # Add CDR metrics if computation succeeded
        if cdr_result:
            response_data.update({
                "cdr_value": cdr_result["cdr_value"],
                "disc_area": cdr_result["disc_area"],
                "cup_area": cdr_result["cup_area"],
                "disc_diameter": cdr_result["disc_diameter"],
                "cup_diameter": cdr_result["cup_diameter"],
                "bounding_boxes": cdr_result["bounding_boxes"]
            })
        else:
            response_data["cdr_value"] = None

        return response_data

    except Exception as e:
        logger.error(f"[SegFormer] Failed on {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"CDR analyze failed: {e}")