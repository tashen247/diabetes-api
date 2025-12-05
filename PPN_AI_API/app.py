from io import BytesIO
#from msilib.schema import Condition
import os
#from symbol import comparison
import os
import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import jwt, JWTError
import tensorflow as tf
import pyodbc
import json
#import tensorflow_addons as tfa
import logging
from typing import Literal
# --- add these imports at top of your router file ---
from typing import Optional, Literal, Dict, Any
import os, re, httpx
from fastapi import UploadFile, File, HTTPException
#from skimage.measure import regionprops, label
#import pandas as pd
#from keras import backend as K
import tempfile
import gc

from starlette.responses import JSONResponse

#from dotenv import load_dotenv
from config import MODEL_PATHS, CLASS_NAMES, SECRET_KEY, condition_to_ad_id, connection_string
from db_functions.db_functions import DbFunctions
from services.model_service import (focal_loss, calculate_cdr, preprocess_image,
                                    extract_features_from_image, extract_cdr_and_ecc,
                                    format_prediction_response, models)

# create an instance of the class
dbf = DbFunctions(use_live=True)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# load_dotenv('components/utils/.env')
# #  Authentication Setup
oauth_scheme = OAuth2PasswordBearer(tokenUrl="/token")
#SECRET_KEY = "46e2011cb14891bb9c57ed2b4c688250c57ad3ad354359e87ee0d715d581112e"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 2
username = os.getenv('AI_USERNAME')  # 'PPN_@I_Admin'
password = os.getenv('AI_PASSWORD')  # 'BGbiT#(#8_$)'
base_url = "http://ppnai-prod:8000"
#base_url = "http://triton:8000"
# logger = dbf.setup_logging()


# Bilateral promotion master switch: off | shadow | on
BILAT_MODE = os.getenv("MYOPIA_BILATERAL_MODE", "off").lower()

# Strict thresholds (conservative defaults)
BILAT_EITHER_MIN_PROB = float(os.getenv("MYOPIA_BILATERAL_EITHER_MIN_PROB", "0.95"))  # single eye very high
BILAT_EITHER_MIN_COMP = float(os.getenv("MYOPIA_BILATERAL_EITHER_MIN_COMP", "0.25"))
BILAT_BOTH_MIN_PROB = float(os.getenv("MYOPIA_BILATERAL_BOTH_MIN_PROB", "0.80"))  # both eyes highish
BILAT_BOTH_MIN_COMP = float(os.getenv("MYOPIA_BILATERAL_BOTH_MIN_COMP", "0.35"))
BILAT_EXTREME_WEAK_COMP = float(os.getenv("MYOPIA_BILATERAL_EXTREME_WEAK_COMP", "0.10"))  # hard veto if both below
BILAT_FORBID_IF_BOTH_BLURRY = os.getenv("MYOPIA_BILATERAL_FORBID_IF_BOTH_BLURRY", "1") == "1"


def _extract_eye_block(result: dict) -> dict:
    blk = result.get("myopia", {})
    conf = blk.get("confidence_scores", {}) or {}
    feats = blk.get("metrics", {}) or {}
    qc = blk.get("quality", {}) or {}
    pred = str(blk.get("prediction", "") or "")
    is_blurry = bool(qc.get("is_blurry", False))
    return {
        "prediction": pred,  # e.g., "Myopia", "Normal (Low Morphology Confidence)"
        "ads_id": int(blk.get("ads_id", 26)),
        "p_myopia": float(conf.get("Myopia", 0.0)),
        "comp": float(feats.get("composite_myopia_score", 0.0)),
        "quality_ok": bool(qc.get("quality_ok", True)),
        "is_blurry": is_blurry,
    }

def _apply_promotion_to_eye(eye_result: dict, final_label: str, final_ads_id: int, reason: str, mode: str):
    if "myopia" not in eye_result:
        return
    # Only mutate output when mode == "on"
    if mode == "on":
        eye_result["myopia"]["prediction"] = final_label
        eye_result["myopia"]["ads_id"] = final_ads_id
        eye_result["myopia"]["decision"] = f"final (bilateral_promotion: {reason})"
    # Always attach audit
    eye_result["myopia"].setdefault("bilateral_rule", {})
    eye_result["myopia"]["bilateral_rule"].update({"mode": mode, "reason": reason})

def bilateral_promote(left_result: dict, right_result: dict) -> dict:
    """
    Conservative patient-level promotion:
      - Mode: off (default), shadow (log only), on (apply)
      - Requires at least one eye already 'Myopia' per-eye.
      - Forbids if both morphology extremely weak or both blurry (configurable).
      - Promote if:
          A) Either eye p>=BILAT_EITHER_MIN_PROB AND comp>=BILAT_EITHER_MIN_COMP AND quality_ok
         OR
          B) Both eyes p>=BILAT_BOTH_MIN_PROB AND comp>=BILAT_BOTH_MIN_COMP AND both quality_ok
    Returns: {"should_promote": bool, "final_label": "Myopia", "final_ads_id": 42, "reason": str}
    """
    mode = BILAT_MODE  # "off" | "shadow" | "on"
    L = _extract_eye_block(left_result)
    R = _extract_eye_block(right_result)

    # Always prepare an audit baseline
    audit = {
        "mode": mode,
        "L": L, "R": R,
        "rules": {
            "either_min_prob": BILAT_EITHER_MIN_PROB,
            "either_min_comp": BILAT_EITHER_MIN_COMP,
            "both_min_prob": BILAT_BOTH_MIN_PROB,
            "both_min_comp": BILAT_BOTH_MIN_COMP,
            "extreme_weak_comp": BILAT_EXTREME_WEAK_COMP,
            "forbid_if_both_blurry": BILAT_FORBID_IF_BOTH_BLURRY,
        }
    }

    # MASTER switch
    if mode not in ("shadow", "on"):
        return {"should_promote": False, "reason": "bilateral_off", "audit": audit}

    # 0) Require at least one eye already called Myopia per-eye
    if not (L["prediction"].startswith("Myopia") or R["prediction"].startswith("Myopia")):
        return {"should_promote": False, "reason": "no_eye_called_myopia", "audit": audit}

    # 1) Hard veto: both morphology extremely weak
    if (L["comp"] < BILAT_EXTREME_WEAK_COMP) and (R["comp"] < BILAT_EXTREME_WEAK_COMP):
        return {"should_promote": False, "reason": "both_extremely_weak_morphology", "audit": audit}

    # 2) Optional veto: both blurry
    if BILAT_FORBID_IF_BOTH_BLURRY and (L["is_blurry"] and R["is_blurry"]):
        return {"should_promote": False, "reason": "both_blurry", "audit": audit}

    # Rule A: a single very-strong eye with some morphology + QC
    ruleA = (
            (max(L["p_myopia"], R["p_myopia"]) >= BILAT_EITHER_MIN_PROB) and
            ((L["p_myopia"] >= BILAT_EITHER_MIN_PROB and L["comp"] >= BILAT_EITHER_MIN_COMP and L["quality_ok"]) or
             (R["p_myopia"] >= BILAT_EITHER_MIN_PROB and R["comp"] >= BILAT_EITHER_MIN_COMP and R["quality_ok"]))
    )

    # Rule B: both eyes high-ish prob + morphology + QC
    ruleB = (
            (L["p_myopia"] >= BILAT_BOTH_MIN_PROB and R["p_myopia"] >= BILAT_BOTH_MIN_PROB) and
            (L["comp"] >= BILAT_BOTH_MIN_COMP and R["comp"] >= BILAT_BOTH_MIN_COMP) and
            (L["quality_ok"] and R["quality_ok"])
    )

    if ruleA:
        return {"should_promote": True, "final_label": "Myopia", "final_ads_id": 42,
                "reason": "either_very_high_prob_with_morph_qc", "audit": audit}
    if ruleB:
        return {"should_promote": True, "final_label": "Myopia", "final_ads_id": 42,
                "reason": "both_high_prob_with_morph_qc", "audit": audit}

    return {"should_promote": False, "reason": "no_rule_met", "audit": audit}

def get_ad_id(condition):
    condition = condition.strip().lower()
    return condition_to_ad_id.get(condition, None)


def save_to_tempfile(image_bytes):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(image_bytes)
    temp.flush()
    return temp.name


# Initialize FastAPI
app = FastAPI(
    title="Fundus Image Diagnosis API",
    description="API for diagnosing various retinal and eye conditions.",
    version="5.0.0"
)

LABEL_FILE_PATH = r"C:\AI_Images\Glaucoma1\labels.csv"

print(MODEL_PATHS)


# Compare Results
def compare_results(api_results, existing_results):
    if not existing_results:
        return {
            "status": "No existing results found for comparison",
            "api_result": api_results
        }

    comparison = {}
    for key, value in api_results.items():
        # Fetch the corresponding value for the key from existing_results
        existing_value = existing_results.get(key)
        logging.info(f"Key: {key}, API Value: {value}, Existing Value: {existing_value}")
        # Construct the comparison entry

        comparison[key] = {
            "api_result": api_results,
            "existing_result": existing_results,  # Use the specific value, not the whole dictionary
            "match": value == existing_value if existing_value is not None else None
        }

    return comparison


def write_eye_result_to_db(eye_type, eye_id, result, condition, referral_id, model_id):
    """
    Write eye result to database.

    Args:
        eye_type (str): 'left' or 'right'
        eye_id (int): Eye ID (130 for right, 131 for left)
        result (dict): The result dictionary
        condition (str): The condition being checked (e.g., 'myopia')
        referral_id (str): The referral ID
        model_id (str): The model ID
    """
    try:
        logging.info(f"Writing {eye_type} eye result to DB for condition={condition}")

        # --- Defensive extraction ---
        condition_result = result.get(condition)
        if not condition_result:
            raise ValueError(f"Missing '{condition}' key in result: {result}")

        ads_id = condition_result.get("ads_id", None)
        if ads_id is None:
            raise ValueError(f"Missing ads_id for condition {condition}")

        conf = condition_result.get("confidence_scores")

        # --- Handle both float and dict formats ---
        if isinstance(conf, (int, float)):
            logging.warning(f"{condition.upper()} returned single confidence value ({conf}); converting to dict")
            conf_dict = {condition_result.get("prediction", "confidence"): float(conf)}
        elif isinstance(conf, dict):
            conf_dict = conf
        else:
            logging.warning(f"{condition.upper()} returned invalid confidence_scores: {conf}; defaulting to empty dict")
            conf_dict = {}

        confidence_scores_json = json.dumps(conf_dict)

        # --- Write to DB ---
        dbf.write_results_to_db(
            referral_id=referral_id,
            ai_id=eye_id,
            ads_ad_id=get_ad_id(condition),
            ads_id=ads_id,
            model_no=model_id,
            confidence_scores=confidence_scores_json
        )

        logging.info(
            f"✅ {eye_type.capitalize()} eye result written successfully "
            f"(condition={condition}, ads_id={ads_id}, confidence_keys={list(conf_dict.keys())})"
        )

    except Exception as e:
        logging.error(f"❌ Failed to write {eye_type} eye result for {condition}: {e}", exc_info=True)


# Prediction Logic
def predict_condition(file: UploadFile, condition: str):
    """
    Predicts the diagnosis using the appropriate model.
    Handles binary and multi-class classification for multiple conditions.
    Args:
        file: Uploaded image file.
        condition: Selected condition for prediction.
    Returns:
        predicted_class: The predicted class.
        confidence_scores: Confidence scores for the prediction.
        keras_file_name: The Keras model file name used for the prediction.
    """
    try:
        # Step 1: Read and preprocess the uploaded file
        print(file.filename)
        file_bytes = file.file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or unsupported format.")

        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(file.file.read())

        image_input = preprocess_image(temp_path)
        input_size = (256, 256) if condition == "the_model_that_wants_256" else (224, 224)
        image_input = preprocess_image(temp_path, target_size=input_size)

        # tabular_input = np.array([extract_features_from_image(temp_path)])
        # os.remove(temp_path)  # Clean up

        if image_input is None:
            raise HTTPException(status_code=500, detail="Image preprocessing failed.")

        # Step 2: Handle glaucoma-specific inputs
        if condition == "glaucoma":
            # Extract numeric features (CDR, Ecc-Cup, Ecc-Disc)
            cdr, ecc_cup, ecc_disc = extract_cdr_and_ecc(image)

            # Combine numeric features into a single tensor
            numeric_input = tf.convert_to_tensor([[cdr, ecc_cup, ecc_disc]], dtype=tf.float32)  # Shape: (1, 3)
            cdr_input = np.array([[0.5]])  # Replace with actual CDR value
            ecc_cup_input = np.array([[0.7]])  # Replace with actual Ecc-Cup value
            ecc_disc_input = np.array([[0.8]])  # Replace with actual Ecc-Disc value
            # Debug: Log input shapes
            # print(f"Image Input Shape: {image_input.shape}")
            print(f"Numeric Input Shape: {numeric_input.shape}")

            # **New logic to include tabular inputs**
            # Mock tabular inputs (replace with real logic to extract features)
            tabular_input = np.array([[1000, 1050, 2000, 2048]])  # Example tabular data
            # Extract real features from the image
            #tabular_input = extract_features_from_image(file.filename)
            if tabular_input is None:
                raise ValueError("Failed to extract features")
            tabular_input_tensor = tf.convert_to_tensor(tabular_input, dtype=tf.float32)

            # if len(image_input) != len(tabular_input_tensor):
            #     raise ValueError("Mismatch between number of images and tabular features")

            # Predict using the glaucoma models
            predictions = models["glaucoma"].predict([image_input, tabular_input_tensor])
            #predictions = models["glaucoma"].predict([image_input])
            keras_file_name = os.path.basename(MODEL_PATHS["glaucoma"])  # Extract the file name

        else:
            # For other conditions, use the respective model
            predictions = models[condition].predict(image_input)
            keras_file_name = os.path.basename(MODEL_PATHS[condition])  # Extract the file name

        # Step 3: Interpret predictions
        if condition in ["glaucoma", "cnv", "rvo"]:
            threshold = 0.5
            # Binary classification
            confidence_score = float(predictions.flatten()[0])
            predicted_index = 1 if confidence_score > threshold else 0
            confidence_scores = {
                CLASS_NAMES[condition][0]["label"]: round(1 - confidence_score, 6),
                CLASS_NAMES[condition][1]["label"]: round(confidence_score, 6)
            }
            predicted_class_data = CLASS_NAMES[condition][predicted_index]
            predicted_class = predicted_class_data["label"]
            ads_id = predicted_class_data["id"]

        else:
            # Multi-class classification
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence_scores = {
                CLASS_NAMES[condition][i]["label"]: round(float(predictions[0][i]), 6)
                for i in range(len(CLASS_NAMES[condition]))
            }
            predicted_class_data = CLASS_NAMES[condition][predicted_class_idx]
            predicted_class = predicted_class_data["label"]
            ads_id = predicted_class_data["id"]

        # Return the prediction, confidence scores, and model file name
        return predicted_class, confidence_scores, keras_file_name

    except Exception as e:
        logging.error(f"Error during {condition} diagnosis: {e}")
        raise HTTPException(status_code=500, detail=f"Error during {condition} diagnosis: {str(e)}")


def generate_token(user_id: str):
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


@app.post("/token", response_model=dict)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # if form_data.username == stored_creds["client_id"] and form_data.password == stored_creds["client_secret"]:
    if form_data.username == username and form_data.password == password:
        token = generate_token(form_data.username)
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")


# @app.post("/diagnose")
# async def diagnose_condition(
#         referral_id: str,
#         left_eye_file: UploadFile = File(...),
#         right_eye_file: UploadFile = File(...),
#         condition: Literal[
#             "diabetes", "hypertension", "macular_degeneration", "cnv", "rvo", "glaucoma", "pathological", "myopia"
#         ] = Query(..., description="Select the condition to diagnose")
# ):
#     try:
#         # Process and prepare input data
#         temp_path = "temp_image.jpg"
#         with open(temp_path, "wb") as temp_file:
#             temp_file.write(left_eye_file.file.read())
#
#         image_input = preprocess_image(temp_path)
#         tabular_input = np.array([extract_features_from_image(temp_path)])
#         os.remove(temp_path)
#
#         if image_input is None or tabular_input is None:
#             raise ValueError("Preprocessing failed for the uploaded image.")
#
#         keras_file_name = os.path.basename(MODEL_PATHS[condition]["path"])
#
#         # Get model predictions
#         if condition == "glaucoma":
#             glaucoma_classes = [
#                 {"label": "No Glaucoma", "ads_id": 16},
#                 {"label": "Glaucoma", "ads_id": 17}
#             ]
#
#             cdr_value = np.array([[calculate_cdr(image_input)]], dtype=np.float32)
#             predictions = models["glaucoma"].predict([image_input])
#             print(f"Glaucoma predictions: {predictions}")
#
#             if np.isnan(predictions).any():
#                 predictions = np.nan_to_num(predictions)
#
#             confidence_score = float(predictions.flatten()[0])
#             threshold = 0.5
#             predicted_index = 1 if confidence_score > threshold else 0
#
#             confidence_scores = {
#                 glaucoma_classes[0]["label"]: round(1 - confidence_score, 6),
#                 glaucoma_classes[1]["label"]: round(confidence_score, 6)
#             }
#
#             predicted_class_data = glaucoma_classes[predicted_index]
#             predicted_class = predicted_class_data["label"]
#             ads_id = predicted_class_data["ads_id"]
#             model_no = MODEL_PATHS[condition]["number"]
#
#             left_id, right_id = [130, 131]  #dbf.get_left_and_right_eye_image_ids(referral_id)
#             logging.info(f"Left Eye Image ID: {left_id}, Right Eye Image ID: {right_id}")
#
#             for eye_id in [left_id, right_id]:
#                 dbf.write_results_to_db(
#                     referral_id=referral_id,
#                     ai_id=eye_id,
#                     ads_ad_id=get_ad_id(condition),
#                     ads_id=ads_id,
#                     model_no=model_no,
#                     confidence_scores=confidence_scores
#                 )
#
#             # ✅ Return here to avoid running generic processing below
#             return format_prediction_response(condition, predicted_class, confidence_scores, ads_id=ads_id) | {
#                 "keras_file_name": keras_file_name}
#
#         if condition == "myopia":
#             myopia_classes = [
#                 {"label": "No Myopia", "ads_id": 20},
#                 {"label": "Myopia", "ads_id": 21}
#             ]
#
#             predictions = models["myopia"].predict(image_input)
#             print(f"Myopia predictions: {predictions}")
#
#             if np.isnan(predictions).any():
#                 predictions = np.nan_to_num(predictions)
#
#             confidence_score = float(predictions.flatten()[0])
#             threshold = 0.5
#             predicted_index = 1 if confidence_score > threshold else 0
#
#             confidence_scores = {
#                 myopia_classes[0]["label"]: round(1 - confidence_score, 6),
#                 myopia_classes[1]["label"]: round(confidence_score, 6)
#             }
#
#             predicted_class_data = myopia_classes[predicted_index]
#             predicted_class = predicted_class_data["label"]
#             ads_id = predicted_class_data["ads_id"]
#             model_no = MODEL_PATHS[condition]["number"]
#
#             left_id, right_id = [130, 131]  # dbf.get_left_and_right_eye_image_ids(referral_id)
#             logging.info(f"Left Eye Image ID: {left_id}, Right Eye Image ID: {right_id}")
#
#             for eye_id in [left_id, right_id]:
#                 dbf.write_results_to_db(
#                     referral_id=referral_id,
#                     ai_id=eye_id,
#                     ads_ad_id=get_ad_id(condition),
#                     ads_id=ads_id,
#                     model_no=model_no,
#                     confidence_scores=confidence_scores
#                 )
#
#             return format_prediction_response(condition, predicted_class, confidence_scores, ads_id=ads_id) | {
#                 "keras_file_name": keras_file_name}
#
#         elif condition == "hypertension":
#             # Define classes for hypertension
#             hypertension_classes = [
#                 {"label": "No Suspect or Negative", "ads_id": 7},
#                 {"label": "Decreased Retinal Artery Elasticity", "ads_id": 8},
#                 {"label": "Hypertensive Retinopathy Grade 1 or 2", "ads_id": 9},
#                 {"label": "Hypertensive Retinopathy Grade 3 or 4", "ads_id": 10}
#             ]
#
#             # Get predictions
#             predictions = np.array(models[condition].predict(image_input))
#
#             logging.info(
#                 f"Hypertension predictions type: {type(predictions)}, shape: {predictions.shape if hasattr(predictions, 'shape') else 'no shape'}")
#
#             # Validate predictions
#             if isinstance(predictions, list) and len(predictions) == 0:
#                 raise ValueError("Model returned empty predictions")
#
#             if not isinstance(predictions, np.ndarray):
#                 predictions = np.array(predictions)
#
#             if predictions.size == 0:
#                 raise ValueError("Model returned empty predictions")
#
#             # Extract prediction values
#             if len(predictions.shape) == 1:
#                 raw_predictions = predictions
#             else:
#                 try:
#                     raw_predictions = predictions[0]
#                 except IndexError:
#                     raise ValueError("Predictions array has unexpected structure")
#
#         else:
#             # Standard prediction for other conditions
#             predictions = models[condition].predict(image_input)
#
#         # Defensive check for all conditions
#         if predictions is None or len(predictions) == 0 or (hasattr(predictions, 'size') and predictions.size == 0):
#             raise ValueError(f"No predictions returned from {condition} model.")
#
#         logging.info(f"{condition} prediction shape: {predictions.shape}, values: {predictions}")
#
#         # Process predictions based on condition
#         if condition == "hypertension":
#             raw_predictions = predictions[0]
#             predicted_class_idx = int(np.argmax(raw_predictions))
#
#             # Safety check on class index
#             if predicted_class_idx >= len(hypertension_classes):
#                 logging.warning(f"Predicted index {predicted_class_idx} out of bounds for hypertension classes")
#                 predicted_class_idx = len(hypertension_classes) - 1
#
#             # Calculate confidence scores
#             confidence_scores = {}
#             for i in range(len(raw_predictions)):
#                 if i < len(hypertension_classes):
#                     class_name = hypertension_classes[i]["label"]
#                     confidence_scores[class_name] = round(float(raw_predictions[i]), 6)
#
#             # Get predicted class data
#             predicted_class_data = hypertension_classes[predicted_class_idx]
#
#         else:
#             # Handle standard multi-class predictions
#             expected_classes = len(CLASS_NAMES[condition])
#             actual_outputs = len(predictions[0])
#
#             # Validate class count
#             if expected_classes != actual_outputs:
#                 logging.error(f"[{condition}] model-class mismatch: expected {expected_classes}, got {actual_outputs}")
#                 raise ValueError(
#                     f"Class count mismatch for {condition}: expected {expected_classes}, got {actual_outputs}")
#
#             # Get prediction index and confidence scores
#             predicted_class_idx = np.argmax(predictions, axis=1)[0]
#             confidence_scores = {
#                 CLASS_NAMES[condition][i]["label"]: round(float(predictions[0][i]), 6)
#                 for i in range(expected_classes)
#             }
#             print(confidence_scores)
#             # Get predicted class data
#             predicted_class_data = CLASS_NAMES[condition][predicted_class_idx]
#
#         # Extract prediction details
#         predicted_class = predicted_class_data["label"]
#         ads_id = predicted_class_data["ads_id"]
#         model_no = MODEL_PATHS[condition]["number"]
#
#         # Get image IDs and write results to database
#         left_id, right_id = [130, 131]  # dbf.get_left_and_right_eye_image_ids(referral_id)
#         logging.info(f"Left Eye Image ID: {left_id}, Right Eye Image ID: {right_id}")
#
#         for eye_id in [left_id, right_id]:
#             dbf.write_results_to_db(
#                 referral_id=referral_id,
#                 ai_id=eye_id,
#                 ads_ad_id=get_ad_id(condition),
#                 ads_id=ads_id,
#                 model_no=model_no,
#                 confidence_scores=confidence_scores
#             )
#
#         # Return formatted response
#         return format_prediction_response(condition, predicted_class, confidence_scores, ads_id=ads_id) | {
#             "keras_file_name": keras_file_name}
#
#     except Exception as e:
#         logging.error(f"Error during {condition} diagnosis: {e}")
#         raise HTTPException(status_code=500, detail=f"Error during {condition} diagnosis: {str(e)}")


# @app.post("/diagnoseAll")
# async def diagnose_all_conditions(
#         referral_id: str,
#         left_eye_file: UploadFile = File(...),
#         right_eye_file: UploadFile = File(...)
# ):
#     global condition, eye_label
#     try:
#         responses = {}
#
#         # Read both files into memory
#         left_image_bytes = await left_eye_file.read()
#         right_image_bytes = await right_eye_file.read()
#
#         # Save to temp files
#         left_temp_path = save_to_tempfile(left_image_bytes)
#         right_temp_path = save_to_tempfile(right_image_bytes)
#
#         # Preprocess both
#         left_image_input = preprocess_image(left_temp_path)
#         right_image_input = preprocess_image(right_temp_path)
#
#         #features
#         left_features = extract_features_from_image(left_temp_path)
#         right_features = extract_features_from_image(right_temp_path)
#
#         # Clean up temp files
#         os.remove(left_temp_path)
#         os.remove(right_temp_path)
#
#         left_tabular = np.array(left_features).reshape(1, -1)
#         right_tabular = np.array(right_features).reshape(1, -1)
#
#         # ================================
#         # Load all models once
#         # ================================
#         # inside /diagnoseAll where you load models
#         models = {}
#         for condition, meta in MODEL_PATHS.items():
#             if condition.startswith('#'):
#                 continue
#             # no Addons>F1Score mapping needed
#             with tf.keras.utils.custom_object_scope({'loss': focal_loss()}):
#                 models[condition] = tf.keras.models.load_model(meta["path"], compile=False)
#
#         if any(x is None for x in [left_image_input, right_image_input, left_features, right_features]):
#             raise ValueError("Preprocessing failed for one or both eye images.")
#
#         left_id, right_id = dbf.get_left_and_right_eye_image_ids(referral_id)
#
#         # Build a list of dictionaries for both eyes
#         eyes = [
#             {"label": "left", "image": left_image_input, "file": left_eye_file, "id": left_id, "tabular": left_tabular},
#             {"label": "right", "image": right_image_input, "file": right_eye_file, "id": right_id,
#              "tabular": right_tabular}
#         ]
#
#         # Iterate through each eye
#         for eye_data in eyes:
#             eye_label = eye_data["label"]
#             image_input = eye_data["image"]
#             eye_file = eye_data["file"]
#             ai_id = eye_data["id"]
#             tabular_input = eye_data["tabular"]
#
#             # existing_results = get_existing_results(eye_file.filename)
#
#             for condition, meta in MODEL_PATHS.items():
#                 keras_file_name = os.path.basename(MODEL_PATHS[condition]["path"])
#                 if condition.startswith('#'):
#                     continue  # Skip disabled/commented-out models
#
#                 model = models[condition]
#
#                 if condition == "glaucoma":
#                     print(f"Running prediction for {condition} on {eye_label} eye")
#                     predictions = model.predict([image_input])
#                 else:
#                     print(f"Running prediction for {condition} on {eye_label} eye")
#                     predictions = model.predict(image_input)
#
#                 #=== Inference & confidence ===
#                 if condition in ["glaucoma", "cnv", "rvo", "myopia"]:
#                     threshold = 0.5
#                     confidence_score = float(predictions.flatten()[0])
#                     predicted_index = 1 if confidence_score > threshold else 0
#                     confidence_scores = {
#                         CLASS_NAMES[condition][0]["label"]: round(1 - confidence_score, 6),
#                         CLASS_NAMES[condition][1]["label"]: round(confidence_score, 6)
#                     }
#                     predicted_class_data = CLASS_NAMES[condition][predicted_index]
#                 else:
#                     predicted_class_idx = np.argmax(predictions, axis=1)[0]
#                     predicted_class_data = CLASS_NAMES[condition][predicted_class_idx]
#                     expected_classes = len(CLASS_NAMES[condition])
#                     actual_outputs = len(predictions[0])
#                     confidence_scores = {
#                         CLASS_NAMES[condition][i]["label"]: round(float(predictions[0][i]), 6)
#                         for i in range(expected_classes)
#                     }
#                     if expected_classes != actual_outputs:
#                         dbf.log_error_to_db(
#                             error_message=f"[{condition}] Model-Class Mismatch",
#                             stack_trace=f"Expected {expected_classes}, got {actual_outputs}",
#                             endpoint="/diagnoseAll"
#                         )
#                         continue
#
#                 predicted_class = predicted_class_data["label"]
#                 ads_id = predicted_class_data["id"]
#                 model_no = meta["number"]
#                 print(f"{eye_label} eye -{ai_id} -  {condition} - model_no: {model_no}")
#                 #cleanup # Cleanup
#                 # del model
#                 # gc.collect()
#
#                 # comparison = compare_results(
#                 #     api_results={"prediction": predicted_class, "confidence_scores": confidence_scores},
#                 #     # existing_results=existing_results
#                 # )
#
#                 left_id, right_id = [130, 131]  # dbf.get_left_and_right_eye_image_ids(referral_id)
#                 logging.info(f"Left Eye Image ID: {left_id}, Right Eye Image ID: {right_id}")
#
#                 for eye_id in [left_id, right_id]:
#                     dbf.write_results_to_db(
#                         referral_id=referral_id,
#                         ai_id=eye_id,
#                         ads_ad_id=get_ad_id(condition),
#                         ads_id=ads_id,
#                         model_no=model_no,
#                         confidence_scores=confidence_scores
#                     )
#
#                 responses_key = f"{condition}_{eye_label}"
#                 responses[responses_key] = format_prediction_response(
#                     condition, predicted_class, confidence_scores,
#                     # comparison=comparison,
#                     # existing_results=existing_results,
#                     ads_id=ads_id
#                 )
#
#                 #responses[responses_key]["keras_file_name"] = os.path.basename(meta["path"])
#
#         return responses
#
#     except Exception as e:
#         # logging.error(f"[{condition}] failed on {e}")#: {e}")
#         dbf.log_error_to_db(
#             error_message="Error during full diagnosis",
#             stack_trace=str(e),
#             endpoint="/diagnoseAll"
#         )
#     raise HTTPException(status_code=500, detail=f"Error during full diagnosis: ")  #{e}")
#

# Create router for individual condition endpoints
condition_router = APIRouter()


@app.post("/predict-diabetes")
async def predict_diabetes(
        referral_id: str,
        left_eye_file: UploadFile = File(...),
        right_eye_file: UploadFile = File(...)
):
    condition = "diabetes"
    try:

        model_id = 2
        left_id, right_id = [130, 131]
        #start
        dbf.insert_endpoint_logs(referral_id, f"'{base_url}/predict-diabetes?referral_id={referral_id}", condition,
                                 left_id, right_id, 2)

        #left_result = await predict_generic_condition(left_eye_file, condition, "left", referral_id)
        #left_result = await model_service.diagnose(referral_id, left_eye_file, condition, 131)
        left_result = await handle_diabetes(referral_id, left_eye_file)

        #write_eye_result_to_db('left', left_id, left_result["confidence"], condition, referral_id, 9)
        dbf.write_results_to_db(
            referral_id=referral_id, ai_id=left_id, ads_ad_id=get_ad_id(condition), ads_id=left_result["ads_id"],
            model_no=9, confidence_scores=left_result["confidence"]
        )
        #right_result = await predict_generic_condition(right_eye_file, condition, "right", referral_id)
        # right_result = await model_service.diagnose(referral_id, right_eye_file, condition,130)
        # write_eye_result_to_db('right', right_id, right_result, condition, referral_id, 9)
        right_result = await handle_diabetes(referral_id, right_eye_file)
        dbf.write_results_to_db(
            referral_id, ai_id=right_id, ads_ad_id=get_ad_id(condition), ads_id=right_result["ads_id"], model_no=9,
            confidence_scores=right_result["confidence"]
        )

        print(f"[LEFT] Writing result: {left_result}")
        print(f"[RIGHT] Writing result: {right_result}")
        #finish
        dbf.insert_endpoint_logs(referral_id, f"'{base_url}/predict-diabetes?referral_id={referral_id}", condition,
                                 left_id,
                                 right_id, 2)
        return {
            "referral_id": referral_id,
            "diagnosis": condition,
            "left_eye_result": left_result,
            "right_eye_result": right_result
        }
    except Exception as e:
        logging.error(f"Error during diagnosis: {e}")
        dbf.log_error_to_db(
            error_message="Error during full diagnosis",
            stack_trace=str(e),
            endpoint="/diagnoseAll"
        )
        raise HTTPException(status_code=500, detail=f"Error during full diagnosis: {str(e)}")


@condition_router.post("/handle-cnv")
async def handle_cnv(referral_id: str, image: UploadFile):
    image_bytes = await image.read()
    files = {"file": (image.filename, image_bytes, image.content_type)}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:8007/predict-cnv?referral_id={referral_id}",
            files=files
        )
    return response.json()


import httpx
from fastapi import UploadFile, HTTPException


@condition_router.post("/handle-myopia")
async def handle_myopia(referral_id: str, image: UploadFile):
    image_bytes = await image.read()
    files = {"file": (image.filename, image_bytes, image.content_type)}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://ppnai-prod:8009/predict-myopia?referral_id={referral_id}",  # no ?referral_id
                files=files
            )

        if response.status_code != 200:
            # Surface error message from the microservice
            detail = response.text[:300]
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Myopia microservice failed ({response.status_code}): {detail}"
            )

        # Ensure valid JSON response
        try:
            return response.json()
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"Myopia microservice returned invalid JSON: {response.text[:300]}"
            )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Connection to Myopia microservice failed: {str(e)}"
        )


# Pathology (macular) microservice URL (our FastAPI from macular_pathology_api.py)
PATHOLOGY_API = os.environ.get("PATHOLOGY_API_URL", "http://localhost:8006/predict")


@condition_router.post("/handle-pathological")
async def handle_pathological(referral_id: str, image: UploadFile) -> dict:
    """
    Proxy a single eye to the macular pathology microservice (default :8013/predict).
    Normalizes microservice JSON into your standard contract under key 'pathological'.
    """
    try:
        image_bytes = await image.read()
        files = {"file": (image.filename, image_bytes, image.content_type or "image/jpeg")}

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.post(f"{PATHOLOGY_API}?referral_id={referral_id}", files=files)
            resp.raise_for_status()
        raw = resp.json()

        # Prefer thresholded decision if present; fall back to 'top'
        decision = raw.get("decision") or {}
        top = raw.get("top") or {}

        pred_name = decision.get("pred_name") or top.get("name")
        ads_id = decision.get("ads_id") or top.get("ads_id")
        prob = decision.get("prob") or top.get("prob")

        if pred_name is None or ads_id is None:
            raise HTTPException(status_code=502, detail="Pathology service response missing required fields.")

        # Build confidence_scores from the 'predictions' array
        preds = raw.get("predictions", [])
        confidence_scores = {
            (p.get("name") or f"class_{p.get('index', i)}"): round(float(p.get("prob", 0.0)), 6)
            for i, p in enumerate(preds)
        }

        # Match your contract shape so write_eye_result_to_db(...) works:
        return {
            "pathological": {
                "prediction": pred_name,
                "ads_id": int(ads_id),
                "confidence_scores": confidence_scores
            },
            # Keep a filename field for parity with other endpoints (keras naming kept for compatibility)
            "keras_file_name": os.path.basename(os.environ.get("WEIGHTS", "pathology_best_model.pt")),
            # Optional: include microservice raw for debugging (safe to remove)
            # "raw": raw
        }

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Pathology service error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Formatting error (pathology): {e}")


@condition_router.post("/handle-rvo")
async def handle_rvo(referral_id: str, image: UploadFile) -> dict:
    """
    Proxy a single eye to the RVO microservice (8008),
    then normalize the JSON into your standard contract under key 'rvo'.
    """
    try:
        image_bytes = await image.read()
        files = {"image": (image.filename, image_bytes, image.content_type or "image/jpeg")}

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.post(
                f"http://localhost:8008/predict-rvo?referral_id={referral_id}",
                files=files,
            )
            resp.raise_for_status()
        raw = resp.json()

        # Expected microservice fields:
        #  - prediction: "Normal" or "RVO"
        #  - probabilities: {"Normal": float, "RVO": float}
        #  - ads_id: 24 (Normal) / 25 (RVO)
        pred_name = raw.get("prediction")
        probs = raw.get("probabilities", {})
        ads_id = raw.get("ads_id")

        # Fallback in case ads_id wasn’t returned:
        if ads_id is None and pred_name:
            ads_id = 24 if pred_name.lower() == "normal" else 25

        if pred_name is None or ads_id is None or not probs:
            raise HTTPException(status_code=502, detail="RVO service response missing required fields.")

        confidence_scores = {
            "Normal": round(float(probs.get("Normal", 0.0)), 6),
            "RVO": round(float(probs.get("RVO", 0.0)), 6),
        }

        return {
            "rvo": {
                "prediction": pred_name,
                "ads_id": int(ads_id),
                "confidence_scores": confidence_scores,
            },
            # keep for parity with other handlers
            "keras_file_name": os.path.basename(os.environ.get("RVO_MODEL_PATH", "rvo_best_model.pth")),
        }

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"RVO service error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Formatting error (rvo): {e}")


# Friendly label remaps for display (our model outputs "No", etc.)
_HUMAN_LABEL = {
    "No": "No Suspect or Negative",
    "DecreasedRetinalArteryElasticity": "Decreased Retinal Artery Elasticity",
    "HypertensiveRetinopathyGrade1or2": "Hypertensive Retinopathy Grade 1 or 2",
    "HypertensiveRetinopathyGrade3or4": "Hypertensive Retinopathy Grade 3 or 4",
}

KERAS_FILE_NAME = "hypertension_grading_model.keras"  # keep your contract


# def apply_pair_override(left_result: dict, right_result: dict) -> tuple:
#     """
#     Enforce rule: If either eye is labeled NO SUSPECT (ads_id = 7),
#     force BOTH eyes to NO SUSPECT — BUT keep original confidence scores untouched.
#     """
#     LEFT = left_result.get("hypertension", {})
#     RIGHT = right_result.get("hypertension", {})
#
#     left_ads = LEFT.get("ads_id")
#     right_ads = RIGHT.get("ads_id")
#
#     # Rule: If either eye is NO SUSPECT (ads_id = 7), override both
#     if left_ads == 7 or right_ads == 7:
#         # Force override on LEFT
#         LEFT["prediction"] = "No Suspect or Negative"
#         LEFT["ads_id"] = 7
#
#         # Force override on RIGHT
#         RIGHT["prediction"] = "No Suspect or Negative"
#         RIGHT["ads_id"] = 7
#         return left_result, right_result
#
#         # No changes to confidence_scores per Option B
#     if left_ads == 8 or right_ads == 8:
#         # Force override on LEFT
#         LEFT["prediction"] = "Decreased Retinal Artery Elasticity"
#         LEFT["ads_id"] = 8
#
#         # Force override on RIGHT
#         RIGHT["prediction"] = "Decreased Retinal Artery Elasticity"
#         RIGHT["ads_id"] = 8
#         return left_result, right_result
#
#     if left_ads == 9 or right_ads == 9:
#         # Force override on LEFT
#         LEFT["prediction"] = "Hypertensive Retinopathy Grade 1 or 2"
#         LEFT["ads_id"] = 9
#
#         # Force override on RIGHT
#         RIGHT["prediction"] = "Hypertensive Retinopathy Grade 1 or 2"
#         RIGHT["ads_id"] = 9
#
#     # Return updated structures
#     return left_result, right_result
# def apply_pair_override(left_result: dict, right_result: dict) -> tuple:
#     """
#     Pair override with priority logic:
#     🚨 Priority: HR3/4 (10) > HR1/2 (9) > DRAE (8, but only if ≥ 0.90) > No (7)
#     ✅ If DRAE ≥ 0.90 in either eye → OVERRIDE BOTH to DRAE regardless of other eye.
#     ✅ If HR1/2 or HR3/4 present → override both accordingly.
#     ✅ If No present but disease confidence is weak (< threshold), fallback remains No.
#     """
#
#     LEFT = left_result.get("hypertension", {})
#     RIGHT = right_result.get("hypertension", {})
#
#     left_ads = LEFT.get("ads_id")
#     right_ads = RIGHT.get("ads_id")
#     left_conf = LEFT.get("confidence_scores", {})
#     right_conf = RIGHT.get("confidence_scores", {})
#
#     # Extract class-wise probabilities
#     l_drae = left_conf.get("Decreased Retinal Artery Elasticity", 0.0)
#     r_drae = right_conf.get("Decreased Retinal Artery Elasticity", 0.0)
#
#     l_hr12 = left_conf.get("Hypertensive Retinopathy Grade 1 or 2", 0.0)
#     r_hr12 = right_conf.get("Hypertensive Retinopathy Grade 1 or 2", 0.0)
#
#     l_hr34 = left_conf.get("Hypertensive Retinopathy Grade 3 or 4", 0.0)
#     r_hr34 = right_conf.get("Hypertensive Retinopathy Grade 3 or 4", 0.0)
#
#     # 🚨 1️⃣ If either eye has Grade 3/4 ≥ 0.55 → override both to HR3/4 (ADS 10)
#     if l_hr34 >= 0.55 or r_hr34 >= 0.55:
#         LEFT["prediction"], RIGHT["prediction"] = "Hypertensive Retinopathy Grade 3 or 4", "Hypertensive Retinopathy Grade 3 or 4"
#         LEFT["ads_id"], RIGHT["ads_id"] = 10, 10
#         return left_result, right_result
#
#     # 2️⃣ If either eye has consistent Grade 1/2 ≥ 0.60 → override both to HR1/2 (ADS 9)
#     if l_hr12 >= 0.60 or r_hr12 >= 0.60:
#         LEFT["prediction"], RIGHT["prediction"] = "Hypertensive Retinopathy Grade 1 or 2", "Hypertensive Retinopathy Grade 1 or 2"
#         LEFT["ads_id"], RIGHT["ads_id"] = 9, 9
#         return left_result, right_result
#
#     # 🌟 NEW RULE 3️⃣ — If either eye has DRAE ≥ 0.90 → override both to DRAE (ADS 8)
#     if l_drae >= 0.90 or r_drae >= 0.90:
#         LEFT["prediction"], RIGHT["prediction"] = "Decreased Retinal Artery Elasticity", "Decreased Retinal Artery Elasticity"
#         LEFT["ads_id"], RIGHT["ads_id"] = 8, 8
#         return left_result, right_result
#
#     # 4️⃣ If ANY eye is "No" but no strong disease present → keep No
#     if left_ads == 7 or right_ads == 7:
#         LEFT["prediction"], RIGHT["prediction"] = "No Suspect or Negative", "No Suspect or Negative"
#         LEFT["ads_id"], RIGHT["ads_id"] = 7, 7
#         return left_result, right_result
#
#     return left_result, right_result
def apply_pair_override(left_result: dict, right_result: dict) -> tuple:
    L = left_result.get("hypertension", {})
    R = right_result.get("hypertension", {})
    # read predictions
    l_ads, r_ads = L.get("ads_id"), R.get("ads_id")
    # ✅ RULE: NO SUSPECT (ADS_ID = 7) DOMINATES EVERYTHING
    # if l_ads == 7 or r_ads == 7:
    #     L["prediction"] = "No Suspect or Negative"
    #     L["ads_id"] = 7
    #
    #     R["prediction"] = "No Suspect or Negative"
    #     R["ads_id"] = 7
    # ✅ If either eye is DRAE — DOMINATE
    l_lab, r_lab = L["prediction"], R["prediction"]
    if l_lab == "Decreased Retinal Artery Elasticity" or r_lab == "Decreased Retinal Artery Elasticity":
        L["prediction"] = R["prediction"] = "Decreased Retinal Artery Elasticity"
        L["ads_id"] = R["ads_id"] = 8
        return left_result, right_result

    # read per-eye probabilities safely
    lp = L.get("confidence_scores", {}) or {}
    rp = R.get("confidence_scores", {}) or {}

    def p(d, k):
        return float(d.get(k, 0.0))

    # helper for No / DRAE keys depending on your internal naming
    NO = "No Suspect or Negative"
    DRAE = "Decreased Retinal Artery Elasticity"

    # --- RULE 0: If either eye is clearly NO (No ≥ 0.50 and is top), do NOT propagate disease to that eye
    def is_clear_no(scores: dict) -> bool:
        top_lab, top_prob = max(scores.items(), key=lambda kv: kv[1]) if scores else (NO, 0.0)
        return (top_lab == NO) and (top_prob >= 0.50)

    # --- RULE 1: DRAE propagation only if source eye has strong DRAE
    def is_strong_drae(scores: dict) -> bool:
        if not scores: return False
        top_lab, top_prob = max(scores.items(), key=lambda kv: kv[1])
        return (top_lab == DRAE) and (top_prob >= 0.45)  # was 0.40; tighten to 0.45 for safety

    left_strong_drae = is_strong_drae(lp)
    right_strong_drae = is_strong_drae(rp)
    # If exactly one eye has strong DRAE, only propagate if the other eye is NOT clear No
    if left_strong_drae ^ right_strong_drae:
        if left_strong_drae and not is_clear_no(rp):
            L["prediction"], L["ads_id"] = DRAE, 8
            R["prediction"], R["ads_id"] = DRAE, 8
            return left_result, right_result
        if right_strong_drae and not is_clear_no(lp):
            L["prediction"], L["ads_id"] = DRAE, 8
            R["prediction"], R["ads_id"] = DRAE, 8
            return left_result, right_result
    # If both eyes strong DRAE → propagate
    if left_strong_drae and right_strong_drae:
        L["prediction"], L["ads_id"] = DRAE, 8
        R["prediction"], R["ads_id"] = DRAE, 8
        return left_result, right_result
    # (keep your other overrides here: HR3/4, HR1/2, final No, etc.)
    return left_result, right_result


@condition_router.post("/handle-hypertension")
async def handle_hypertension(referral_id: str, image: UploadFile) -> dict:
    """
    Proxy a single eye to the hypertension microservice (8003),
    then format the response into your expected schema (singular).
    """
    # 1) forward the file to the microservice
    image_bytes = await image.read()
    files = {"file": (image.filename, image_bytes, image.content_type)}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://ppnai-prod:8003/predict-hypertension?referral_id={referral_id}",
                files=files,
                timeout=160.0,
            )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        # bubble up as 502 so your outer try/except can log correctly
        raise HTTPException(status_code=502, detail=f"Hypertension service error: {e}")

    raw = resp.json()

    # 2) reshape to your contract
    try:
        # v3.11 single-eye schema
        if "probabilities" in raw:
            probs = raw["probabilities"]  # canonical keys like "No", "DecreasedRetinalArteryElasticity", ...
            # argmax from probs (ignore any 'best_ads_overridden')
            canon = {CANON.get(k, k): float(v) for k, v in probs.items()}
            label = max(canon, key=canon.get)
            ads_id = ADS[label]
        # legacy schema (keep as fallback)
        elif "results" in raw and raw["results"]:
            r0 = raw["results"][0]
            probs = r0.get("probabilities", {})
            best = r0.get("best_ads", {})
            if probs:
                canon = {CANON.get(k, k): float(v) for k, v in probs.items()}
                label = max(canon, key=canon.get)
                ads_id = ADS[label]
            else:
                label = CANON.get(best.get("label"), best.get("label"))
                ads_id = int(best.get("ads_id"))
        else:
            raise KeyError("Unsupported hypertension microservice response")

        # pretty display names
        confidence_scores = {_HUMAN_LABEL.get(k, k): round(float(v), 6) for k, v in probs.items()}

        return {
            "hypertension": {
                "prediction": _HUMAN_LABEL.get(label, label),
                "confidence_scores": confidence_scores,
                "ads_id": int(ads_id),
            },
            "keras_file_name": KERAS_FILE_NAME,
        }

    except Exception as e:
        # If the microservice schema changes, you'll see a clear error here
        raise HTTPException(status_code=500, detail=f"Formatting error (hypertension): {e}")


@condition_router.post("/handle-diabetes")
async def handle_diabetes(referral_id: str, image: UploadFile):
    image_bytes = await image.read()
    files = {"file": (image.filename, image_bytes, image.content_type)}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:8013/predict?referral_id={referral_id}",
            files=files
        )
    return response.json()


# Point this at your cdr_service.py instance
CDR_SERVICE_URL = os.environ.get(
    "CDR_SERVICE_URL",
    "http://ppnai-prod:8019/cdr/analyze"
)
from typing import Optional, Literal, Dict, Any

# --- endpoints (override in ENV if needed) ---
MODEL_API = os.environ.get("GLAUCOMA_MODEL_URL", "http://localhost:8012/predict-glaucoma")
CDR_API = os.environ.get("CDR_ANALYZE_URL", "http://localhost:8019/cdr/analyze")

# --- helpers ---------------------------------------------------------------

FUSE_CDR_THRESH = 0.75  # upgrade threshold on CDR
FUSE_CONF_MAX = 0.60  # only upgrade if model max-prob < 0.60


def _guess_eye_from_name(name: str) -> Literal["left", "right"]:
    """
    Heuristic for your file convention:
      - AI_131_xxxx.*  -> LEFT
      - AI_130_xxxx.*  -> RIGHT
    Fallback: 'right'
    """
    n = name.upper()
    if "AI_131" in n:
        return "left"
    if "AI_130" in n:
        return "right"
    return "right"


def _extract_referral_from_name(name: str) -> Optional[str]:
    """
    Extracts the 4-digit id between the 2nd underscore and extension, e.g.
    AI_131_3251.JPG -> '3251'. Returns None if not a 4-digit id.
    """
    m = re.search(r"AI_(?:130|131)_(\d{4})\b", name, flags=re.IGNORECASE)
    return m.group(1) if m else None


# --- core fusion for ONE eye ----------------------------------------------

from typing import Optional, Literal, Dict, Any
import httpx
from fastapi import HTTPException


async def _predict_one_eye_fused(
        referral_id: str,
        image_file,
        eye_side: Optional[Literal["left", "right"]] = None,
        g_folder: Optional[str] = None,
        db_image_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls :8012 model + :8019 CDR, applies fuse:
      - Use model ads_id by default (17=Glaucoma, 16=Normal)
      - If model says Normal (16) with confidence < 0.60 AND CDR >= 0.75 -> upgrade to 17
    Returns merged dict + 'fuse' section.
    """
    # read once; reuse bytes for both posts
    data = await image_file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")

    # 1) determine eye (respect explicit arg; fallback to guess; else 'auto')
    eye = eye_side or _guess_eye_from_name(image_file.filename or "") or "auto"

    # 2) resolve G folder and filename to use for overlays
    g_root = g_folder
    db_name_from_db = None
    if not g_root:
        try:
            # ensure referral_id is int for the DB helper
            g_root, db_name_from_db = dbf._db_get_g_folder(int(referral_id))
        except Exception:
            # soft-fail; g_root stays None and service will fall back to local overlays
            g_root, db_name_from_db = (g_root, None)

    save_name = db_image_name or db_name_from_db or (image_file.filename or "image.jpg")

    # ---------- call both services ----------
    async with httpx.AsyncClient(timeout=httpx.Timeout(40.0)) as client:
        # model (8012)
        model_files = {"file": (image_file.filename, data, getattr(image_file, "content_type", None) or "image/jpeg")}
        model_resp = await client.post(f"{MODEL_API}?referral_id={referral_id}", files=model_files)
        model_resp.raise_for_status()
        m = model_resp.json()

        # cdr (8019) — pass per-eye bias + G folder + original DB filename
        cdr_files = {"file": (image_file.filename, data, getattr(image_file, "content_type", None) or "image/jpeg")}
        cdr_form = {
            "bias": eye,  # left / right / auto
            "g_folder": g_root or "",  # forces saving into \\...\G (and any siblings like \6)
            "db_image_name": save_name or "",  # ensures overlay filenames match original
            "g_folder_f": g_root,
        }
        cdr_url = f"{CDR_API}?bias={eye}&referral_id={referral_id}&db_image_name_q={image_file.filename}&g_folder_q={g_root}"
        cdr_resp = await client.post(cdr_url, files=cdr_files, data=cdr_form, timeout=140.0)
        cdr_resp.raise_for_status()
        c = cdr_resp.json()

    # ---------- fuse logic ----------
    model_ads = int(m.get("ads_id", 16))
    g_prob = float(m.get("glaucoma_probability", 0.0))
    n_prob = float(m.get("normal_probability", 0.0))
    model_conf = max(g_prob, n_prob)
    model_pred = "Glaucoma" if (model_ads == 17 or g_prob > n_prob) else "Normal"
    FUSE_CONF_MAX = 0.60
    FUSE_CDR_MIN_UP = 0.75
    DOWNGRADE_CONF_MAX = 0.55
    DOWNGRADE_CDR_MAX = 0.35

    v_cdr = c.get("vertical_cdr")
    v_cdr = float(v_cdr) if isinstance(v_cdr, (int, float)) else None

    final_ads = model_ads
    final_pred = model_pred
    fuse_applied = False
    fuse_reason = None

    if model_ads == 16 and model_conf < FUSE_CONF_MAX and (v_cdr is not None) and (v_cdr >= FUSE_CDR_THRESH):
        final_ads = 17
        final_pred = "Glaucoma"
        fuse_applied = True
        fuse_reason = f"Upgraded by CDR safeguard (CDR={v_cdr:.2f} ≥ {FUSE_CDR_THRESH}, model_conf={model_conf:.2f} < {FUSE_CONF_MAX})."

    # ---------- merged payload ----------
    merged = {
        "eye": eye,
        "filename": image_file.filename,
        "referral_id": referral_id,
        "model": {
            "glaucoma_probability": g_prob,
            "normal_probability": n_prob,
            "prediction": model_pred,
            "ads_id": model_ads,
            "overlay_url": m.get("overlay_url"),
            "cdr_analysis": m.get("cdr_analysis"),
        },
        "cdr": {
            "vertical_cdr": v_cdr,
            "clinical_label": c.get("clinical_label"),
            "overlay_url_full": c.get("overlay_url_full"),
            "overlay_url_crop": c.get("overlay_url_crop"),
            "raw": c,
        },
        "final": {
            "prediction": final_pred,
            "ads_id": final_ads,
            "confidence": round(model_conf, 4),
        },
        "fuse": {
            "applied": fuse_applied,
            "reason": fuse_reason,
            "rules": {
                "if_model_normal_and_conf_lt": FUSE_CONF_MAX,
                "and_cdr_ge": FUSE_CDR_THRESH,
                "then_ads_id": 17
            }
        }
    }
    return merged


def _full_path(loc: Optional[str], name: Optional[str]) -> str:
    """
    Join a directory and filename safely (Windows UNC or *nix),
    trimming only *trailing* separators on the dir and *leading*
    separators on the file.
    """
    loc = (loc or "").rstrip("\\/")  # keep leading \\ for UNC
    name = (name or "").lstrip("\\/")  # avoid accidental // or \\
    return os.path.join(loc, name) if loc else name


def _eye_from_filename(name: str) -> Optional[str]:
    n = (name or "").lower()
    if "_130_" in n:  # your convention
        return "right"
    if "_131_" in n:
        return "left"
    return None


def _ai_id_for_eye(eye: str) -> int:
    return 131 if eye == "left" else 130  # 131 = LEFT, 130 = RIGHT


async def predict_both_eyes_fused_by_referral(referral_id: int) -> dict:
    """
    Loads both eyes from AI_IMAGES (130=right, 131=left) and runs your existing
    _predict_one_eye_fused pipeline for each, saving overlays in the same G folder.
    """
    # Resolve once so both eyes land in the same folder
    g_root, _ = dbf._db_get_g_folder(referral_id)

    rows = dbf.db_get_referral_images(referral_id)  # [(name, loc, oculus)]
    if not rows:
        raise HTTPException(status_code=404, detail=f"No images for referral_id={referral_id}")

    out = {"referral_id": referral_id, "eyes": []}

    async with httpx.AsyncClient(timeout=httpx.Timeout(40.0)) as client:
        for name, loc, oculus in rows:
            # Load bytes
            path = _full_path(loc, name)
            with open(path, "rb") as f:
                data = f.read()

            eye_side = "left" if oculus == 131 else "right"

            # --- model (8012)
            model_files = {"file": (name, data, "image/jpeg")}
            model_resp = await client.post(f"{MODEL_API}?referral_id={referral_id}", files=model_files)
            model_resp.raise_for_status()
            m = model_resp.json()

            # --- cdr (8019) — pass G and original DB name so filenames stay exact
            cdr_files = {"file": (name, data, "image/jpeg")}
            cdr_form = {"bias": eye_side, "g_folder": g_root or "", "db_image_name": name}
            cdr_resp = await client.post(CDR_API, files=cdr_files, data=cdr_form)
            cdr_resp.raise_for_status()
            c = cdr_resp.json()

            # --- fuse rule (reuse your existing logic)
            model_ads = int(m.get("ads_id", 16))
            g_prob = float(m.get("glaucoma_probability", 0.0))
            n_prob = float(m.get("normal_probability", 0.0))
            model_conf = max(g_prob, n_prob)
            model_pred = "Glaucoma" if model_ads == 17 or g_prob > n_prob else "Normal"
            v_cdr = c.get("vertical_cdr")
            v_cdr = float(v_cdr) if isinstance(v_cdr, (int, float)) else None

            final_ads = model_ads
            final_pred = model_pred
            fuse_reason = None
            if model_ads == 16 and model_conf < FUSE_CONF_MAX and (v_cdr is not None) and (v_cdr >= FUSE_CDR_THRESH):
                final_ads = 17
                final_pred = "Glaucoma"
                fuse_applied = True
                fuse_reason = (
                    f"Upgraded by CDR safeguard (CDR={v_cdr:.2f} ≥ {FUSE_CDR_THRESH}, "
                    f"model_conf={model_conf:.2f} < {FUSE_CONF_MAX})."
                )

            out["eyes"].append({
                "eye": eye_side,
                "filename": name,
                "model": {
                    "glaucoma_probability": g_prob,
                    "normal_probability": n_prob,
                    "prediction": model_pred,
                    "ads_id": model_ads,
                    "overlay_url": m.get("overlay_url"),
                    "cdr_analysis": m.get("cdr_analysis"),
                },
                "cdr": {
                    "vertical_cdr": v_cdr,
                    "overlay_url_full": c.get("overlay_url_full"),
                    "overlay_url_crop": c.get("overlay_url_crop"),
                    "clinical_label": c.get("clinical_label"),
                    "raw": c,
                },
                "final": {
                    "prediction": final_pred,
                    "ads_id": final_ads,
                    "confidence": round(model_conf, 4),
                },
                "fuse": {
                    "applied": final_ads != model_ads,
                    "reason": fuse_reason,
                }
            })

    return out


@condition_router.post("/handle-amd")
async def handle_amd(referral_id: str, image: UploadFile) -> dict:
    """
    Proxy a single eye to the AMD microservice (8004),
    then format the response into your expected schema (singular).
    """
    # 1) forward the file to the microservice
    image_bytes = await image.read()
    files = {"file": (image.filename, image_bytes, image.content_type)}

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://localhost:8004/predict-macular-degeneration?referral_id={referral_id}",
                files=files,
                timeout=60.0,
            )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"AMD service error: {e}")

    raw = resp.json()

    # 2) reshape to your contract
    try:
        # top-level fields from AMD service
        top_label = raw["top_label"]
        top_index = raw["top_index"]
        top_probability = raw["top_probability"]
        ads = raw.get("ads", {})

        # pretty confidence score keys
        confidence_scores = {
            _HUMAN_LABEL.get(p["label"], p["label"]): round(float(p["probability"]), 6)
            for p in raw.get("probs", [])
        }

        formatted = {
            "amd": {
                "prediction": _HUMAN_LABEL.get(top_label, top_label),
                "confidence_scores": confidence_scores,
                "ads_id": int(ads.get("ADS_ID")) if ads.get("ADS_ID") else None,
            },
            "keras_file_name": KERAS_FILE_NAME,
        }
        return formatted

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Formatting error (amd): {e}")


@condition_router.post("/handle-glaucoma")
async def handle_glaucoma(referral_id: str, image: UploadFile = File(...),
                          eye_side: Optional[str] = None,
                          g_folder: Optional[str] = None,
                          db_image_name: Optional[str] = None, ):
    """
    Replacement for your earlier handle_glaucoma:
    - Calls both services
    - Applies fuse
    - Returns merged payload with final.ads_id (17=Glaucoma, 16=Normal)
    """
    # if you want to auto-correct referral id from filename when missing:
    if not referral_id:
        rid = _extract_referral_from_name(image.filename or "")
        if rid:
            referral_id = rid

    # _predict_one_eye_fused already has `eye` param in your earlier snippet;
    # add optional g_folder/db_image_name passthrough.
    result = await _predict_one_eye_fused(
        referral_id=referral_id,
        image_file=image,
        eye_side=eye_side,
        g_folder=g_folder,
        db_image_name=db_image_name,
    )
    return result


# condition_routes_hypertension.py
from fastapi import APIRouter, UploadFile, File, HTTPException
import httpx, logging, json
from typing import Dict

# import your app-specific helpers
# from your_project.db_funcs import dbf
# from your_project.config import base_url, get_ad_id

condition_router = APIRouter()

# === Canonical labels & ADS mapping (single source of truth) ===
CANON = {
    "No": "NoSuspectOrNegative",
    "No Suspect or Negative": "NoSuspectOrNegative",
    "NoSuspectOrNegative": "NoSuspectOrNegative",
    "Decreased Retinal Artery Elasticity": "DecreasedRetinalArteryElasticity",
    "DecreasedRetinalArteryElasticity": "DecreasedRetinalArteryElasticity",
    "Hypertensive Retinopathy Grade 1 or 2": "HypertensiveRetinopathyGrade1or2",
    "HypertensiveRetinopathyGrade1or2": "HypertensiveRetinopathyGrade1or2",
    "Hypertensive Retinopathy Grade 3 or 4": "HypertensiveRetinopathyGrade3or4",
    "HypertensiveRetinopathyGrade3or4": "HypertensiveRetinopathyGrade3or4",
}
ADS = {
    "NoSuspectOrNegative": 7,
    "DecreasedRetinalArteryElasticity": 8,
    "HypertensiveRetinopathyGrade1or2": 9,
    "HypertensiveRetinopathyGrade3or4": 10,
}


def _argmax_from_probs(prob_map: Dict[str, float]) -> Dict[str, object]:
    """Normalize keys then argmax → label, ads_id, prob."""
    if not prob_map:
        return {"label": "NoSuspectOrNegative", "ads_id": ADS["NoSuspectOrNegative"], "prob": 0.0}
    canon = {CANON.get(k, k): float(v) for k, v in prob_map.items()}
    label = max(canon, key=canon.get)
    return {"label": label, "ads_id": ADS[label], "prob": canon[label]}


def _pair_priority(left_label: str, right_label: str) -> Dict[str, object]:
    priority = [
        "HypertensiveRetinopathyGrade3or4",
        "HypertensiveRetinopathyGrade1or2",
        "DecreasedRetinalArteryElasticity",
        "NoSuspectOrNegative",
    ]
    labels_present = {left_label, right_label}
    for lab in priority:
        if lab in labels_present:
            return {"pair_label": lab, "pair_ads": ADS[lab], "rule": "PAIR:PRIORITY"}
    return {"pair_label": "NoSuspectOrNegative", "pair_ads": ADS["NoSuspectOrNegative"], "rule": "PAIR:FALLBACK"}


MICRO_URL = "http://localhost:8003"
PAIR_EP = f"{MICRO_URL}/predict-hypertension/pair"
HEALTHZ = f"{MICRO_URL}/healthz"


@condition_router.post("/predict-hypertension")
async def predict_hypertension(
        referral_id: str,
        left_eye_file: UploadFile = File(...),
        right_eye_file: UploadFile = File(...)
):
    try:
        condition = "hypertension"
        model_no = 9
        ads_ad_id = get_ad_id(condition)  # family id in your ADS system

        # Your resolver for AI image ids (replace if you have a lookup)
        left_id = 131
        right_id = 130

        # Read once
        l_bytes = await left_eye_file.read()
        r_bytes = await right_eye_file.read()
        l_ct = left_eye_file.content_type or "application/octet-stream"
        r_ct = right_eye_file.content_type or "application/octet-stream"

        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        timeout = httpx.Timeout(connect=3.0, read=60.0, write=30.0, pool=3.0)

        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            # health gate (optional)
            # try:
            #     h = await client.get(HEALTHZ, timeout=3.0)
            #     if h.status_code != 200:
            #         raise HTTPException(status_code=502, detail=f"HTN /healthz bad status {h.status_code}")
            # except httpx.HTTPError as e:
            #     raise HTTPException(status_code=502, detail=f"HTN /healthz error: {e}")

            files = {
                "left_eye_file": (left_eye_file.filename or "left.jpg", l_bytes, l_ct),
                "right_eye_file": (right_eye_file.filename or "right.jpg", r_bytes, r_ct),
            }

            resp = await client.post(PAIR_EP, files=files)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            result = resp.json()

        # ── BYPASS microservice overrides: always compute from probabilities ──
        left_node = result.get("left", {})
        right_node = result.get("right", {})

        left_probs = left_node.get("probabilities") or left_node.get("probs") or {}
        right_probs = right_node.get("probabilities") or right_node.get("probs") or {}

        # left_best  = _argmax_from_probs(left_probs)
        # right_best = _argmax_from_probs(right_probs)
        # left_best = choose_with_thresholds(left_conf)
        # right_best = choose_with_thresholds(right_conf)
        left_best = left_node.get("best_ads_overridden") or {}
        right_best = right_node.get("best_ads_overridden") or {}

        left_label, left_ads = left_best["label"], int(left_best["ads_id"])
        right_label, right_ads = right_best["label"], int(right_best["ads_id"])

        # DB writes (store raw confidence JSON for audit)
        if left_id:
            dbf.write_results_to_db(
                referral_id,
                ai_id=left_id,
                ads_ad_id=ads_ad_id,
                ads_id=left_ads,
                model_no=model_no,
                confidence_scores=json.dumps(left_probs),
            )
        if right_id:
            dbf.write_results_to_db(
                referral_id,
                ai_id=right_id,
                ads_ad_id=ads_ad_id,
                ads_id=right_ads,
                model_no=model_no,
                confidence_scores=json.dumps(right_probs),
            )

        # log the call
        dbf.insert_endpoint_logs(
            referral_id,
            f"{base_url}/predict-hypertension?referral_id={referral_id}",
            condition,
            left_id,
            right_id,
            model_no,
        )

        pair = _pair_priority(left_label, right_label)

        return {
            "referral_id": referral_id,
            "diagnosis": condition,
            "left_eye_result": {"ads_id": left_ads, "label": left_label, "confidence": left_probs},
            "right_eye_result": {"ads_id": right_ads, "label": right_label, "confidence": right_probs},
            "pair_summary": pair,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[Hypertension] Failed referral {referral_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# @condition_router.post("/predict-hypertension")
# async def predict_hypertension(
#
#         referral_id: str,
#
#         left_eye_file: UploadFile = File(...),
#
#         right_eye_file: UploadFile = File(...)
#
# ):
#     try:
#
#         condition = "hypertension"
#         model_no = 9  # your registry model number
#         ad_id = get_ad_id(condition)
#         # DB image IDs (replace with your resolver if needed)
#         left_id = 131
#         right_id = 130
#
#         # Resolve destination folder (unused here but preserved)
#         g_folder, _ = dbf._db_get_g_folder(int(referral_id))
#
#         # --- call the hypertension microservice (pair endpoint) ---
#         url = "http://localhost:8003/predict-hypertension/pair"
#
#         # use client-provided content types if available
#         l_ct = left_eye_file.content_type or "application/octet-stream"
#         r_ct = right_eye_file.content_type or "application/octet-stream"
#
#         l_bytes = await left_eye_file.read()
#         r_bytes = await right_eye_file.read()
#
#         files = {
#             "left_eye_file": (left_eye_file.filename, l_bytes, l_ct),
#             "right_eye_file": (right_eye_file.filename, r_bytes, r_ct),
#         }
#
#         import httpx
#
#         async with httpx.AsyncClient(timeout=None) as client:
#
#             resp = await client.post(url, files=files)
#
#         if resp.status_code != 200:
#             # bubble up useful detail from the microservice
#
#             raise HTTPException(status_code=resp.status_code, detail=resp.text)
#
#         result = resp.json()
#
#         # ---- MICROservice schema (as returned by the code I gave you):
#
#         # {
#
#         #   "left":  {"file_name": "...", "probabilities": {...}, "best_ads_overridden": {"ads_id": int, "label": str}, "debug": {...}},
#
#         #   "right": {"file_name": "...", "probabilities": {...}, "best_ads_overridden": {"ads_id": int, "label": str}, "debug": {...}},
#
#         #   "pair_debug": {"before_override": [...], "after_override": str, "ads_id": int, "rule": str},
#
#         #   "model": {...}
#
#         # }
#
#         #
#
#         # --- Extract left/right finals (correct keys)
#
#         left_node = result["left"]
#
#         right_node = result["right"]
#
#         left_final = left_node["best_ads_overridden"]
#
#         right_final = right_node["best_ads_overridden"]
#
#         left_ads_id = left_final["ads_id"]
#
#         right_ads_id = right_final["ads_id"]
#
#         left_label = left_final["label"]
#
#         right_label = right_final["label"]
#
#         # --- Confidence maps (probabilities)
#
#         left_conf = left_node["probabilities"]
#
#         right_conf = right_node["probabilities"]
#
#         # --- Write to DB (guard for None/0)
#
#         if left_id and left_ads_id is not None:
#             dbf.write_results_to_db(
#
#                 referral_id,
#
#                 ai_id=left_id,
#
#                 ads_ad_id=ad_id,
#
#                 ads_id=left_ads_id,
#
#                 model_no=model_no,
#
#                 confidence_scores=left_conf
#
#             )
#
#         if right_id and right_ads_id is not None:
#             dbf.write_results_to_db(
#
#                 referral_id,
#
#                 ai_id=right_id,
#
#                 ads_ad_id=ad_id,
#
#                 ads_id=right_ads_id,
#
#                 model_no=model_no,
#
#                 confidence_scores=right_conf
#
#             )
#
#         # --- Log the endpoint call
#
#         dbf.insert_endpoint_logs(
#
#             referral_id,
#
#             f"{base_url}/predict-hypertension?referral_id={referral_id}",
#
#             condition,
#
#             left_id,
#
#             right_id,
#
#             model_no
#
#         )
#
#         # --- Final response to caller (align with your existing contract)
#
#         pair_dbg = result.get("pair_debug")  # not "pair_decision"
#
#         return {
#
#             "referral_id": referral_id,
#
#             "diagnosis": condition,
#
#             "left_eye_result": {
#
#                 "ads_id": left_ads_id,
#
#                 "label": left_label,
#
#                 "confidence": left_conf
#
#             },
#
#             "right_eye_result": {
#
#                 "ads_id": right_ads_id,
#
#                 "label": right_label,
#
#                 "confidence": right_conf
#
#             },
#
#             "pair_summary": pair_dbg  # contains after_override, ads_id, rule, etc.
#
#         }
#
#     except HTTPException:
#
#         # re-raise upstream http errors as-is
#
#         raise
#
#     except Exception as e:
#
#         logging.error(f"[Hypertension] Failed to process referral {referral_id}: {e}", exc_info=True)
#
#         raise HTTPException(status_code=500, detail=str(e))


# @condition_router.post("/predict-macular-degeneration")
# async def predict_macular_degeneration(
#        referral_id: str,
#         left_eye_file: UploadFile = File(...),
#         right_eye_file: UploadFile = File(...)
# ):
#     condition = "macular_degeneration"
#     ad_id = 4
#     model_id = 4
#     left_id, right_id = [130, 131]
#     # start
#     dbf.insert_endpoint_logs(referral_id, f"{base_url}/predict-{condition}?referral_id={referral_id}", condition, left_id,
#                          right_id, 4)

#     left_result =  await predict_generic_condition(left_eye_file, condition,"left", referral_id )
#     write_eye_result_to_db('left', left_id, left_result, condition, referral_id, model_id)
#     right_result = await predict_generic_condition(right_eye_file, condition, "right", referral_id)
#     write_eye_result_to_db('right', right_id, right_result, condition, referral_id, model_id)
#     #finish
#     dbf.insert_endpoint_logs(referral_id, f"{base_url}/predict-{condition}?referral_id={referral_id}", condition, left_id,
#                          right_id,4)
#     return {
#         "referral_id": referral_id,
#         "diagnosis": "macular_degeneration",
#         "left_eye_result": left_result,
#         "right_eye_result": right_result
#     }
@condition_router.post("/predict-macular-degeneration")
async def predict_macular_degeneration(
        referral_id: str,
        left_eye_file: UploadFile = File(...),
        right_eye_file: UploadFile = File(...)
):
    condition = "macular_degeneration"
    ad_id = 4
    model_id = 4
    left_id, right_id = [130, 131]
    # start
    dbf.insert_endpoint_logs(referral_id, f"{base_url}/predict-{condition}?referral_id={referral_id}", condition,
                             left_id,
                             right_id, 4)

    left_result = await handle_amd(referral_id, left_eye_file)
    # normalize to your contract (key == condition)
    left_result = {
        condition: left_result.get("amd", left_result),  # unwrap the "amd" block
        "keras_file_name": left_result.get("keras_file_name"),
    }
    write_eye_result_to_db('left', left_id, left_result, condition, referral_id, model_id)
    right_result = await handle_amd(referral_id, right_eye_file)
    right_result = {
        condition: right_result.get("amd", right_result),
        "keras_file_name": right_result.get("keras_file_name"),
    }
    write_eye_result_to_db('right', right_id, right_result, condition, referral_id, model_id)
    #finish
    dbf.insert_endpoint_logs(referral_id, f"{base_url}/predict-{condition}?referral_id={referral_id}", condition,
                             left_id,
                             right_id, 4)
    return {
        "referral_id": referral_id,
        "diagnosis": "macular_degeneration",
        "left_eye_result": left_result,
        "right_eye_result": right_result
    }


@condition_router.post("/predict-cnv")
async def predict_cnv(
        referral_id: str,
        left_eye_file: UploadFile = File(...),
        right_eye_file: UploadFile = File(...)
):
    condition = "cnv"
    ad_id = 7
    model_id = 7
    left_id, right_id = [130, 131]
    # start
    dbf.insert_endpoint_logs(referral_id, f"{base_url}/predict-{condition}?referral_id={referral_id}", condition,
                             left_id,
                             right_id, 7)

    #left_result =  await predict_generic_condition(left_eye_file, condition,"left", referral_id )
    left_result = await handle_cnv(referral_id, left_eye_file)
    write_eye_result_to_db('left', left_id, left_result, condition, referral_id, model_id)
    #right_result = await predict_generic_condition(right_eye_file, condition, "right", referral_id)
    right_result = await handle_cnv(referral_id, right_eye_file)
    write_eye_result_to_db('right', right_id, right_result, condition, referral_id, model_id)
    #finish
    dbf.insert_endpoint_logs(referral_id, f"{base_url}/predict-{condition}?referral_id={referral_id}", condition,
                             left_id,
                             right_id, 7)
    return {
        "referral_id": referral_id,
        "diagnosis": "cnv",
        "left_eye_result": left_result,
        "right_eye_result": right_result
    }


@condition_router.post("/predict-rvo")
async def predict_rvo(
        referral_id: str,
        left_eye_file: UploadFile = File(...),
        right_eye_file: UploadFile = File(...)
):
    condition = "rvo"
    ad_id = get_ad_id(condition)
    model_id = 6
    left_id, right_id = [130, 131]  #left_id, right_id = dbf.get_left_and_right_eye_image_ids(referral_id)
    # start
    dbf.insert_endpoint_logs(referral_id, f"{base_url}/predict-{condition}?referral_id={referral_id}", condition,
                             left_id,
                             right_id, 8)

    #left_result =  await predict_generic_condition(left_eye_file, condition, "left", referral_id )
    left_result = await handle_rvo(referral_id, left_eye_file)
    write_eye_result_to_db('left', left_id, left_result, condition, referral_id, model_id)
    #right_result = await predict_generic_condition(right_eye_file, condition, "right", referral_id)
    right_result = await handle_rvo(referral_id, right_eye_file)
    write_eye_result_to_db('right', right_id, right_result, condition, referral_id, model_id)
    #finish
    dbf.insert_endpoint_logs(referral_id, f"{base_url}/predict-{condition}?referral_id={referral_id}", condition,
                             left_id,
                             right_id, 8)
    return {
        "referral_id": referral_id,
        "diagnosis": "rvo",
        "left_eye_result": left_result,
        "right_eye_result": right_result
    }


from fastapi import HTTPException


# Point this at your cdr_service.py instance
CDR_SERVICE_URL = os.environ.get(
    "CDR_SERVICE_URL",
    "http://ppnai-prod:8019/cdr/analyze"
)
from typing import Optional, Literal, Dict, Any
# --- endpoints (override in ENV if needed) ---
MODEL_API = os.environ.get("GLAUCOMA_MODEL_URL", "http://ppnai-prod:8012/predict-glaucoma")
CDR_API   = os.environ.get("CDR_ANALYZE_URL",     "http://ppnai-prod:8019/cdr/analyze")

# --- helpers ---------------------------------------------------------------

FUSE_CDR_THRESH = 0.75   # upgrade threshold on CDR
FUSE_CONF_MAX   = 0.60   # only upgrade if model max-prob < 0.60

def _guess_eye_from_name(name: str) -> Literal["left", "right"]:
    """
    Heuristic for your file convention:
      - AI_131_xxxx.*  -> LEFT
      - AI_130_xxxx.*  -> RIGHT
    Fallback: 'right'
    """
    n = name.upper()
    if "AI_131" in n:
        return "left"
    if "AI_130" in n:
        return "right"
    return "right"

def _extract_referral_from_name(name: str) -> Optional[str]:
    """
    Extracts the 4-digit id between the 2nd underscore and extension, e.g.
    AI_131_3251.JPG -> '3251'. Returns None if not a 4-digit id.
    """
    m = re.search(r"AI_(?:130|131)_(\d{4})\b", name, flags=re.IGNORECASE)
    return m.group(1) if m else None

# --- core fusion for ONE eye ----------------------------------------------

from typing import Optional, Literal, Dict, Any
import httpx
from fastapi import HTTPException

async def _predict_one_eye_fused(
    referral_id: str,
    image_file,
    eye_side: Optional[Literal["left", "right"]] = None,
    g_folder: Optional[str] = None,
    db_image_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls :8012 model + :8019 CDR, applies fuse:
      - Use model ads_id by default (17=Glaucoma, 16=Normal)
      - If model says Normal (16) with confidence < 0.60 AND CDR >= 0.75 -> upgrade to 17
    Returns merged dict + 'fuse' section.
    """
    # read once; reuse bytes for both posts
    data = await image_file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")

    # 1) determine eye (respect explicit arg; fallback to guess; else 'auto')
    eye = eye_side or _guess_eye_from_name(image_file.filename or "") or "auto"

    # 2) resolve G folder and filename to use for overlays
    g_root = g_folder
    db_name_from_db = None
    if not g_root:
        try:
            # ensure referral_id is int for the DB helper
            g_root, db_name_from_db = dbf._db_get_g_folder(int(referral_id))
        except Exception:
            # soft-fail; g_root stays None and service will fall back to local overlays
            g_root, db_name_from_db = (g_root, None)

    save_name = db_image_name or db_name_from_db or (image_file.filename or "image.jpg")

    # ---------- call both services ----------
    async with httpx.AsyncClient(timeout=httpx.Timeout(40.0)) as client:
        # model (8012)
        model_files = {"file": (image_file.filename, data, getattr(image_file, "content_type", None) or "image/jpeg")}
        model_resp = await client.post(f"{MODEL_API}?referral_id={referral_id}", files=model_files)
        model_resp.raise_for_status()
        m = model_resp.json()

        # cdr (8019) — pass per-eye bias + G folder + original DB filename
        cdr_files = {"file": (image_file.filename, data, getattr(image_file, "content_type", None) or "image/jpeg")}
        cdr_form  = {
            "bias": eye,                       # left / right / auto
            "g_folder": g_root or "",          # forces saving into \\...\G (and any siblings like \6)
            "db_image_name": save_name or "",  # ensures overlay filenames match original
            "g_folder_f":g_root,
        }
        cdr_url = f"{CDR_API}?bias={eye}&referral_id={referral_id}&db_image_name_q={image_file.filename}&g_folder_q={g_root}"
        cdr_resp = await client.post(cdr_url, files=cdr_files, data=cdr_form, timeout=140.0)
        cdr_resp.raise_for_status()
        c = cdr_resp.json()

    # ---------- fuse logic ----------
    model_ads   = int(m.get("ads_id", 16))
    g_prob      = float(m.get("glaucoma_probability", 0.0))
    n_prob      = float(m.get("normal_probability",   0.0))
    model_conf  = max(g_prob, n_prob)
    model_pred  = "Glaucoma" if (model_ads == 17 or g_prob > n_prob) else "Normal"
    FUSE_CONF_MAX       = 0.60
    FUSE_CDR_MIN_UP     = 0.75
    DOWNGRADE_CONF_MAX  = 0.55
    DOWNGRADE_CDR_MAX   = 0.35

    v_cdr = c.get("vertical_cdr")
    v_cdr = float(v_cdr) if isinstance(v_cdr, (int, float)) else None

    final_ads    = model_ads
    final_pred   = model_pred
    fuse_applied = False
    fuse_reason  = None

    if model_ads == 16 and model_conf < FUSE_CONF_MAX and (v_cdr is not None) and (v_cdr >= FUSE_CDR_THRESH):
        final_ads    = 17
        final_pred   = "Glaucoma"
        fuse_applied = True
        fuse_reason  = f"Upgraded by CDR safeguard (CDR={v_cdr:.2f} ≥ {FUSE_CDR_THRESH}, model_conf={model_conf:.2f} < {FUSE_CONF_MAX})."

    # ---------- merged payload ----------
    merged = {
        "eye": eye,
        "filename": image_file.filename,
        "referral_id": referral_id,
        "model": {
            "glaucoma_probability": g_prob,
            "normal_probability": n_prob,
            "prediction": model_pred,
            "ads_id": model_ads,
            "overlay_url": m.get("overlay_url"),
            "cdr_analysis": m.get("cdr_analysis"),
        },
        "cdr": {
            "vertical_cdr": v_cdr,
            "clinical_label": c.get("clinical_label"),
            "overlay_url_full": c.get("overlay_url_full"),
            "overlay_url_crop": c.get("overlay_url_crop"),
            "raw": c,
        },
        "final": {
            "prediction": final_pred,
            "ads_id": final_ads,
            "confidence": round(model_conf, 4),
        },
        "fuse": {
            "applied": fuse_applied,
            "reason": fuse_reason,
            "rules": {
                "if_model_normal_and_conf_lt": FUSE_CONF_MAX,
                "and_cdr_ge": FUSE_CDR_THRESH,
                "then_ads_id": 17
            }
        }
    }
    return merged


def _full_path(loc: Optional[str], name: Optional[str]) -> str:
    """
    Join a directory and filename safely (Windows UNC or *nix),
    trimming only *trailing* separators on the dir and *leading*
    separators on the file.
    """
    loc  = (loc or "").rstrip("\\/")         # keep leading \\ for UNC
    name = (name or "").lstrip("\\/")        # avoid accidental // or \\
    return os.path.join(loc, name) if loc else name

def _eye_from_filename(name: str) -> Optional[str]:
    n = (name or "").lower()
    if "_130_" in n:  # your convention
        return "right"
    if "_131_" in n:
        return "left"
    return None

def _ai_id_for_eye(eye: str) -> int:
    return 131 if eye == "left" else 130  # 131 = LEFT, 130 = RIGHT

async def predict_both_eyes_fused_by_referral(referral_id: int) -> dict:
    """
    Loads both eyes from AI_IMAGES (130=right, 131=left) and runs your existing
    _predict_one_eye_fused pipeline for each, saving overlays in the same G folder.
    """
    # Resolve once so both eyes land in the same folder
    g_root, _ = dbf._db_get_g_folder(referral_id)

    rows = dbf.db_get_referral_images(referral_id)  # [(name, loc, oculus)]
    if not rows:
        raise HTTPException(status_code=404, detail=f"No images for referral_id={referral_id}")

    out = {"referral_id": referral_id, "eyes": []}

    async with httpx.AsyncClient(timeout=httpx.Timeout(40.0)) as client:
        for name, loc, oculus in rows:
            # Load bytes
            path = _full_path(loc, name)
            with open(path, "rb") as f:
                data = f.read()

            eye_side = "left" if oculus == 131 else "right"

            # --- model (8012)
            model_files = {"file": (name, data, "image/jpeg")}
            model_resp = await client.post(f"{MODEL_API}?referral_id={referral_id}", files=model_files)
            model_resp.raise_for_status()
            m = model_resp.json()

            # --- cdr (8019) — pass G and original DB name so filenames stay exact
            cdr_files = {"file": (name, data, "image/jpeg")}
            cdr_form  = {"bias": eye_side, "g_folder": g_root or "", "db_image_name": name}
            cdr_resp  = await client.post(CDR_API, files=cdr_files, data=cdr_form)
            cdr_resp.raise_for_status()
            c = cdr_resp.json()

            # --- fuse rule (reuse your existing logic)
            model_ads  = int(m.get("ads_id", 16))
            g_prob     = float(m.get("glaucoma_probability", 0.0))
            n_prob     = float(m.get("normal_probability",   0.0))
            model_conf = max(g_prob, n_prob)
            model_pred = "Glaucoma" if model_ads == 17 or g_prob > n_prob else "Normal"
            v_cdr      = c.get("vertical_cdr")
            v_cdr      = float(v_cdr) if isinstance(v_cdr, (int, float)) else None

            final_ads    = model_ads
            final_pred   = model_pred
            fuse_reason  = None
            if model_ads == 16 and model_conf < FUSE_CONF_MAX and (v_cdr is not None) and (v_cdr >= FUSE_CDR_THRESH):
                final_ads = 17
                final_pred = "Glaucoma"
                fuse_applied = True
                fuse_reason = (
                    f"Upgraded by CDR safeguard (CDR={v_cdr:.2f} ≥ {FUSE_CDR_THRESH}, "
                    f"model_conf={model_conf:.2f} < {FUSE_CONF_MAX})."
                )
                
            out["eyes"].append({
                "eye": eye_side,
                "filename": name,
                "model": {
                    "glaucoma_probability": g_prob,
                    "normal_probability":   n_prob,
                    "prediction": model_pred,
                    "ads_id": model_ads,
                    "overlay_url": m.get("overlay_url"),
                    "cdr_analysis": m.get("cdr_analysis"),
                },
                "cdr": {
                    "vertical_cdr": v_cdr,
                    "overlay_url_full": c.get("overlay_url_full"),
                    "overlay_url_crop": c.get("overlay_url_crop"),
                    "clinical_label": c.get("clinical_label"),
                    "raw": c,
                },
                "final": {
                    "prediction": final_pred,
                    "ads_id": final_ads,
                    "confidence": round(model_conf, 4),
                },
                "fuse": {
                    "applied": final_ads != model_ads,
                    "reason": fuse_reason,
                }
            })

    return out


@condition_router.post("/predict-pathological")
async def predict_pathological(
        referral_id: str,
        left_eye_file: UploadFile = File(...),
        right_eye_file: UploadFile = File(...)

):
    """
    Calls the pathology microservice for left/right eye images,
    logs to DB, writes structured results, and applies bilateral override.
    """

    import httpx, logging
    PATHOLOGY_URL = "http://localhost:8006/predict-pair"  # ✅ correct path
    condition = "pathological"
    ad_id = 6
    model_id = 7
    keras_file_name = "other.keras"
    left_id, right_id = [130, 131]  # or: dbf.get_left_and_right_eye_image_ids(referral_id)
    try:
        # --------------------------------------------------
        # Step 1 — Log API call
        # --------------------------------------------------

        dbf.insert_endpoint_logs(
            referral_id,
            f"{base_url}/predict-{condition}?referral_id={referral_id}",
            condition, left_id, right_id, ad_id

        )

        # --------------------------------------------------
        # Step 2 — Call the new pair microservice
        # --------------------------------------------------

        async with httpx.AsyncClient(timeout=200.0) as client:
            files = {
                "left_eye_file": (left_eye_file.filename, await left_eye_file.read(), left_eye_file.content_type),
                "right_eye_file": (right_eye_file.filename, await right_eye_file.read(), right_eye_file.content_type),

            }
            resp = await client.post(PATHOLOGY_URL, files=files, params={"referral_id": referral_id})

            if resp.status_code == 404:
                raise HTTPException(status_code=502, detail="Pathology pair endpoint not found")
            resp.raise_for_status()
            result = resp.json()

        # --------------------------------------------------
        # Step 3 — Extract left/right + pair info
        # --------------------------------------------------
        left_section = result.get("left", {})
        right_section = result.get("right", {})
        left_dec = left_section.get("decision", {})
        right_dec = right_section.get("decision", {})
        # Merge the per-class probabilities
        left_conf = left_section.get("probabilities", {})
        right_conf = right_section.get("probabilities", {})
        left_dec["confidence_scores"] = left_conf
        right_dec["confidence_scores"] = right_conf

        # ✅ Fix: Handle Uncertain / null ads_id cases

        for dec in [left_dec, right_dec]:
            label = dec.get("pred_name") or dec.get("label")
            ads_val = dec.get("ads_id")

            if ads_val is None or ads_val == "null" or label == "Uncertain":
                dec["ads_id"] = 18
                dec["pred_name"] = "No Suspect or Negative"
                dec["label"] = "No Suspect or Negative"
                dec["reason"] = (dec.get("reason", "") + " → downgraded to No Suspect").strip()
                dec["gated"] = True

        pair_sum = result.get("pair_summary", {})

        # --------------------------------------------------
        # Step 4 — Write both eyes to DB
        # --------------------------------------------------

        write_eye_result_to_db('left', left_id, {condition: left_dec}, condition, referral_id, model_id)
        write_eye_result_to_db('right', right_id, {condition: right_dec}, condition, referral_id, model_id)

        # --------------------------------------------------
        # Step 5 — Return structured JSON
        # --------------------------------------------------

        return {
            "referral_id": referral_id,
            "diagnosis": condition,
            "left_eye_result": {condition: left_dec, "keras_file_name": keras_file_name},
            "right_eye_result": {condition: right_dec, "keras_file_name": keras_file_name},
            "pair_summary": pair_sum,
            "source": "PATHOLOGY_MICROSERVICE"
        }

    except httpx.HTTPError as e:
        logging.error(f"Pathology microservice call failed: {e}")
        raise HTTPException(status_code=502, detail=f"Pathology service error: {e}")
    except Exception as e:
        logging.error(f"Unhandled exception in predict_pathological: {e}")
        dbf.log_error_to_db(str(e), "predict_pathological", "predict-pathological")
        raise HTTPException(status_code=500, detail=f"Error in pathology prediction: {str(e)}")


@condition_router.post("/predict-myopia")
async def predict_myopia(
        referral_id: str,
        left_eye_file: UploadFile = File(...),
        right_eye_file: UploadFile = File(...)
):
    condition = "myopia"
    ad_id = 9
    model_id = "8"
    left_id, right_id = [130, 131]  # or: dbf.get_left_and_right_eye_image_ids(referral_id)

    # start log
    dbf.insert_endpoint_logs(
        referral_id,
        f"{base_url}/predict-{condition}?referral_id={referral_id}",
        condition, left_id, right_id, 9
    )

    # --- call microservice per eye ---
    left_result = await handle_myopia(referral_id, left_eye_file)
    right_result = await handle_myopia(referral_id, right_eye_file)

    # --- bilateral promotion (CONSERVATIVE) ---
    try:
        bilat = bilateral_promote(left_result, right_result)  # returns {"should_promote": bool, "reason": str, ...}
    except Exception as e:
        bilat = {"should_promote": False, "reason": f"bilateral_error: {e}"}

    # attach audit trail to both eyes
    for eye in (left_result, right_result):
        eye.setdefault("myopia", {}).setdefault("bilateral_rule", {})
        eye["myopia"]["bilateral_rule"].update(bilat)
        eye["myopia"]["bilateral_rule"]["mode"] = BILAT_MODE  # off | shadow | on

    # apply only if ON
    if bilat.get("should_promote") and BILAT_MODE == "on":
        _apply_promotion_to_eye(
            left_result, bilat.get("final_label", "Myopia"), bilat.get("final_ads_id", 42),
            bilat.get("reason", "bilateral"), BILAT_MODE
        )
        _apply_promotion_to_eye(
            right_result, bilat.get("final_label", "Myopia"), bilat.get("final_ads_id", 42),
            bilat.get("reason", "bilateral"), BILAT_MODE
        )
    else:
        # even in off/shadow, stamp decision mode for traceability without changing the label
        _apply_promotion_to_eye(
            left_result, left_result["myopia"]["prediction"], left_result["myopia"]["ads_id"],
            bilat.get("reason", "no_rule_met"), BILAT_MODE
        )
        _apply_promotion_to_eye(
            right_result, right_result["myopia"]["prediction"], right_result["myopia"]["ads_id"],
            bilat.get("reason", "no_rule_met"), BILAT_MODE
        )

    # --- write to DB AFTER final labels/ads_ids are set ---
    try:
        write_eye_result_to_db('left', left_id, left_result, condition, referral_id, model_id)
        write_eye_result_to_db('right', right_id, right_result, condition, referral_id, model_id)
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"DB write error: {e}")

    # finish log
    dbf.insert_endpoint_logs(
        referral_id,
        f"{base_url}/predict-{condition}?referral_id={referral_id}",
        condition, left_id, right_id, 9
    )

    return {
        "referral_id": referral_id,
        "diagnosis": "myopia",
        "left_eye_result": left_result,
        "right_eye_result": right_result
    }


# Mount the new endpoints to the app
app.include_router(condition_router)


# Utility function to be reused across condition endpoints
async def predict_generic_condition(file: UploadFile, condition: str, eye: str, ar_id: str):
    try:
        logging.info(f"Received {condition} {eye} file for {ar_id}")
        left_id, right_id = dbf.get_left_and_right_eye_image_ids(ar_id)

        dbf.insert_endpoint_logs(ar_id, f"{base_url}/predict-{condition}?referral_id={ar_id}", condition,
                                 left_id,
                                 right_id, 5)
        # Read file bytes
        image_bytes = await file.read()
        temp_path = save_to_tempfile(image_bytes)

        image_input = preprocess_image(temp_path)
        tabular_input = np.array([extract_features_from_image(temp_path)])
        os.remove(temp_path)

        if image_input is None or tabular_input is None:
            raise HTTPException(status_code=400, detail="Image preprocessing failed.")

        keras_file_name = os.path.basename(MODEL_PATHS[condition]["path"])

        if condition == "glaucoma":
            predictions = models[condition].predict([image_input])
        else:
            predictions = np.array(models[condition].predict(image_input))
            #predictions = models[condition].predict(image_input)

        if condition in ["glaucoma", "cnv", "rvo", "myopia"]:
            threshold = 0.5
            confidence_score = float(predictions.flatten()[0])
            predicted_index = 1 if confidence_score > threshold else 0
            confidence_scores = {
                CLASS_NAMES[condition][0]["label"]: round(1 - confidence_score, 6),
                CLASS_NAMES[condition][1]["label"]: round(confidence_score, 6)
            }
            predicted_class_data = CLASS_NAMES[condition][predicted_index]
        elif condition == "pathological":
            # Special handling for pathological condition
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            pathology_classes = [
                "No Suspect or Negative",
                "Minor Pathological",
                "Obvious Pathological"
            ]

            # Create confidence scores in the specific format
            confidence_scores = {
                pathology_classes[i]: round(float(predictions[0][i]), 6)
                for i in range(len(pathology_classes))
            }

            # Get the predicted class data
            predicted_class_data = CLASS_NAMES[condition][predicted_class_idx]
        elif condition == "hypertension":
            # Simple, defensive implementation to avoid index errors
            logging.info(f"Hypertension raw prediction values: {predictions}")

            # Define fixed class labels for hypertension
            hypertension_classes = [
                {"ads_id": 7, "label": "No Suspect or Negative"},
                "Very Mild Hypertension",
                {"ads_id": 8, "label": "Decreased Retinal Artery Elasticity"},
                {"ads_id": 9, "label": "Hypertensive Retinopathy Grade 1 or 2"},
                {"ads_id": 10, "label": "Hypertensive Retinopathy Grade 3 or 4"}
            ]

            # Get prediction index safely
            raw_predictions = predictions[0]  # This is a 1D array of 5 values
            predicted_class_idx = int(np.argmax(raw_predictions))
            logging.info(f"Predicted class index: {predicted_class_idx}")

            # Safety check
            if predicted_class_idx >= len(hypertension_classes):
                logging.error(f"Predicted index {predicted_class_idx} exceeds available classes!")
                predicted_class_idx = len(hypertension_classes) - 1  # Cap at max valid index

            # Create confidence scores safely
            confidence_scores = {}
            for i in range(len(raw_predictions)):
                if i < len(hypertension_classes):
                    class_item = hypertension_classes[i]

                    # Check if it's a dictionary or a string
                    if isinstance(class_item, dict):
                        # Use the label from the dictionary as the key
                        class_name = class_item["label"]
                    else:
                        # Use the string itself as the key
                        class_name = class_item
                else:
                    class_name = f"Class_{i}"

                confidence_scores[class_name] = round(float(raw_predictions[i]), 6)
            ads_id = hypertension_classes[predicted_class_idx]["ads_id"]
            # Manually create class data object
            predicted_class_data = {
                "label": hypertension_classes[predicted_class_idx],
                "ads_id": f"{ads_id}"
            }

            # Skip using CLASS_NAMES entirely for hypertension
            logging.info(f"Final hypertension prediction: {predicted_class_data}")

            # Later in the function, make sure you're not trying to use CLASS_NAMES for hypertension
            # If there's any code after this that tries to access CLASS_NAMES[condition],
            # make sure to skip that for hypertension condition
            predicted_class_data = hypertension_classes[predicted_class_idx]
            predicted_class = predicted_class_data["label"]
            ads_id = predicted_class_data["ads_id"]

            keras_file_name = os.path.basename(MODEL_PATHS[condition]["path"])
            return {
                condition: {
                    "prediction": predicted_class,
                    "confidence_scores": confidence_scores,
                    "ads_id": ads_id
                },
                "keras_file_name": keras_file_name
            }

        else:
            predictions = models[condition].predict(image_input)
            logging.info(f"{condition} prediction shape: {predictions.shape}")
            if predictions is None or predictions.size == 0:
                raise ValueError(f"No predictions returned from {condition} model.")

            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence_scores = {
                CLASS_NAMES[condition][i]["label"]: round(float(predictions[0][i]), 6)
                for i in range(len(CLASS_NAMES[condition]))
            }
            predicted_class_data = CLASS_NAMES[condition][predicted_class_idx]
            print(f"predicted_class_data: {predicted_class_data}")

        predicted_class = predicted_class_data["label"]
        ads_id = predicted_class_data["ads_id"]
        model_no = MODEL_PATHS[condition]["number"]

        return format_prediction_response(condition, predicted_class, confidence_scores, ads_id=ads_id) | {
            "keras_file_name": keras_file_name}

    except Exception as e:
        logging.error(f"Error in {condition} endpoint: {e}")
        dbf.log_error_to_db(f"Error in {condition} endpoint: {e}", "predict_generic_condition", f"predict-{condition}")
        raise HTTPException(status_code=500, detail=f"Error in {condition} endpoint: {str(e)}")


import os, io, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile, File, HTTPException

# Optional knob: how many worker threads to use
PREDICT_ALL_WORKERS = 8

# Map friendly keys -> your model keys in `predict_fns`
CONDITIONS = {
    "diabetes": "predict-diabetes",
    "hypertension": "predict-hypertension",
    "macular_degeneration": "predict-macular-degeneration",
    "cnv": "predict-cnv",
    "glaucoma": "predict-glaucoma",
    "rvo": "predict-rvo",
    "myopia": "predict-myopia",
    "pathological": "predict-pathological",
}
# Base URLs (env overridable). Match your logs.
DIABETES_URL = os.getenv("DIABETES_URL", "http://localhost:8000/predict-diabetes")
HYPERTENSION_URL = os.getenv("HYPERTENSION_URL", "http://ppnai-prod:8000/predict-hypertension")
AMD_URL = os.getenv("AMD_URL", "http://ppnai-prod:8000/predict-macular-degeneration")
CNV_URL = os.getenv("CNV_URL", "http://ppnai-prod:8000/predict-cnv")
RVO_URL = os.getenv("RVO_URL", "http://ppnai-prod:8000/predict-rvo")
GLAUCOMA_URL = os.getenv("GLAUCOMA_URL", "http://ppnai-prod:8000/predict-glaucoma")
PATHO_URL = os.getenv("PATHO_URL", "http://localhost:8000/predict-pathological")
MYOPIA_URL = os.getenv("MYOPIA_URL", "http://ppnai-prod:8000/predict-myopia")

# Map condition -> full endpoint URL
DOWNSTREAM = {
    "diabetes": DIABETES_URL,  # expects /predict
    "hypertension": HYPERTENSION_URL,  # expects /predict-hypertension
    "macular_degeneration": AMD_URL,  # expects /predict-macular-degeneration
    "cnv": CNV_URL,  # expects /predict-cnv
    "rvo": RVO_URL,  # expects /predict-rvo
    "glaucoma": GLAUCOMA_URL,  # expects /predict-glaucoma
    "pathological": PATHO_URL,  # expects /predict-pathological
    "myopia": MYOPIA_URL,  # expects /predict-myopia
}

# Optional bearer token for downstreams
AUTH_TOKEN = os.getenv("AI_BEARER_TOKEN", "").strip()
DEFAULT_TIMEOUT = float(os.getenv("PREDICT_ALL_TIMEOUT", "120"))
PREDICT_ALL_WORKERS = int(os.getenv("PREDICT_ALL_WORKERS", "8"))


def _post_two_files(name: str, url: str, referral_id: str,
                    left_bytes: bytes, right_bytes: bytes,
                    left_name: str, right_name: str,
                    left_ct: str, right_ct: str):
    """Send both eyes to one downstream endpoint and return its JSON or error."""
    try:
        params = {"referral_id": referral_id}
        headers = {}
        if AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

        # Fresh file objects for each request (requests may close them)
        files = {
            "left_eye_file": (left_name, io.BytesIO(left_bytes), left_ct or "image/jpeg"),
            "right_eye_file": (right_name, io.BytesIO(right_bytes), right_ct or "image/jpeg"),
        }

        resp = requests.post(url, params=params, files=files, headers=headers or None, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                data = {"raw_text": resp.text}
            return name, True, data
        else:
            return name, False, {"status": resp.status_code, "text": resp.text}
    except Exception as e:
        return name, False, {"error": str(e)}


@app.post("/predict-all")
async def predict_all(
        referral_id: str,
        left_eye_file: UploadFile = File(...),
        right_eye_file: UploadFile = File(...),
):
    """
    Orchestrates ALL 8 diagnoses. Sends both eyes to each microservice in parallel.
    Returns a combined JSON payload. Each microservice still writes to DB (as in your logs).
    """
    # Read both uploads once
    left_bytes = await left_eye_file.read()
    right_bytes = await right_eye_file.read()
    left_name = left_eye_file.filename or "left.jpg"
    right_name = right_eye_file.filename or "right.jpg"
    left_ct = left_eye_file.content_type or "image/jpeg"
    right_ct = right_eye_file.content_type or "image/jpeg"

    # Kick all eight requests in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=PREDICT_ALL_WORKERS) as ex:
        futures = [
            ex.submit(_post_two_files, name, url, referral_id,
                      left_bytes, right_bytes, left_name, right_name, left_ct, right_ct)
            for name, url in DOWNSTREAM.items()
        ]
        for fut in as_completed(futures):
            name, ok, payload = fut.result()
            results[name] = {"ok": ok, "data": payload}

    # Optionally check for hard failures and return 207 Multi-Status style
    any_fail = any(not v["ok"] for v in results.values())
    if any_fail:
        # Still return aggregated body so you can see which failed
        return {"referral_id": referral_id, "n_conditions": len(DOWNSTREAM), "results": results}

    return {"referral_id": referral_id, "n_conditions": len(DOWNSTREAM), "results": results}


@app.get("/")
async def root():
    return {"message": "Fundus Image Diagnosis API is running!"}
