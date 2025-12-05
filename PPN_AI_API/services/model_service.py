import os

import cv2
import tensorflow as tf
import numpy as np
import logging
from typing import Callable, Dict, Any

from fastapi import HTTPException, UploadFile
#from keras import backend as K
# from keras import Model
# from keras.models import load_model
# from keras.models import Model, load_model
#from keras import backend as K
#import keras
#import tensorflow.compat.v2 as tf
import tensorflow as tf

#from keras import Loss

#from tensorflow.keras.losses import Loss
#from keras.losses import Loss

from config import MODEL_PATHS, CLASS_NAMES
from db_functions.db_functions import DbFunctions

# create an instance of the class
dbf = DbFunctions(use_live=False)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Placeholder function to simulate CDR calculation
def calculate_cdr(image):
    # Placeholder: Replace with real CDR extraction logic if available
    return np.random.uniform(0.4, 0.9)  # Simulated dynamic CDR value

# ✅ Define the custom loss function (adjust if you used another custom function)
# def focal_loss(alpha=0.25, gamma=2.0):
#     def loss(y_true, y_pred):
#         y_true = tf.cast(y_true, dtype=tf.float32)
#         y_pred = tf.cast(y_pred, dtype=tf.float32)

#         BCE = K.binary_crossentropy(y_true, y_pred)
#         BCE_EXP = K.exp(-BCE)
#         focal_loss = alpha * K.pow((1 - BCE_EXP), gamma) * BCE
#         return K.mean(focal_loss)

#     return loss

# Custom loss function
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_exp = tf.math.exp(-bce)
        focal = alpha * tf.math.pow((1 - bce_exp), gamma) * bce
        return tf.reduce_mean(focal)

    return loss

#from tensorflow.keras.utils import custom_object_scope
#from tensorflow.keras import backend as K
#models: Dict[str, tf.keras.Model] = {}
models = {} # type: Dict[str, tf.keras.Model] # If using type hints
# Define custom objects (add all required custom objects here)
# def get_focal_loss(gamma: float = 2.0, alpha: float = 0.25) -> Callable:
#     """Define or import focal loss function."""
#     try:
#         from tensorflow_addons.losses import focal_loss
#         return focal_loss
#     except ImportError:
#         logging.warning("tensorflow-addons not installed, using custom focal_loss")
#         def focal_loss_fixed(y_true: Any, y_pred: Any) -> Any:
#             # Placeholder for custom focal loss implementation
#             # Replace with actual implementation
#             import tensorflow.keras.backend as K
#             ce_loss = K.categorical_crossentropy(y_true, y_pred)
#             return ce_loss  # Simplified example
#         return focal_loss_fixed
    
CUSTOM_OBJECTS = {
    'focal_loss': None,  # Will be populated dynamically
    # Add other custom objects, e.g., 'CustomF1Score': CustomF1Score
}
# Populate custom objects
custom_objects = CUSTOM_OBJECTS.copy()
#custom_objects['focal_loss'] = get_focal_loss()
# try:
#     #with tf.keras.utils.custom_object_scope({ 'loss': focal_loss()}):
#     with tf.keras.utils.custom_object_scope(custom_objects):
#     # with tf.keras.utils.custom_object_scope({
#     #     'loss': focal_loss(),
#     #     'Addons>F1Score': CustomF1Score  # Use your CustomF1Score for loading
#     # }):
#         for condition, meta in MODEL_PATHS.items():
#             print(f"Loading model for {condition} from {meta['path']}")
#             models[condition] = tf.keras.models.load_model(meta["path"])
#     logging.info("Models loaded successfully!")
# except Exception as e:
#     # log_error_to_db() # type: ignore
#     logging.error(f"Error loading models: {e}")
#     # dbf.log_error_to_db("Error loading models", str(e), "model_loading") # if dbf is available
#     raise e

predict_fns = {}
for condition, model in models.items():
    def make_predict_fn(m):
        @tf.function
        def predict_fn(image_tensor):
            return m(image_tensor)
        return predict_fn

    predict_fns[condition] = make_predict_fn(model)

def predict_image(condition: str, image_input: np.ndarray, extra_input: Any = None) -> np.ndarray:
    try:
        model = models.get(condition)
        if model is None:
            raise ValueError(f"Model for condition '{condition}' not loaded.")

        if condition == "glaucoma" and extra_input is not None:
            prediction = model.predict([image_input, extra_input])
        else:
            prediction = model.predict(image_input)

        return prediction

    except Exception as e:
        logging.error(f"Error predicting for {condition}: {e}")
        raise

def segment_cup_and_disc(image):
    """
    Segments the cup and disc from the input image.
    For now, this is a placeholder that generates synthetic masks.

    Args:
        image: Input fundus image.

    Returns:
        cup_mask: Binary mask of the cup region.
        disc_mask: Binary mask of the disc region.
    """
    # # Replace this with actual segmentation model predictions
    # # For now, create dummy circular masks
    # h, w, _ = image.shape
    # center = (w // 2, h // 2)

    # disc_mask = np.zeros((h, w), dtype=np.uint8)
    # cup_mask = np.zeros((h, w), dtype=np.uint8)

    # # Draw dummy disc and cup masks
    # cv2.circle(disc_mask, center, int(min(h, w) * 0.4), 1, -1)  # Large circle (disc)
    # cv2.circle(cup_mask, center, int(min(h, w) * 0.2), 1, -1)  # Smaller circle (cup)

    # return cup_mask, disc_mask
    try:
        # Preprocess image for the segmentation model
        preprocessed_image = preprocess_image(image)  # Ensure image is resized and normalized
        prediction = models["segmentation"].predict(preprocessed_image)  # Segmentation model

        # Assuming the model outputs two channels: [Disc Mask, Cup Mask]
        disc_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)  # Threshold disc mask
        cup_mask = (prediction[0, :, :, 1] > 0.5).astype(np.uint8)  # Threshold cup mask

        return cup_mask, disc_mask
    except Exception as e:
       logging.error(f"Segmentation failed: {e}")
    return None, None

def calculate_eccentricity(mask):
    """
    Calculates the eccentricity of a binary mask using image moments.

    Args:
        mask: Binary mask (NumPy array).

    Returns:
        Eccentricity of the region.
    """
    moments = cv2.moments(mask)
    if moments["mu20"] + moments["mu02"] == 0:
        return 0.0  # Avoid division by zero

    # Eigenvalues of the covariance matrix
    a = moments["mu20"] / moments["m00"]
    b = 2 * moments["mu11"] / moments["m00"]
    c = moments["mu02"] / moments["m00"]

    lambda1 = (a + c + np.sqrt((a - c) ** 2 + b ** 2)) / 2.0
    lambda2 = (a + c - np.sqrt((a - c) ** 2 + b ** 2)) / 2.0

    # Calculate eccentricity
    if lambda1 == 0:
        return 0.0
    eccentricity = np.sqrt(1 - (lambda2 / lambda1))
    return eccentricity


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
            # tabular_input = extract_features_from_image(file.filename)
            if tabular_input is None:
                raise ValueError("Failed to extract features")
            tabular_input_tensor = tf.convert_to_tensor(tabular_input, dtype=tf.float32)

            # if len(image_input) != len(tabular_input_tensor):
            #     raise ValueError("Mismatch between number of images and tabular features")

            # Predict using the glaucoma models
            predictions = models["glaucoma"].predict([image_input, tabular_input_tensor])
            # predictions = models["glaucoma"].predict([image_input])
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


def extract_cdr_and_ecc(image):
    try:
        # Generate segmentation masks
        cup_mask, disc_mask = segment_cup_and_disc(image)

        # Debug: Print mask statistics
        print(f"Cup Mask Sum: {np.sum(cup_mask)}, Disc Mask Sum: {np.sum(disc_mask)}")

        # Check if the masks have any valid regions
        if np.sum(cup_mask) == 0 or np.sum(disc_mask) == 0:
            logging.warning("One or both masks are empty. Returning default values.")
            return 0.0, 0.0, 0.0

        # Calculate properties
        cup_area = np.sum(cup_mask)
        disc_area = np.sum(disc_mask)
        cdr = cup_area / disc_area if disc_area > 0 else 0.0
        print(f"CDR: {cdr}")

        # Calculate eccentricity
        cup_eccentricity = calculate_eccentricity(cup_mask)
        disc_eccentricity = calculate_eccentricity(disc_mask)
        print(f"Eccentricities - Cup: {cup_eccentricity}, Disc: {disc_eccentricity}")

        return round(cdr, 4), round(cup_eccentricity, 4), round(disc_eccentricity, 4)
    except Exception as e:
        logging.error(f"Error extracting CDR and ECC: {e}")
        return 0.0, 0.0, 0.0


def preprocess_image(image, target_size=(224, 224)):
    try:
        if isinstance(image, str):  # If a path is passed
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Image not found: {image}")

        # Resize and preprocess
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        return np.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        return None

# Feature Extraction Logic
def extract_features(image):
    """
    Extract CDR (Cup-to-Disc Ratio) and ECC values using segmentation model.
    """
    try:
        # Predict segmentation masks (assumes the model outputs two masks: disc and cup)
        preprocessed_image = preprocess_image(image)
        predicted_masks = models['segmentation'].predict(preprocessed_image)[0]

        # Threshold the masks
        mask_disc = (predicted_masks[:, :, 0] > 0.5).astype(np.uint8)
        mask_cup = (predicted_masks[:, :, 1] > 0.5).astype(np.uint8)

        # Calculate properties
        labeled_disc = label(mask_disc)
        labeled_cup = label(mask_cup)
        props_disc = regionprops(labeled_disc)[0] if np.any(labeled_disc) else None
        props_cup = regionprops(labeled_cup)[0] if np.any(labeled_cup) else None

        if props_disc and props_cup:
            # Compute CDR (Cup Diameter / Disc Diameter)
            cup_diameter = max(props_cup.major_axis_length, props_cup.minor_axis_length)
            disc_diameter = max(props_disc.major_axis_length, props_disc.minor_axis_length)
            cdr = cup_diameter / disc_diameter

            # Compute ECC (Eccentricity)
            ecc_cup = props_cup.eccentricity
            ecc_disc = props_disc.eccentricity
        else:
            cdr, ecc_cup, ecc_disc = 0.0, 0.0, 0.0  # Default values if masks fail
            # cdr, ecc_cup, ecc_disc = extract_features(image)

        return cdr, ecc_cup, ecc_disc
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return 0.0, 0.0, 0.0

def extract_features_from_image(image_path):

    try:
        image_path = os.path.join(r'C:\AI_Images\Glaucoma1\glaucoma', image_path)
        if not os.path.exists(image_path):
            dbf.log_error_to_db(f"Image does not exist: {image_path}", r"extract_features_from_image({image_path})",
                                "/diagnose")
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            dbf.log_error_to_db(f"Image could not be read: {image_path}", "", "/diagnose")
            raise ValueError(f"Image could not be read: {image_path}")

        resized_image = cv2.resize(image, (224, 224))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        _, _, _, max_loc = cv2.minMaxLoc(gray_image)
        fovea_x, fovea_y = max_loc
        size_y, size_x = image.shape[:2]
        return [fovea_x, fovea_y, size_x, size_y]
    except Exception as e:
        logging.error(f"Error extracting features from image: {e}")
        dbf.log_error_to_db(f"Error extracting features from image: {e}",
                            r"extract_features_from_image({image_path})", "/diagnose")
        return [0, 0, 0, 0]

# Format Prediction Response
def format_prediction_response(condition, prediction, confidence_scores, comparison=None, existing_results=None, ads_id=None):
    response = {
        condition: {
            "prediction": prediction,
            "ads_id": ads_id,
            "confidence_scores": confidence_scores,
            # "comparison": comparison,
            # "existing_results": existing_results,
        }
    }
    #print(f"prediction {prediction} confidence_scores {confidence_scores} comparison {comparison} existing_results {existing_results}")
    return response
