# config.py
# Model Paths
# MODEL_PATHS = {
#     "diabetes": r'C:\source_controls\development_sanele\components\trained_models\resnet50V2_diabetes_grades_finetuned.keras',
#     "hypertension": r'C:\source_controls\development_sanele\components\trained_models\hypertension_grading_model.keras',
#     "macular_degeneration": r'C:\source_controls\development_sanele\components\trained_models\macular_degeneration_model.keras',
#     "cnv": r'C:\source_controls\development_sanele\components\trained_models\cnv_model.keras',
#     "rvo": r'C:\AI_Images\trained_models\rvo_model.keras',
#     #"glaucoma": r'C:\AI_Images\Glaucoma_dataset\models\glaucoma_model.keras',
#     # "glaucoma":r'C:\AI_Images\trained_models\glaucoma_augmented_model.keras',
#      "glaucoma":r'C:\AI_Images\trained_models\glaucoma_augmented_model_v1.keras',
#     #  "glaucoma" :r"C:\AI_Images\trained_models\refuge_glaucoma_classifier_finetuned.keras",
#      #"glaucoma":r'C:\source_controls\development_sanele\components\trained_models\ResNet50V2_glaucoma_finetuned.keras',
#     #"glaucoma": r'C:\AI_Images\REFUGE-glaucoma\train\glaucoma_model_refuge_finetuned.keras',
#     # "glaucoma_with_features": r'C:\AI_Images\Glaucoma_dataset\glaucoma_model.keras',
#     #"glaucoma_with_features" : r'C:\source_controls\development_sanele\glaucoma_model_v3.keras',
#     #"glaucoma_with_features": r'C:\AI_Images\trained_models\glaucoma_multi_input_model.keras',
#     #"glaucoma": r'C:\source_controls\development_tashen\glaucoma_detection_model.keras',
#      #"glaucoma_with_features": r"C:\source_controls\development_tashen\model_with_numeric_features.keras",  # Model 1
#     # "glaucoma_image_only": r"C:\source_controls\development_tashen\model_image_only.keras",        # Model 2
#     "pathological": r'C:\AI_Images\trained_models\other.keras',
#     "segmentation": r'C:\AI_Images\trained_models\glaucoma_segmentation_unet.keras',
#     "myopia": r'C:\source_controls\development_sanele\myopia_detection_model.keras'
# }
import os
from dotenv import load_dotenv
load_dotenv('components/utils/.env')
MODEL_PATHS = {
    "diabetes": {
        "number": 2,
        "path": r'C:\source_controls\development_sanele\components\trained_models\resnet50V2_diabetes_grades_finetuned.keras'
    },
    "hypertension": {
        "number": 3,
        "path": r'C:\source_controls\development_sanele\components\trained_models\hypertension_grading_model.keras'
    },
    "macular_degeneration": {
        "number": 4,
        "path": r'C:\source_controls\development_sanele\components\trained_models\macular_degeneration_model.keras'
    },
    "cnv": {
        "number": 5,
        "path": r'C:\source_controls\development_sanele\components\trained_models\cnv_model.keras'
    },
    "rvo": {
        "number": 6,
        "path": r'C:\AI_Images\trained_models\rvo_model.keras'
    },
    "glaucoma": {
        "number": 1,
        "path": r'C:\AI_Images\trained_models\glaucoma_augmented_model_v1.keras'
    },
    "pathological": {
        "number": 7,
        "path": r'C:\AI_Images\trained_models\other.keras'
    },
    # "segmentation": {
    #     "number": 9,
    #     "path": r'C:\AI_Images\trained_models\glaucoma_segmentation_unet.keras'
    # },
    "myopia": {
        "number": 8,
        "path": r'C:\source_controls\development_sanele\myopia_detection_model.keras'
    }
}

CLASS_NAMES = {
    "diabetes": [
        {"label": "No Suspect or Negative", "ads_id": 2},
        {"label": "Mild", "ads_id": 3},
        {"label": "Moderate", "ads_id": 4},
        {"label": "Severe", "ads_id": 5},
        {"label": "Proliferative Diabetic Retinopathy (PDR)", "ads_id": 6}
    ],
    "hypertension": [
        {"label": "No Suspect or Negative", "ads_id": 7},
        {"label": "Decreased Retinal Artery Elasticity", "ads_id": 8},
        {"label": "Hypertensive Retinopathy Grade 1 or 2", "ads_id": 9},
        {"label": "Hypertensive Retinopathy Grade 3 or 4", "ads_id": 10}
    ],
    "macular_degeneration": [
        {"label": "Drusen", "ads_id": 12},
        {"label": "AMD Early or Intermediate Stage", "ads_id": 13},
        {"label": "AMD Advanced Stage", "ads_id": 14},
        {"label": "Large Drusen with Abnormal Pigmentation", "ads_id": 15},
        {"label": "No Suspect or Negative", "ads_id": 11},
        {"label": "Unknown Class", "ads_id": -1},
    ],
    "cnv": [
        {"label": "No Suspect or Negative", "ads_id": 22},
        {"label": "Suspect or Positive", "ads_id": 23}
    ],
    "rvo": [
        {"label": "No Suspect or Negative", "ads_id": 24},
        {"label": "Suspect or Positive", "ads_id": 25}
    ],
    "glaucoma": [
        {"label": "No Suspect or Negative", "ads_id": 16},
        {"label": "Suspect or Positive", "ads_id": 17}
    ],
    "pathological": [
        {"label": "No Suspect or Negative", "ads_id": 18},
        {"label": "Minor Pathological", "ads_id": 19},
        {"label": "Obvious Pathological", "ads_id": 20},
       # {"label": "Other Macular Pathology", "id": 21}
    ],
    "myopia": [
         {"label": "No Suspect or Negative", "ads_id": 26},
         {"label": "Myopia", "ads_id": 27},
        # {"label": "No Suspect or Negative", "id": 26},
        # {"label": "Mild", "id": 27},
        # {"label": "Moderate", "id": 28},
        # {"label": "Mild", "id": 29},
        # {"label": "Mild", "id": 30}
    ]
}

# Define Classes
# CLASS_NAMES = {
#     "diabetes": ['No Suspect or Negative', 'Mild', 'Moderate', 'Severe', 'Proliferative Diabetic Retinopathy (PDR)'],
#     "hypertension": ['No Suspect or Negative',  'Suspect or Positive',
#         'Decreased Retinal Artery Elasticity',
#         'Hypertensive Retinopathy Grade 1 or 2',#         'Hypertensive Retinopathy Grade 3 or 4'
#     ],
#     "macular_degeneration": [
#         "Drusen",
#         "AMD Early or Intermediate Stage",
#         "AMD Advanced Stage",
#         "Large Drusen with Abnormal Pigmentation"
#     ],
#     "cnv": ['No Suspect or Negative', 'Suspect or Positive'],
#     "rvo": ['No Suspect or Negative', 'Suspect or Positive'],
#     "glaucoma": ['No Suspect or Negative', 'Suspect or Positive'],
#     "pathological": ['No Suspect or Negative', 'Minor Pathological', 'Obvious Pathological'],
#     "myopia": ['No Suspect or Negative', 'Suspect or Positive']
# }

# Dictionary mapping (case-insensitive)
condition_to_ad_id = {
    "diabetes": 2,
    "diabetic": 2,
    "hypertension": 3,
    "hypertensive": 3,
    "macular degeneration": 4,
    "macular_degeneration": 4,
    "glaucoma": 5,
    "pathological": 6,
    "cnv": 7,
    "choroidal neovascularization": 7,
    "retinal vein occlusion": 8,
    "rvo": 8,
    "myopia": 9
}

# Connection Strings
# CONNECTION_STRING = (
#     "DRIVER={SQL Server};"
#     "SERVER=sql-dev-ppn;"
#     "DATABASE=Ophthalmology;"
#     "Trusted_Connection=yes;"
# )

# Database Connection String
# connection_string = (
#     "DRIVER={SQL Server};"
#     "SERVER=sql-dev-ppn;"
#     "DATABASE=Ophthalmology;"
#     "Trusted_Connection=yes;"
# )
connection_string = os.getenv("LIVE_CONNECTION_STRING")
if not connection_string:
    raise EnvironmentError("Connection string not found in environment variables.")



# Secret Key for JWT
SECRET_KEY = "46e2011cb14891bb9c57ed2b4c688250c57ad3ad354359e87ee0d715d581112e"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 2
