# Ophthalmology CRM AI API — Retinal Fundus Image Analysis Platform

A FastAPI-based medical AI platform for automated diagnosis of retinal and eye conditions from fundus photographs. The system runs multiple deep learning models in a microservice architecture, persists results to a SQL Server database, and exposes a secured REST API.

---

## Table of Contents

- [Overview](#overview)
- [Supported Conditions](#supported-conditions)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
  - [Authentication](#authentication)
  - [Diagnosis Endpoints](#diagnosis-endpoints)
  - [Handler Endpoints](#handler-endpoints)
- [Microservices](#microservices)
- [Image Preprocessing](#image-preprocessing)
- [Bilateral Promotion (Myopia)](#bilateral-promotion-myopia)
- [Database](#database)
- [Error Handling & Logging](#error-handling--logging)
- [Project Structure](#project-structure)

---

## Overview

The Ophthalmology CRM AI API receives fundus (back-of-eye) photographs and runs them through condition-specific deep learning models to produce diagnostic classifications and confidence scores. Results are written to a SQL Server database via stored procedures.

Key capabilities:

- Single-endpoint multi-condition analysis (`/predict-all`)
- Per-condition endpoints for targeted diagnosis
- Left-eye / right-eye tracking
- Bilateral coordination logic for myopia
- JWT-secured endpoints
- Async inter-service communication between the main API and condition microservices

---

## Supported Conditions

| Condition | Service File | ADS IDs | Output Classes |
|-----------|-------------|---------|----------------|
| Diabetic Retinopathy | `diabetes_service.py` | 2–6 | No DR, Mild, Moderate, Severe, Proliferative DR |
| Hypertension | `hypertension_service*.py` | 7–10 | No Suspect, DRAE, Grade 1–2, Grade 3–4 |
| Age-related Macular Degeneration (AMD) | `amd_service.py` | 11–15 | Drusen, Early, Intermediate, Advanced, Other |
| Glaucoma | `glaucoma_service.py` | 16–17 | Normal, Suspect / Positive |
| Macular Pathology | `pathology_service.py` | 18–21 | No Suspect, Minor, Obvious, Other Macular |
| CNV (Choroidal Neovascularization) | `cnv_service.py` | 22–23 | No Suspect, Suspect / Positive |
| RVO (Retinal Vein Occlusion) | `rvo_service.py` | 24–25 | No Suspect, Suspect / Positive |
| Myopia | `myopia_service.py` | 26–42 | Normal, Myopia (multiple severity grades) |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Client / Frontend                 │
└──────────────────────────┬──────────────────────────┘
                           │ HTTPS  (JWT Bearer)
┌──────────────────────────▼──────────────────────────┐
│               Main API  (app.py — port 8000)        │
│  FastAPI  •  OAuth2/JWT auth  •  image validation   │
│                                                     │
│  /predict-*   →  calls service or microservice      │
│  /handle-*    →  internal microservice handlers     │
└──┬──────────┬────────────┬──────────┬───────────────┘
   │          │            │          │
   ▼          ▼            ▼          ▼
Myopia     RVO/CNV    Pathology   Diabetes / HTN /
service    services    service    AMD / Glaucoma
(8009)   (8007–8008)  (8006/8013) (in-process)
   │          │            │          │
   └──────────┴────────────┴──────────┘
                           │
              ┌────────────▼────────────┐
              │   SQL Server Database   │
              │  (Ophthalmology schema) │
              └─────────────────────────┘
```

---

## Prerequisites

- Python 3.8+
- SQL Server (Windows Auth or username/password)
- ODBC Driver 17 for SQL Server
- Trained model weight files (paths set via environment variables or `config.py`)
- PyTorch ≥ 1.9 and TensorFlow / Keras ≥ 2.x

Install Python dependencies:

```bash
pip install fastapi uvicorn python-jose[cryptography] httpx pyodbc \
            torch torchvision timm tensorflow opencv-python pillow numpy
```

---

## Configuration

All runtime configuration is supplied through environment variables. Default values are defined in `OPHTHALMOLOGY_CRM_AI_API/config.py`.

### Authentication

| Variable | Description |
|----------|-------------|
| `AI_USERNAME` | API login username |
| `AI_PASSWORD` | API login password |

### Database

| Variable | Description | Example |
|----------|-------------|---------|
| `LIVE_CONNECTION_STRING` | pyodbc connection string | `Driver={ODBC Driver 17 for SQL Server};Server=...;Database=Ophthalmology;Trusted_Connection=yes;` |

### Model Paths

| Variable | Condition |
|----------|-----------|
| `DR_WEIGHTS` | Diabetic Retinopathy model weights |
| `AMD_MODEL_PATH` | AMD model weights |
| `AMD_CLASS_TO_IDX` | AMD class-index mapping JSON |
| `AMD_CLASS_META` | AMD class metadata JSON |
| `MYOPIA_MODEL_PATH` | Myopia model weights |

### Myopia Bilateral Promotion

| Variable | Default | Description |
|----------|---------|-------------|
| `MYOPIA_BILATERAL_MODE` | `off` | `on` / `off` / `shadow` |
| `MYOPIA_BILATERAL_EITHER_MIN_PROB` | `0.95` | Min probability for single-eye promotion |
| `MYOPIA_BILATERAL_EITHER_MIN_COMP` | `0.25` | Min complementary score |
| `MYOPIA_BILATERAL_BOTH_MIN_PROB` | `0.80` | Min probability when both eyes analysed |
| `MYOPIA_BILATERAL_BOTH_MIN_COMP` | `0.35` | Min complementary score (both eyes) |
| `MYOPIA_BILATERAL_EXTREME_WEAK_COMP` | `0.10` | Threshold below which promotion is forbidden |
| `MYOPIA_BILATERAL_FORBID_IF_BOTH_BLURRY` | `1` | Block promotion when both images are blurry |

### Myopia Feature Flags

| Variable | Description |
|----------|-------------|
| `MYOPIA_CLAHE` | Enable CLAHE preprocessing (`1`/`0`) |
| `MYOPIA_FOVCROP` | Enable FOV cropping (`1`/`0`) |
| `MYOPIA_TTA` | Enable test-time augmentation (`1`/`0`) |
| `MYOPIA_ENSEMBLE` | Enable model ensembling (`1`/`0`) |
| `MYOPIA_IMG_SIZE` | Input image size (default `224`) |

### Microservice URLs

| Variable | Default | Description |
|----------|---------|-------------|
| `PATHOLOGY_API_URL` | `http://localhost:8013` | Pathology microservice base URL |

---

## Running the Application

### Main API

```bash
cd OPHTHALMOLOGY_CRM_AI_API
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Individual Microservices

Each condition microservice is started independently on its own port:

```bash
# Myopia service
uvicorn myopia_service:app --host 0.0.0.0 --port 8009

# RVO service
uvicorn rvo_service:app --host 0.0.0.0 --port 8008

# CNV service
uvicorn cnv_service:app --host 0.0.0.0 --port 8007

# Pathology service
uvicorn pathology_service:app --host 0.0.0.0 --port 8006
```

---

## API Reference

All endpoints (except `/` and `/token`) require a valid JWT Bearer token in the `Authorization` header.

### Authentication

#### `POST /token`

Obtain a JWT access token.

**Request** (form data):

| Field | Type | Description |
|-------|------|-------------|
| `username` | string | API username (`AI_USERNAME`) |
| `password` | string | API password (`AI_PASSWORD`) |

**Response:**

```json
{
  "access_token": "<jwt>",
  "token_type": "bearer"
}
```

Tokens expire after **2 hours**.

---

### Diagnosis Endpoints

All diagnosis endpoints accept a JSON body with image references and return classification results with confidence scores. Results are also written to the database.

#### `GET /`

Health check — returns `{ "message": "API is running" }`.

---

#### `POST /predict-diabetes`

Diagnose **Diabetic Retinopathy** for both eyes of a referral.

**Request body:**

```json
{
  "ar_id": 12345
}
```

**Response:**

```json
{
  "left_eye":  { "label": "No DR",   "confidence": { "No DR": 0.92, "Mild": 0.05, ... } },
  "right_eye": { "label": "Moderate","confidence": { "No DR": 0.10, "Mild": 0.15, "Moderate": 0.60, ... } }
}
```

---

#### `POST /predict-hypertension`

Diagnose **Hypertension** grades for both eyes.

Same request/response pattern as `/predict-diabetes`. Labels: `No Suspect`, `DRAE`, `Grade 1-2`, `Grade 3-4`.

---

#### `POST /predict-macular-degeneration`

Diagnose **Age-related Macular Degeneration** for both eyes.

Labels: `Drusen`, `Early AMD`, `Intermediate AMD`, `Advanced AMD`, `Other`.

---

#### `POST /predict-cnv`

Diagnose **Choroidal Neovascularization** for both eyes.

Labels: `No Suspect`, `Suspect / Positive`.

---

#### `POST /predict-rvo`

Diagnose **Retinal Vein Occlusion** for both eyes.

Labels: `No Suspect`, `Suspect / Positive`.

---

#### `POST /predict-pathological`

Diagnose **Macular Pathology** for both eyes.

Labels: `No Suspect`, `Minor`, `Obvious`, `Other Macular`.

---

#### `POST /predict-myopia`

Diagnose **Myopia** for both eyes, with optional bilateral promotion logic applied.

Labels vary by severity grade. See [Bilateral Promotion](#bilateral-promotion-myopia).

---

#### `POST /predict-all`

Run **all conditions** on both eyes in a single request. Returns a combined result object keyed by condition name.

---

### Handler Endpoints

Internal routing endpoints used by the main API to forward requests to individual condition microservices. These follow the same auth requirements.

| Endpoint | Condition |
|----------|-----------|
| `POST /handle-diabetes` | Diabetic Retinopathy |
| `POST /handle-hypertension` | Hypertension |
| `POST /handle-amd` | AMD |
| `POST /handle-glaucoma` | Glaucoma |
| `POST /handle-cnv` | CNV |
| `POST /handle-rvo` | RVO |
| `POST /handle-pathological` | Macular Pathology |
| `POST /handle-myopia` | Myopia |

---

## Microservices

| Service | Default Port | File |
|---------|-------------|------|
| Myopia | 8009 | `services/myopia_service.py` |
| RVO | 8008 | `services/rvo_service.py` |
| CNV | 8007 | `services/cnv_service.py` |
| Pathology | 8006 / 8013 | `services/pathology_service.py` |

All other conditions (Diabetes, Hypertension, AMD, Glaucoma) are handled **in-process** within the main API.

---

## Image Preprocessing

Images are preprocessed before inference using a pipeline defined in `model_service.py`:

1. **Circular masking** — isolates the fundus region, discards background.
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — enhances local contrast.
3. **Sharpening / blur reduction** — improves edge definition.
4. **Resize** — scales to model input size (224×224 or 256×256 depending on the model).
5. **Normalization** — per-channel mean/std normalisation for PyTorch models; `[0, 1]` scaling for Keras models.

Quality control checks (blur detection, dark/bright mean thresholds) can block promotion decisions for the Myopia service.

---

## Bilateral Promotion (Myopia)

The myopia service can **promote** a borderline result to a positive diagnosis by considering both eyes together. This is controlled by `MYOPIA_BILATERAL_MODE`:

- `off` — bilateral logic is disabled; each eye is classified independently.
- `shadow` — bilateral logic runs and is logged, but the classification is **not** changed.
- `on` — bilateral logic is active and overrides individual eye results.

**Promotion rules:**

| Rule | Condition |
|------|-----------|
| Rule A (single eye) | Either eye probability ≥ `EITHER_MIN_PROB` AND complementary score ≥ `EITHER_MIN_COMP` |
| Rule B (both eyes) | Both eye probabilities ≥ `BOTH_MIN_PROB` AND complementary score ≥ `BOTH_MIN_COMP` |
| Forbidden | Complementary score < `EXTREME_WEAK_COMP`, OR both images are blurry (when `FORBID_IF_BOTH_BLURRY=1`) |

---

## Database

The API uses **SQL Server** (Ophthalmology database) via `pyodbc`.

### Key Tables

| Table | Description |
|-------|-------------|
| `AI_IMAGES` | Fundus images; columns include `AI_ID`, `AI_AR_ID` (referral), `AI_CON_OCULUS_ID` (130=left, 131=right), `AI_IMAGE_NAME`, `AI_IMAGE_LOCATION` |
| `AI_DIAGNOSIS_SEVERITIES` | Lookup table mapping `ADS_ID` to severity labels per condition |
| `RETINAL_PATHOLOGY_CLASSIFICATIONS` | Pathology classification results |

### Key Stored Procedures

| Procedure | Purpose |
|-----------|---------|
| `STP_INSERT_DIAGNOSIS_RESULT` | Write a diagnosis result |
| `STP_INSERT_AI_ERROR_LOG` | Log application errors |
| `STP_INSERT_AI_ENDPOINT_LOG` | Log API request metadata |
| `STP_GET_AI_IMAGE_PATH` | Retrieve image file path and G-folder |
| `STP_GET_MODELS` | Retrieve currently active models |

Database operations are encapsulated in `db_functions/db_functions.py` (`DbFunctions` class).

---

## Error Handling & Logging

- All unhandled exceptions are caught, logged to file (rotating log handler), and also written to the database via `STP_INSERT_AI_ERROR_LOG`.
- Every API request is logged via `STP_INSERT_AI_ENDPOINT_LOG` (endpoint name, referral ID, timing).
- Microservice errors are propagated back to the caller as HTTP exceptions with descriptive messages.
- Stack traces are captured and stored for debugging.

---

## Project Structure

```
diabetes-api/
└── OPHTHALMOLOGY_CRM_AI_API/
    ├── app.py                          # Main FastAPI application & all routes
    ├── config.py                       # Model paths, class names, DB config
    ├── htn_calibrator.py               # Hypertension probability calibration
    ├── db_functions/
    │   ├── __init__.py
    │   └── db_functions.py             # SQL Server operations (DbFunctions class)
    └── services/
        ├── model_service.py            # Shared ML utilities & custom loss functions
        ├── diabetes_service.py         # Diabetic Retinopathy model & inference
        ├── hypertension_service.py     # Hypertension model & inference
        ├── hypertension_service_v3_11.py
        ├── hypertension_service_v4.py
        ├── glaucoma_service.py         # Glaucoma model & inference
        ├── myopia_service.py           # Myopia model, QC, bilateral logic
        ├── amd_service.py              # AMD model & inference
        ├── cnv_service.py              # CNV model & inference
        ├── rvo_service.py              # RVO model & inference
        ├── pathology_service.py        # Macular pathology model & inference
        └── cdr_service.py              # Cup-to-Disc Ratio calculation
```
