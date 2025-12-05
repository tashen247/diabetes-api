# htn_calibrator.py

import json

import numpy as np

from pathlib import Path

from typing import Dict, List

# Point this to your actual JSON path from the training script

CALIB_PATH = Path(r"C:\AI_Images\Hypertension\htn_calibrator.json")


class HypertensionCalibrator:

    def __init__(self, path: Path = CALIB_PATH):

        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        with path.open("r", encoding="utf-8") as f:

            d = json.load(f)

        self.classes: List[str] = d["classes"]

        self.coef = np.array(d["coef"], dtype=float)  # shape (4,4)

        self.intercept = np.array(d["intercept"], float)  # shape (4,)

        if self.coef.shape != (4, 4):
            raise ValueError(f"Expected coef 4x4, got {self.coef.shape}")

        if self.intercept.shape != (4,):
            raise ValueError(f"Expected intercept (4,), got {self.intercept.shape}")

    def predict_proba(self, prob_map: Dict[str, float]) -> Dict[str, float]:

        """

        Input:

          prob_map = {

             "NoSuspectOrNegative": p0,

             "DecreasedRetinalArteryElasticity": p1,

             "HypertensiveRetinopathyGrade1or2": p2,

             "HypertensiveRetinopathyGrade3or4": p3

          }

        Output: calibrated probs with same keys.

        """

        x = np.array([

            prob_map.get("NoSuspectOrNegative", 0.0),

            prob_map.get("DecreasedRetinalArteryElasticity", 0.0),

            prob_map.get("HypertensiveRetinopathyGrade1or2", 0.0),

            prob_map.get("HypertensiveRetinopathyGrade3or4", 0.0),

        ], dtype=float)

        logits = self.coef @ x + self.intercept  # (4,)

        e = np.exp(logits - logits.max())

        p = e / e.sum()

        return {self.classes[i]: float(p[i]) for i in range(4)}
