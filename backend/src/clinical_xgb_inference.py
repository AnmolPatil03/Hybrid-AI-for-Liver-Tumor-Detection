"""
Inference for the clinical XGBoost HCC risk model.

Used by POST /clinical/predict and GET /clinical/tree/{i} in app/routes.py.
A predicted probability above 0.5 is reported as elevated risk and the app
recommends a CT scan.
"""

import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.clinical_xgb_data import (
    FEATURE_COLUMNS, NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES,
    load_preprocessor,
)

MODEL_PATH = os.path.join("outputs", "models", "clinical_xgb_model.json")
PREPROCESSOR_PATH = os.path.join("outputs", "models", "clinical_xgb_preprocessor.joblib")
META_PATH = os.path.join("outputs", "models", "clinical_xgb_meta.json")

RISK_THRESHOLD = 0.5


class ClinicalXGBInference:
    """Loads the clinical model and preprocessor and serves predictions."""

    def __init__(self):
        self.model: XGBClassifier | None = None
        self.preprocessor = None
        self.meta = None
        self.is_loaded = False

    def load_model(self, model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH,
                    meta_path=META_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Clinical XGBoost model not found at '{model_path}'. "
                f"Train it first: python train_clinical_xgb.py"
            )

        self.model = XGBClassifier()
        self.model.load_model(model_path)

        self.preprocessor = load_preprocessor(preprocessor_path)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        self.is_loaded = True
        print(f"[INFO] Clinical XGBoost model loaded from: {model_path}")

    def _build_feature_frame(self, clinical: dict):
        """
        Build a one-row DataFrame from the input dict. Missing values stay NaN
        and are filled by the preprocessor's imputers.

        Returns (raw_frame, missing_feature_names).
        """
        row = {}
        missing = []
        for col in FEATURE_COLUMNS:
            v = clinical.get(col)
            if v is None:
                missing.append(col)
            row[col] = v
        raw_frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        return raw_frame, missing

    def predict(self, clinical: dict):
        """
        Run the clinical model on a (possibly partial) feature dict.

        Args:
            clinical: dict keyed by FEATURE_COLUMNS names, e.g.
                {"AFP_ng_mL": 620, "AFP_L3_percent": 18.5, "DCP_mAU_mL": 340,
                 "Albumin_g_dL": 3.1, "Bilirubin_mg_dL": 2.4,
                 "Platelet_count_x10e9_L": 98, "Age": 54, "Sex": "M",
                 "HCV_status": 0, "HBV_status": 1, "Cirrhosis_status": 1}

        Returns:
            dict with risk_probability, risk_level ("elevated"/"normal"),
            recommend_ct (bool), and any missing_features that were imputed.
        """
        if not self.is_loaded:
            raise RuntimeError("Clinical XGBoost model not loaded. Call load_model() first.")

        raw_frame, missing = self._build_feature_frame(clinical)
        X = self.preprocessor.transform(raw_frame[FEATURE_COLUMNS])
        X = pd.DataFrame(X, columns=self.preprocessor.get_feature_names_out())

        proba = float(self.model.predict_proba(X)[0, 1])
        recommend_ct = proba > RISK_THRESHOLD

        return {
            "risk_probability": round(proba, 4),
            "risk_level": "elevated" if recommend_ct else "normal",
            "recommend_ct": recommend_ct,
            "missing_features": missing,
            "interpretation": (
                "Clinical risk model indicates an elevated risk of HCC — a CT scan "
                "is recommended for imaging analysis."
                if recommend_ct else
                "Clinical risk model indicates no significant risk of HCC based on "
                "the clinical markers."
            ),
        }

    def n_estimators(self):
        if not self.is_loaded:
            raise RuntimeError("Clinical XGBoost model not loaded.")
        # n_estimators is not restored by load_model(), so count the trees in the booster.
        return len(self.model.get_booster().get_dump())

    def get_tree_json(self, tree_index):
        """Return one boosted tree as a nested dict (from the booster's JSON dump)."""
        if not self.is_loaded:
            raise RuntimeError("Clinical XGBoost model not loaded.")

        n = self.n_estimators()
        if not (0 <= tree_index < n):
            raise ValueError(f"tree_index must be in [0, {n - 1}], got {tree_index}")

        booster = self.model.get_booster()
        dumps = booster.get_dump(dump_format="json")
        tree_dict = json.loads(dumps[tree_index])
        return {"tree_index": tree_index, "n_estimators": n, "tree": tree_dict}


clinical_xgb_engine = ClinicalXGBInference()
