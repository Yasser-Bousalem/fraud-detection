import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from src.explainability.reason_codes import ReasonCodeExplainer

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"

# feature columns the model expects, in order.
# saved once during training so the API knows the exact schema.
FEATURE_COLS_PATH = "models/feature_cols.json"


class FraudScoringService:
    """
    Loads the champion model, calibrator, threshold, and SHAP explainer once,
    then scores individual transactions with reason codes.

    Instantiate ONCE at app startup — never per request.
    """

    def __init__(
        self,
        model_path: str = "models/lgbm_champion.txt",
        calibrator_path: str = "models/isotonic_calibrator.pkl",
        threshold_path: str = "models/operating_threshold.json",
        explainer_path: str = "models/shap_explainer.pkl",
        feature_cols_path: str = FEATURE_COLS_PATH,
    ):
        logger.info("Loading fraud scoring service …")

        self.model = lgb.Booster(model_file=model_path)
        self.calibrator = joblib.load(calibrator_path)
        self.threshold = json.loads(Path(threshold_path).read_text())[
            "chosen_threshold"
        ]
        self.feature_cols = json.loads(Path(feature_cols_path).read_text())
        self.categorical_cols = json.loads(
            Path("models/categorical_cols.json").read_text()
        )

        # reason code explainer (wraps model + SHAP)
        self.explainer = ReasonCodeExplainer(model_path, explainer_path)

        logger.info(
            "Service ready. Model v%s, threshold=%.4f, %d features",
            MODEL_VERSION,
            self.threshold,
            len(self.feature_cols),
        )

    def _build_feature_row(self, tx: dict) -> pd.DataFrame:
        """
        Build a single-row DataFrame with ALL expected feature columns in order.
        Missing → NaN. Categorical cols → category dtype. Everything else → numeric.
        """
        row = {col: tx.get(col, np.nan) for col in self.feature_cols}
        df = pd.DataFrame([row], columns=self.feature_cols)

        cat_set = set(self.categorical_cols)

        for col in df.columns:
            if col in cat_set:
                df[col] = df[col].astype("category")
            else:
                # force numeric — coerces stray objects/None to proper float NaN
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def score(self, tx: dict) -> dict:
        """
        Score a single transaction.
        Returns dict with fraud_score, decision, threshold, reason_codes, version.
        """
        import time

        X = self._build_feature_row(tx)

        t0 = time.perf_counter()
        # raw model score → calibrated probability
        raw = float(self.model.predict(X)[0])
        proba = float(self.calibrator.transform([raw])[0])
        t1 = time.perf_counter()

        decision = "review" if proba >= self.threshold else "approve"

        # reason codes (interpretable features preferred)
        if decision == "review":
            reason_codes = self.explainer.explain_one(X, top_k=3)["reason_codes"]
        else:
            reason_codes = []
        t2 = time.perf_counter()
        logger.info("predict=%.1fms  shap=%.1fms", (t1 - t0) * 1000, (t2 - t1) * 1000)
        return {
            "fraud_score": round(proba, 4),
            "decision": decision,
            "threshold": round(self.threshold, 4),
            "reason_codes": reason_codes,
            "model_version": MODEL_VERSION,
        }
