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
        model_path:      str = "models/lgbm_champion.txt",
        calibrator_path: str = "models/isotonic_calibrator.pkl",
        threshold_path:  str = "models/operating_threshold.json",
        explainer_path:  str = "models/shap_explainer.pkl",
        feature_cols_path: str = FEATURE_COLS_PATH,
    ):
        logger.info("Loading fraud scoring service …")

        self.model      = lgb.Booster(model_file=model_path)
        self.calibrator = joblib.load(calibrator_path)
        self.threshold  = json.loads(Path(threshold_path).read_text())["chosen_threshold"]
        self.feature_cols = json.loads(Path(feature_cols_path).read_text())
        self.categorical_cols = json.loads(
            Path("models/categorical_cols.json").read_text()
        )
        
        # reason code explainer (wraps model + SHAP)
        self.explainer = ReasonCodeExplainer(model_path, explainer_path)

        logger.info("Service ready. Model v%s, threshold=%.4f, %d features",
                    MODEL_VERSION, self.threshold, len(self.feature_cols))

    def _build_feature_row(self, tx: dict) -> pd.DataFrame:
        """
        Turn an incoming transaction dict into a single-row DataFrame with
        ALL expected feature columns in the right order. Missing → NaN.
        Categorical columns are cast to 'category' dtype so LightGBM matches
        the training schema.
        """
        row = {col: tx.get(col, np.nan) for col in self.feature_cols}
        df = pd.DataFrame([row], columns=self.feature_cols)

        # cast the KNOWN categorical columns (from training) to category dtype
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df

    def score(self, tx: dict) -> dict:
        """
        Score a single transaction.
        Returns dict with fraud_score, decision, threshold, reason_codes, version.
        """
        X = self._build_feature_row(tx)

        # raw model score → calibrated probability
        raw   = float(self.model.predict(X)[0])
        proba = float(self.calibrator.transform([raw])[0])

        decision = "review" if proba >= self.threshold else "approve"

        # reason codes (interpretable features preferred)
        explanation = self.explainer.explain_one(X, top_k=3)

        return {
            "fraud_score":   round(proba, 4),
            "decision":      decision,
            "threshold":     round(self.threshold, 4),
            "reason_codes":  explanation["reason_codes"],
            "model_version": MODEL_VERSION,
        }