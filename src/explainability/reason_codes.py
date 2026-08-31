import logging
import json
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore", message=".*TreeExplainer shap values output has changed.*"
)

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import joblib
import re

logger = logging.getLogger(__name__)


# ── Feature → human template mapping ────────────────────────────────
# Each entry: feature_name -> function(value) -> human string
# For anonymized features we fall back to a generic template.

FEATURE_TEMPLATES = {
    "TransactionAmt": lambda v: f"Transaction amount is ${v:,.0f}",
    "card1_tx_count_7d": lambda v: (
        f"No prior transactions on this card in 7 days"
        if pd.isna(v) or v == 0
        else f"{int(v)} prior transactions on this card in 7 days"
    ),
    "card1_tx_count_24h": lambda v: (
        f"No prior transactions on this card in 24h"
        if pd.isna(v) or v == 0
        else f"{int(v)} prior transactions on this card in 24h"
    ),
    "card1_seconds_since_last": lambda v: (
        f"First recorded transaction on this card"
        if pd.isna(v)
        else f"Last transaction on this card was {v/3600:.1f}h ago"
    ),
    "device_distinct_cards_24h": lambda v: (
        f"Device has used {int(v)} distinct cards in 24h"
        if not pd.isna(v) and v > 1
        else "Single card on this device"
    ),
    "device_tx_count_1h": lambda v: (
        f"{int(v)} transactions on this device in the last hour"
        if not pd.isna(v) and v > 0
        else "No recent device activity"
    ),
    "email_distinct_cards_24h": lambda v: (
        f"Email used with {int(v)} distinct cards in 24h"
        if not pd.isna(v) and v > 1
        else "Single card for this email"
    ),
    "card6": lambda v: f"Card type: {v}",
    "card4": lambda v: f"Card network: {v}",
    "P_emaildomain": lambda v: f"Email domain: {v}",
    "ProductCD": lambda v: f"Product category: {v}",
    "DeviceType": lambda v: f"Device type: {v}",
    "addr1": lambda v: f"Billing region code: {v}",
    "has_identity": lambda v: (
        "Device/browser data captured" if v == 1 else "No device/browser data captured"
    ),
}

# Anonymized feature families get a generic message
GENERIC_PREFIXES = {
    "C": "Internal count signal",
    "D": "Days-since-event signal",
    "V": "Internal risk signal",
    "M": "Identity match signal",
    "id_": "Identity attribute",
}


def humanize_feature(name: str, value) -> str:
    """Translate a feature name + value into a human-readable phrase."""
    if name in FEATURE_TEMPLATES:
        try:
            return FEATURE_TEMPLATES[name](value)
        except Exception:
            return f"{name} = {value}"

    # anonymized feature families — match ONLY <Letter><digits> like C1, D2, V91, M6
    m = re.match(r"^(id_|[CDVM])\d+$", name)
    if m:
        prefix = m.group(1)
        labels = {
            "C": "Internal count signal",
            "D": "Days-since-event signal",
            "V": "Internal risk signal",
            "M": "Identity match signal",
            "id_": "Identity attribute",
        }
        return f"{labels[prefix]} ({name}) is atypical"

    # add DeviceInfo explicitly
    if name == "DeviceInfo":
        return f"Device: {value}"

    return f"{name} = {value}"


class ReasonCodeExplainer:
    """
    Wraps a LightGBM model + SHAP explainer to produce per-transaction
    scores with human-readable reason codes.
    """

    def __init__(self, model_path: str, explainer_path: str):
        self.model = lgb.Booster(model_file=model_path)
        self.explainer = joblib.load(explainer_path)
        self.base_value = self.explainer.expected_value

    def explain_one(self, x_row, top_k=3, prefer_interpretable=True):
        contribs = self.model.predict(x_row, pred_contrib=True)[0]
        shap_vals = contribs[:-1]   # last element is base value
        raw_score = float(self.model.predict(x_row)[0])

        feature_names = x_row.columns.tolist()
        contributions = list(zip(feature_names, shap_vals, x_row.iloc[0].values))
        contributions.sort(key=lambda t: t[1], reverse=True)

        positive = [c for c in contributions if c[1] > 0]

        if prefer_interpretable:
            interpretable = [c for c in positive if c[0] in FEATURE_TEMPLATES]
            anonymized    = [c for c in positive if c[0] not in FEATURE_TEMPLATES]
            ordered = interpretable + anonymized
        else:
            ordered = positive

        reason_codes = []
        for name, shap_val, value in ordered[:top_k]:
            reason_codes.append({
                "feature":     name,
                "explanation": humanize_feature(name, value),
                "shap_impact": round(float(shap_val), 4),
            })

        return {"fraud_score": round(raw_score, 4), "reason_codes": reason_codes}

def demo():
    """Load champion model + explainer, explain a few sample transactions."""
    from src.config import TARGET_COL, TIME_COL
    from src.data.load import load_raw
    from src.features.velocity import add_velocity_features
    from src.features.pipeline import get_feature_cols

    logger.info("Loading data …")
    df = load_raw()
    df = add_velocity_features(df)

    cutoff = df[TIME_COL].quantile(0.80)
    df_test = df[df[TIME_COL] > cutoff].copy()

    df_tv = df[df[TIME_COL] <= cutoff].sort_values(TIME_COL).reset_index(drop=True)
    t_min, t_max = df_tv[TIME_COL].min(), df_tv[TIME_COL].max()
    train_end = t_min + (t_max - t_min) * 0.75
    train_df = df_tv[df_tv[TIME_COL] <= train_end]

    cat_cols, num_cols = get_feature_cols(train_df)
    feature_cols = num_cols + cat_cols

    X_test = df_test[feature_cols].copy()
    for c in cat_cols:
        X_test[c] = pd.Categorical(
            X_test[c], categories=train_df[c].astype("category").cat.categories
        )
    y_test = df_test[TARGET_COL]

    explainer = ReasonCodeExplainer(
        model_path="models/lgbm_champion.txt",
        explainer_path="models/shap_explainer.pkl",
    )

    # find some high-scoring fraud and some false positives to show
    scores = explainer.model.predict(X_test)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # top 3 highest-scoring TRUE frauds
    fraud_idx = y_test[y_test == 1].index
    fraud_scores = pd.Series(scores[fraud_idx], index=fraud_idx).sort_values(
        ascending=False
    )

    print("\n" + "=" * 70)
    print("  EXAMPLE REASON CODES — high-confidence TRUE frauds")
    print("=" * 70)
    for idx in fraud_scores.head(3).index:
        result = explainer.explain_one(X_test.iloc[[idx]])
        print(
            f"\n  Transaction {idx} | score={result['fraud_score']:.3f} | ACTUAL: FRAUD"
        )
        for i, rc in enumerate(result["reason_codes"], 1):
            print(f"    {i}. {rc['explanation']}  (impact: +{rc['shap_impact']})")

    # a couple of false positives (legit but high score)
    legit_idx = y_test[y_test == 0].index
    legit_scores = pd.Series(scores[legit_idx], index=legit_idx).sort_values(
        ascending=False
    )

    print("\n" + "=" * 70)
    print("  EXAMPLE REASON CODES — FALSE POSITIVES (legit, high score)")
    print("=" * 70)
    for idx in legit_scores.head(2).index:
        result = explainer.explain_one(X_test.iloc[[idx]])
        print(
            f"\n  Transaction {idx} | score={result['fraud_score']:.3f} | ACTUAL: LEGIT"
        )
        for i, rc in enumerate(result["reason_codes"], 1):
            print(f"    {i}. {rc['explanation']}  (impact: +{rc['shap_impact']})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    demo()
