# dashboard/generate_alerts.py
"""
One-time: score a sample of test transactions and save the flagged ones
as the analyst alert queue. Simulates 'today's alerts'.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

from src.config import TARGET_COL, TIME_COL
from src.data.load import load_raw
from src.features.velocity import add_velocity_features
from src.features.pipeline import get_feature_cols


def main():
    print("Loading test data …")
    df = load_raw()
    df = add_velocity_features(df)

    cutoff = df[TIME_COL].quantile(0.80)
    df_test = df[df[TIME_COL] > cutoff].copy()

    # take the most recent ~3000 test transactions as "today"
    df_test = df_test.sort_values(TIME_COL).tail(3000).reset_index(drop=True)

    df_tv = df[df[TIME_COL] <= cutoff].sort_values(TIME_COL).reset_index(drop=True)
    t_min, t_max = df_tv[TIME_COL].min(), df_tv[TIME_COL].max()
    train_end = t_min + (t_max - t_min) * 0.75
    train_df = df_tv[df_tv[TIME_COL] <= train_end]

    cat_cols, num_cols = get_feature_cols(train_df)
    feature_cols = num_cols + cat_cols

    X = df_test[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=train_df[c].astype("category").cat.categories)

    model      = lgb.Booster(model_file="models/lgbm_champion.txt")
    calibrator = joblib.load("models/isotonic_calibrator.pkl")
    threshold  = json.loads(Path("models/operating_threshold.json").read_text())["chosen_threshold"]

    print("Scoring …")
    raw   = model.predict(X)
    proba = calibrator.transform(raw)

    df_test["fraud_score"] = proba
    df_test["flagged"]     = (proba >= threshold).astype(int)

    # keep only flagged ones — the alert queue
    alerts = df_test[df_test["flagged"] == 1].copy()
    alerts = alerts.sort_values("fraud_score", ascending=False).reset_index(drop=True)

    # columns to keep for display + reason codes
    display_cols = [
        "TransactionID", "TransactionAmt", "TransactionDT", "ProductCD",
        "card4", "card6", "P_emaildomain", "DeviceType", "DeviceInfo",
        "addr1", TARGET_COL, "fraud_score",
    ] + feature_cols  # keep all features for SHAP in the detail view

    # dedupe if TransactionID appears in both
    display_cols = list(dict.fromkeys([c for c in display_cols if c in alerts.columns]))
    alerts = alerts[display_cols]

    Path("dashboard/data").mkdir(parents=True, exist_ok=True)
    alerts.to_parquet("dashboard/data/alert_queue.parquet")

    print(f"Saved {len(alerts)} alerts → dashboard/data/alert_queue.parquet")
    print(f"Of these, {int(alerts[TARGET_COL].sum())} are actually fraud "
          f"({alerts[TARGET_COL].mean()*100:.1f}% precision)")


if __name__ == "__main__":
    main()