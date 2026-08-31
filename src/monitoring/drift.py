import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from src.config import TARGET_COL, TIME_COL
from src.data.load import load_raw
from src.features.velocity import add_velocity_features
from src.features.pipeline import get_feature_cols

logger = logging.getLogger(__name__)


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index between two distributions.
    PSI < 0.1  : no significant shift
    0.1-0.25   : moderate shift
    > 0.25     : major shift (retrain signal)
    """
    ref = reference.dropna()
    cur = current.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    # bin edges from reference quantiles
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    # convert to proportions, add epsilon to avoid div/0 and log(0)
    ref_prop = ref_counts / ref_counts.sum() + 1e-6
    cur_prop = cur_counts / cur_counts.sum() + 1e-6

    psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
    return float(psi)


def prepare_reference_and_current():
    """
    Reference = training window (first 60% of timeline).
    Current   = test window (last 20%) — simulates production data.
    """
    df = load_raw()
    df = add_velocity_features(df)

    t20 = df[TIME_COL].quantile(0.60)
    t80 = df[TIME_COL].quantile(0.80)

    reference = df[df[TIME_COL] <= t20].copy()      # training period
    current   = df[df[TIME_COL] >  t80].copy()      # production (test) period

    logger.info("Reference: %d rows (training period)", len(reference))
    logger.info("Current:   %d rows (production period)", len(current))

    return reference, current, df


def score_both(reference, current):
    """Score both windows so we can measure prediction drift."""
    df_all = load_raw().pipe(add_velocity_features)
    t60 = df_all[TIME_COL].quantile(0.60)
    train_df = df_all[df_all[TIME_COL] <= df_all[TIME_COL].quantile(0.60)]
    train_end = train_df[TIME_COL].quantile(0.75)
    train_sub = train_df[train_df[TIME_COL] <= train_end]

    cat_cols, num_cols = get_feature_cols(train_sub)
    feature_cols = num_cols + cat_cols

    model      = lgb.Booster(model_file="models/lgbm_champion.txt")
    calibrator = joblib.load("models/isotonic_calibrator.pkl")

    def _score(df):
        X = df[feature_cols].copy()
        for c in cat_cols:
            X[c] = pd.Categorical(X[c], categories=train_sub[c].astype("category").cat.categories)
        raw = model.predict(X)
        return calibrator.transform(raw)

    ref_scores = _score(reference)
    cur_scores = _score(current)
    return ref_scores, cur_scores, feature_cols


def run_psi_report(reference, current, feature_cols, top_n=30):
    """Compute PSI for the most important features."""
    # focus on interpretable + top model features
    key_features = [
        "TransactionAmt", "card1_tx_count_7d", "card1_tx_count_24h",
        "device_distinct_cards_24h", "email_distinct_cards_24h",
        "C1", "C13", "C14", "D1", "D2", "D15",
        "addr1", "dist1", "TransactionDT",
    ]
    key_features = [f for f in key_features if f in reference.columns and f != TIME_COL]

    rows = []
    for feat in key_features:
        if pd.api.types.is_numeric_dtype(reference[feat]):
            psi = compute_psi(reference[feat], current[feat])
            rows.append({"feature": feat, "psi": psi})

    psi_df = pd.DataFrame(rows).sort_values("psi", ascending=False)

    def _severity(psi):
        if psi > 0.25:  return "MAJOR"
        if psi > 0.10:  return "moderate"
        return "stable"

    psi_df["severity"] = psi_df["psi"].apply(_severity)
    return psi_df


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Preparing reference (training) and current (production) windows …")
    reference, current, df = prepare_reference_and_current()

    # ── feature drift (PSI) ──
    logger.info("Computing feature drift (PSI) …")
    _, _, feature_cols = score_both(reference, current)
    psi_df = run_psi_report(reference, current, feature_cols)

    print("\n" + "=" * 55)
    print("  FEATURE DRIFT REPORT (PSI: reference → current)")
    print("=" * 55)
    print(psi_df.to_string(index=False))
    print("-" * 55)
    print(f"  MAJOR drift    (PSI>0.25): {(psi_df['severity']=='MAJOR').sum()}")
    print(f"  Moderate drift (PSI>0.10): {(psi_df['severity']=='moderate').sum()}")
    print(f"  Stable                   : {(psi_df['severity']=='stable').sum()}")
    print("=" * 55)

    # ── prediction drift ──
    logger.info("Computing prediction drift …")
    ref_scores, cur_scores, _ = score_both(reference, current)
    pred_psi = compute_psi(pd.Series(ref_scores), pd.Series(cur_scores))

    print(f"\n  PREDICTION DRIFT (score distribution PSI): {pred_psi:.4f}")
    if pred_psi > 0.25:
        print("  → MAJOR prediction drift — model output distribution shifted significantly")
    elif pred_psi > 0.10:
        print("  → Moderate prediction drift")
    else:
        print("  → Prediction distribution stable")

    # ── target drift (fraud rate) ──
    ref_fraud = reference[TARGET_COL].mean()
    cur_fraud = current[TARGET_COL].mean()
    print(f"\n  TARGET DRIFT (fraud rate):")
    print(f"    Reference (training): {ref_fraud*100:.2f}%")
    print(f"    Current (production): {cur_fraud*100:.2f}%")
    print(f"    Relative change:      {(cur_fraud-ref_fraud)/ref_fraud*100:+.1f}%")

    # ── save ──
    Path("reports").mkdir(exist_ok=True)
    psi_df.to_csv("reports/drift_psi.csv", index=False)

    drift_summary = {
        "prediction_psi":   float(pred_psi),
        "reference_fraud_rate": float(ref_fraud),
        "current_fraud_rate":   float(cur_fraud),
        "features_major_drift":    int((psi_df["severity"]=="MAJOR").sum()),
        "features_moderate_drift": int((psi_df["severity"]=="moderate").sum()),
        "top_drifting_features": psi_df.head(5)[["feature","psi"]].to_dict("records"),
    }
    with open("reports/drift_summary.json", "w") as f:
        json.dump(drift_summary, f, indent=2)
    logger.info("Saved → reports/drift_summary.json, reports/drift_psi.csv")


if __name__ == "__main__":
    main()