# src/explainability/fairness_audit.py

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import lightgbm as lgb
import joblib
import mlflow

from src.config import TARGET_COL, TIME_COL
from src.data.load import load_raw
from src.features.velocity import add_velocity_features
from src.features.pipeline import get_feature_cols

logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MIN_GROUP_SIZE = 500   # ignore groups too small for stable rates


def prepare_test_with_predictions():
    """Load test set, score it, apply operating threshold."""
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
        X_test[c] = pd.Categorical(X_test[c], categories=train_df[c].astype("category").cat.categories)

    model      = lgb.Booster(model_file="models/lgbm_champion.txt")
    calibrator = joblib.load("models/isotonic_calibrator.pkl")
    threshold  = json.loads(Path("models/operating_threshold.json").read_text())["chosen_threshold"]

    raw   = model.predict(X_test)
    proba = calibrator.transform(raw)
    flagged = (proba >= threshold).astype(int)

    audit_df = df_test[[TARGET_COL, "addr1", "P_emaildomain"]].copy().reset_index(drop=True)
    audit_df["flagged"] = flagged
    audit_df["proba"]   = proba
    return audit_df


def audit_group(audit_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute per-group fairness metrics."""
    rows = []
    for group_val, sub in audit_df.groupby(group_col, dropna=False):
        n = len(sub)
        if n < MIN_GROUP_SIZE:
            continue

        n_fraud  = int(sub[TARGET_COL].sum())
        n_legit  = n - n_fraud
        n_flagged = int(sub["flagged"].sum())

        # flagging rate — how often this group gets flagged at all
        flag_rate = n_flagged / n

        # false positive rate — among LEGIT, how often wrongly flagged
        legit = sub[sub[TARGET_COL] == 0]
        fpr = legit["flagged"].mean() if len(legit) > 0 else 0.0

        # true positive rate (recall) — among FRAUD, how often caught
        fraud = sub[sub[TARGET_COL] == 1]
        tpr = fraud["flagged"].mean() if len(fraud) > 0 else np.nan

        rows.append({
            "group":       str(group_val),
            "n":           n,
            "fraud_rate":  n_fraud / n,
            "flag_rate":   flag_rate,
            "fpr":         fpr,
            "tpr":         tpr,
        })

    return pd.DataFrame(rows).sort_values("fpr", ascending=False)


def plot_fairness(group_df: pd.DataFrame, group_name: str, out_path: str):
    """Plot FPR and flag rate across groups."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    top = group_df.head(20)

    ax1.barh(range(len(top))[::-1], top["fpr"], color="tomato", alpha=0.8)
    ax1.axvline(group_df["fpr"].mean(), color="black", linestyle="--",
                linewidth=1, label=f"Mean FPR: {group_df['fpr'].mean():.3f}")
    ax1.set_yticks(range(len(top))[::-1])
    ax1.set_yticklabels(top["group"], fontsize=8)
    ax1.set_xlabel("False Positive Rate (legit wrongly flagged)")
    ax1.set_title(f"FPR by {group_name}")
    ax1.legend()
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax2.barh(range(len(top))[::-1], top["flag_rate"], color="steelblue", alpha=0.8)
    ax2.axvline(group_df["flag_rate"].mean(), color="black", linestyle="--",
                linewidth=1, label=f"Mean flag rate: {group_df['flag_rate'].mean():.3f}")
    ax2.set_yticks(range(len(top))[::-1])
    ax2.set_yticklabels(top["group"], fontsize=8)
    ax2.set_xlabel("Flagging rate")
    ax2.set_title(f"Flag rate by {group_name}")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def disparity_ratio(group_df: pd.DataFrame, metric: str) -> float:
    """Ratio of max to min group metric — a common fairness summary stat."""
    vals = group_df[metric].replace(0, np.nan).dropna()
    if len(vals) < 2:
        return np.nan
    return vals.max() / vals.min()


def main():
    logger.info("Scoring test set …")
    audit_df = prepare_test_with_predictions()

    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for group_col, group_name in [("addr1", "billing region"),
                                   ("P_emaildomain", "email domain")]:
        logger.info("Auditing by %s …", group_name)
        gdf = audit_group(audit_df, group_col)

        fpr_disp  = disparity_ratio(gdf, "fpr")
        flag_disp = disparity_ratio(gdf, "flag_rate")

        print("\n" + "=" * 70)
        print(f"  FAIRNESS AUDIT — {group_name.upper()}")
        print("=" * 70)
        print(f"  Groups analyzed (n >= {MIN_GROUP_SIZE}): {len(gdf)}")
        print(f"  FPR disparity ratio (max/min)  : {fpr_disp:.2f}x")
        print(f"  Flag rate disparity (max/min)  : {flag_disp:.2f}x")
        print(f"\n  Highest-FPR groups:")
        print(gdf.head(5)[["group", "n", "fraud_rate", "fpr", "flag_rate"]].to_string(index=False))
        print(f"\n  Lowest-FPR groups:")
        print(gdf.tail(5)[["group", "n", "fraud_rate", "fpr", "flag_rate"]].to_string(index=False))

        plot_path = out_dir / f"fairness_{group_col}.png"
        plot_fairness(gdf, group_name, str(plot_path))

        gdf.to_csv(f"reports/fairness_{group_col}.csv", index=False)
        results[group_col] = {
            "n_groups":       len(gdf),
            "fpr_disparity":  float(fpr_disp),
            "flag_disparity": float(flag_disp),
        }

    print("\n" + "=" * 70)

    with open("reports/fairness_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    mlflow.set_experiment("champion_model")
    with mlflow.start_run(run_name="fairness_audit"):
        for group_col, r in results.items():
            mlflow.log_metric(f"{group_col}_fpr_disparity",  r["fpr_disparity"])
            mlflow.log_metric(f"{group_col}_flag_disparity", r["flag_disparity"])
            mlflow.log_artifact(f"reports/fairness_{group_col}.csv")
            mlflow.log_artifact(f"reports/figures/fairness_{group_col}.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()