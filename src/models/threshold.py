import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import mlflow
import lightgbm as lgb
import joblib

from src.config import TARGET_COL, TIME_COL, RANDOM_SEED
from src.data.load import load_raw
from src.features.velocity import add_velocity_features
from src.features.pipeline import get_feature_cols

logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ── Business assumptions ─────────────────────────────────────────────
REVIEW_COST_USD         = 5.0     # $ per alert (analyst time)
ANALYST_CAPACITY_DAILY  = 200     # max alerts the fraud team can review per day
# avg_fraud_amount and daily_volume are computed from data


def compute_business_context(df: pd.DataFrame) -> dict:
    """Derive dataset-specific business parameters."""
    n_days = (df[TIME_COL].max() - df[TIME_COL].min()) / 86400
    daily_volume = len(df) / n_days

    fraud_amounts = df.loc[df[TARGET_COL] == 1, "TransactionAmt"]
    avg_fraud_amt = fraud_amounts.mean()

    return {
        "n_days":         n_days,
        "daily_volume":   daily_volume,
        "avg_fraud_amt":  avg_fraud_amt,
        "review_cost":    REVIEW_COST_USD,
        "daily_capacity": ANALYST_CAPACITY_DAILY,
    }


def sweep_thresholds(
    y_true: pd.Series,
    y_score: np.ndarray,
    context: dict,
    n_thresholds: int = 200,
) -> pd.DataFrame:
    """
    For each threshold candidate, compute confusion counts and daily-scaled economics.
    """
    thresholds = np.linspace(0.001, 0.999, n_thresholds)
    
    # scale factor: convert per-window metrics → per-day
    n_days = context["n_days"]
    scale  = 1.0 / n_days

    rows = []
    for t in thresholds:
        flagged = y_score >= t
        tp = int(((flagged) & (y_true == 1)).sum())
        fp = int(((flagged) & (y_true == 0)).sum())
        fn = int(((~flagged) & (y_true == 1)).sum())
        tn = int(((~flagged) & (y_true == 0)).sum())

        alerts     = tp + fp
        alerts_day = alerts * scale
        
        # daily economics
        savings_day  = tp * context["avg_fraud_amt"] * scale
        cost_day     = alerts * context["review_cost"] * scale
        net_day      = savings_day - cost_day

        precision = tp / alerts if alerts > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        rows.append({
            "threshold":    t,
            "tp":           tp,
            "fp":           fp,
            "fn":           fn,
            "tn":           tn,
            "alerts_day":   alerts_day,
            "precision":    precision,
            "recall":       recall,
            "savings_day":  savings_day,
            "cost_day":     cost_day,
            "net_day":      net_day,
            "within_capacity": alerts_day <= context["daily_capacity"],
        })

    return pd.DataFrame(rows)


def plot_threshold_analysis(sweep: pd.DataFrame, context: dict, chosen_t: float, output_path: str):
    """Four-panel diagnostic — the money story visualized."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # ── panel 1: expected daily savings ──
    ax = axes[0, 0]
    ax.plot(sweep["threshold"], sweep["net_day"], color="steelblue", linewidth=2)
    ax.axvline(chosen_t, color="green", linestyle="--", label=f"Chosen: {chosen_t:.3f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Net daily savings ($)")
    ax.set_title("Expected daily net savings vs threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── panel 2: precision & recall ──
    ax = axes[0, 1]
    ax.plot(sweep["threshold"], sweep["precision"], color="tomato",   linewidth=2, label="Precision")
    ax.plot(sweep["threshold"], sweep["recall"],    color="steelblue", linewidth=2, label="Recall")
    ax.axvline(chosen_t, color="green", linestyle="--", label=f"Chosen: {chosen_t:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Precision & recall vs threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── panel 3: alerts per day + capacity ──
    ax = axes[1, 0]
    ax.plot(sweep["threshold"], sweep["alerts_day"], color="purple", linewidth=2)
    ax.axhline(context["daily_capacity"], color="red", linestyle="--",
               linewidth=1, label=f"Capacity: {context['daily_capacity']}/day")
    ax.axvline(chosen_t, color="green", linestyle="--")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Alerts per day")
    ax.set_yscale("log")
    ax.set_title("Alert volume vs threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── panel 4: TP / FP breakdown ──
    ax = axes[1, 1]
    ax.plot(sweep["threshold"], sweep["tp"], color="green", linewidth=2, label="True Positives (fraud caught)")
    ax.plot(sweep["threshold"], sweep["fp"], color="tomato", linewidth=2, label="False Positives (false alarms)")
    ax.plot(sweep["threshold"], sweep["fn"], color="orange", linewidth=2, label="False Negatives (fraud missed)")
    ax.axvline(chosen_t, color="green", linestyle="--")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count (val window)")
    ax.set_yscale("log")
    ax.set_title("Outcome counts vs threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"Threshold analysis  |  avg fraud = ${context['avg_fraud_amt']:.0f}, "
        f"review cost = ${context['review_cost']:.0f}, "
        f"capacity = {context['daily_capacity']}/day",
        y=1.00
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    logger.info("Loading + engineering …")
    df = load_raw()
    df = add_velocity_features(df)

    cutoff = df[TIME_COL].quantile(0.80)
    df_cv = df[df[TIME_COL] <= cutoff].copy()

    # split CV portion 75/25 → same train/val as Day 11
    df_sorted = df_cv.sort_values(TIME_COL).reset_index(drop=True)
    t_min, t_max = df_sorted[TIME_COL].min(), df_sorted[TIME_COL].max()
    train_end = t_min + (t_max - t_min) * 0.75
    val_df = df_sorted[df_sorted[TIME_COL] > train_end].copy()

    context = compute_business_context(val_df)

    logger.info("Business context:")
    for k, v in context.items():
        logger.info("  %-15s : %s", k, f"{v:.2f}" if isinstance(v, float) else v)

    # ── load champion model + calibrator ──
    logger.info("Loading champion model + calibrator …")
    model      = lgb.Booster(model_file="models/lgbm_champion.txt")
    calibrator = joblib.load("models/isotonic_calibrator.pkl")

    # prepare val features (must match training preprocessing)
    train_df = df_sorted[df_sorted[TIME_COL] <= train_end]
    cat_cols, num_cols = get_feature_cols(train_df)
    feature_cols = num_cols + cat_cols

    X_val = val_df[feature_cols].copy()
    for c in cat_cols:
        X_val[c] = pd.Categorical(X_val[c], categories=train_df[c].astype("category").cat.categories)

    y_val = val_df[TARGET_COL]

    # ── score & calibrate ──
    y_score_raw = model.predict(X_val)
    y_score_cal = calibrator.transform(y_score_raw)

    # ── sweep thresholds ──
    logger.info("Sweeping thresholds …")
    sweep = sweep_thresholds(y_val, y_score_cal, context)

    # ── choose threshold: maximize net_day subject to capacity constraint ──
    valid = sweep[sweep["within_capacity"]]
    if len(valid) == 0:
        raise RuntimeError("No threshold satisfies daily capacity — increase capacity or improve model.")

    best_row = valid.loc[valid["net_day"].idxmax()]
    chosen_t = best_row["threshold"]

    logger.info("\n" + "=" * 60)
    logger.info("  OPTIMAL THRESHOLD ANALYSIS")
    logger.info("=" * 60)
    logger.info("  Chosen threshold      : %.4f", chosen_t)
    logger.info("  Alerts / day          : %.1f  (capacity: %d)",
                best_row["alerts_day"], context["daily_capacity"])
    logger.info("  Precision             : %.4f", best_row["precision"])
    logger.info("  Recall                : %.4f", best_row["recall"])
    logger.info("  TP (fraud caught)     : %d", best_row["tp"])
    logger.info("  FP (false alarms)     : %d", best_row["fp"])
    logger.info("  FN (fraud missed)     : %d", best_row["fn"])
    logger.info("  ")
    logger.info("  Daily savings ($)     : %.2f", best_row["savings_day"])
    logger.info("  Daily review cost ($) : %.2f", best_row["cost_day"])
    logger.info("  NET DAILY SAVINGS ($) : %.2f", best_row["net_day"])
    logger.info("  Annual (× 365)        : %.0f", best_row["net_day"] * 365)
    logger.info("=" * 60)

    # ── save artifacts ──
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plot_path = "reports/figures/threshold_analysis.png"
    plot_threshold_analysis(sweep, context, chosen_t, plot_path)
    logger.info("Saved threshold plot → %s", plot_path)

    sweep.to_csv("reports/threshold_sweep.csv", index=False)

    threshold_config = {
        "chosen_threshold":   float(chosen_t),
        "expected_daily_net": float(best_row["net_day"]),
        "expected_annual":    float(best_row["net_day"] * 365),
        "context":            {k: float(v) if isinstance(v, (int, float)) else v
                               for k, v in context.items()},
    }
    with open("models/operating_threshold.json", "w") as f:
        json.dump(threshold_config, f, indent=2)
    logger.info("Saved threshold config → models/operating_threshold.json")

    # ── MLflow logging ──
    mlflow.set_experiment("champion_model")
    with mlflow.start_run(run_name="threshold_optimization"):
        mlflow.log_param("review_cost_usd",      REVIEW_COST_USD)
        mlflow.log_param("daily_capacity",       ANALYST_CAPACITY_DAILY)
        mlflow.log_param("avg_fraud_amt",        context["avg_fraud_amt"])

        mlflow.log_metric("chosen_threshold",    chosen_t)
        mlflow.log_metric("alerts_per_day",      best_row["alerts_day"])
        mlflow.log_metric("precision",           best_row["precision"])
        mlflow.log_metric("recall",              best_row["recall"])
        mlflow.log_metric("daily_net_savings",   best_row["net_day"])
        mlflow.log_metric("annual_savings",      best_row["net_day"] * 365)

        mlflow.log_artifact(plot_path)
        mlflow.log_artifact("reports/threshold_sweep.csv")
        mlflow.log_artifact("models/operating_threshold.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()