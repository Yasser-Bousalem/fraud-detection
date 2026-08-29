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
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

from src.config import TARGET_COL, TIME_COL, RANDOM_SEED
from src.data.load import load_raw
from src.features.velocity import add_velocity_features
from src.features.pipeline import get_feature_cols
from src.models.evaluate import evaluate, precision_at_k
from src.models.threshold import compute_business_context, REVIEW_COST_USD, ANALYST_CAPACITY_DAILY

logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def bootstrap_ci(y_true: pd.Series, y_score: np.ndarray, metric_fn, n_boot: int = 1000, seed: int = RANDOM_SEED) -> tuple[float, float]:
    """95% bootstrap CI for a metric function."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def plot_final_diagnostics(y_test, y_score_cal, threshold, context, out_dir: Path):
    """4-panel diagnostic for the final report."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── panel 1: PR curve ──
    ax = axes[0, 0]
    prec, rec, _ = precision_recall_curve(y_test, y_score_cal)
    ax.plot(rec, prec, color="steelblue", linewidth=2)
    ax.axhline(y_test.mean(), color="gray", linestyle=":", linewidth=1, label=f"Baseline (fraud rate): {y_test.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test)")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── panel 2: ROC curve ──
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score_cal)
    ax.plot(fpr, tpr, color="steelblue", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve (test)")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── panel 3: reliability diagram ──
    ax = axes[1, 0]
    frac_pos, mean_pred = calibration_curve(y_test, y_score_cal, n_bins=15, strategy="quantile")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", color="steelblue", linewidth=2, markersize=7, label="Calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual)")
    ax.set_title("Reliability diagram (test)")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── panel 4: confusion matrix at operating threshold ──
    ax = axes[1, 1]
    y_pred = (y_score_cal >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Flag"])
    ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix @ threshold={threshold:.3f}")

    plt.suptitle("Final test-set diagnostics", y=1.00, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "final_test_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    logger.info("Loading + engineering …")
    df = load_raw()
    df = add_velocity_features(df)

    # ── isolate splits (same time cutoff as before) ──
    cutoff = df[TIME_COL].quantile(0.80)
    df_train_val = df[df[TIME_COL] <= cutoff].copy()
    df_test      = df[df[TIME_COL] >  cutoff].copy()

    logger.info("Train+val: %d rows | Test: %d rows (fraud rate %.2f%%)",
                len(df_train_val), len(df_test), df_test[TARGET_COL].mean() * 100)

    # feature column selection must match training
    df_train_val_sorted = df_train_val.sort_values(TIME_COL).reset_index(drop=True)
    t_min, t_max = df_train_val_sorted[TIME_COL].min(), df_train_val_sorted[TIME_COL].max()
    train_end = t_min + (t_max - t_min) * 0.75
    train_df = df_train_val_sorted[df_train_val_sorted[TIME_COL] <= train_end]

    cat_cols, num_cols = get_feature_cols(train_df)
    feature_cols = num_cols + cat_cols

    X_test = df_test[feature_cols].copy()
    for c in cat_cols:
        X_test[c] = pd.Categorical(X_test[c], categories=train_df[c].astype("category").cat.categories)
    y_test = df_test[TARGET_COL]

    # ── load artifacts ──
    logger.info("Loading champion model, calibrator, threshold …")
    model      = lgb.Booster(model_file="models/lgbm_champion.txt")
    calibrator = joblib.load("models/isotonic_calibrator.pkl")
    threshold  = json.loads(Path("models/operating_threshold.json").read_text())["chosen_threshold"]

    # ── score, calibrate, evaluate ──
    y_score_raw = model.predict(X_test)
    y_score_cal = calibrator.transform(y_score_raw)

    metrics = evaluate(y_test, y_score_cal, k=200)
    brier   = brier_score_loss(y_test, y_score_cal)

    # ── bootstrap 95% CIs ──
    logger.info("Computing 95%% bootstrap CIs (1000 samples) …")
    from sklearn.metrics import roc_auc_score, average_precision_score
    ci_roc = bootstrap_ci(y_test, y_score_cal, roc_auc_score)
    ci_pr  = bootstrap_ci(y_test, y_score_cal, average_precision_score)
    ci_p200 = bootstrap_ci(y_test, y_score_cal, lambda y, s: precision_at_k(pd.Series(y), s, k=200))

    # ── apply threshold and compute business metrics on test ──
    y_pred = (y_score_cal >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())

    context = compute_business_context(df_test)
    n_days  = context["n_days"]
    daily_savings   = tp * context["avg_fraud_amt"] / n_days
    daily_cost      = (tp + fp) * REVIEW_COST_USD  / n_days
    daily_net       = daily_savings - daily_cost
    annual_net      = daily_net * 365

    alerts_per_day = (tp + fp) / n_days
    test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    test_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # ── report ──
    print("\n" + "=" * 65)
    print("  FINAL TEST-SET EVALUATION")
    print("=" * 65)
    print(f"  Test rows        : {len(y_test):,}")
    print(f"  Test frauds      : {int(y_test.sum()):,} ({y_test.mean()*100:.2f}%)")
    print(f"  Test window days : {n_days:.1f}")
    print("-" * 65)
    print(f"  ROC-AUC       : {metrics['roc_auc']:.4f}  95% CI [{ci_roc[0]:.4f}, {ci_roc[1]:.4f}]")
    print(f"  PR-AUC        : {metrics['pr_auc']:.4f}  95% CI [{ci_pr[0]:.4f}, {ci_pr[1]:.4f}]")
    print(f"  Precision@200 : {metrics['precision@k']:.4f}  95% CI [{ci_p200[0]:.4f}, {ci_p200[1]:.4f}]")
    print(f"  Brier score   : {brier:.5f}")
    print("-" * 65)
    print(f"  Operating threshold : {threshold:.4f}")
    print(f"  Alerts / day        : {alerts_per_day:.1f}  (capacity {ANALYST_CAPACITY_DAILY})")
    print(f"  Precision           : {test_precision:.4f}")
    print(f"  Recall              : {test_recall:.4f}")
    print(f"  TP / FP / FN        : {tp:,} / {fp:,} / {fn:,}")
    print("-" * 65)
    print(f"  Daily net savings   : ${daily_net:,.2f}")
    print(f"  Annual (× 365)      : ${annual_net:,.0f}")
    print("=" * 65)

    # ── save all artifacts ──
    out_dir = Path("reports")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    plot_final_diagnostics(y_test, y_score_cal, threshold, context, out_dir / "figures")

    final_report = {
        "test_size":         len(y_test),
        "test_frauds":       int(y_test.sum()),
        "test_fraud_rate":   float(y_test.mean()),
        "test_window_days":  float(n_days),
        "metrics": {
            "roc_auc":       {"value": metrics["roc_auc"],       "ci_95": ci_roc},
            "pr_auc":        {"value": metrics["pr_auc"],        "ci_95": ci_pr},
            "precision_at_200": {"value": metrics["precision@k"], "ci_95": ci_p200},
            "brier":         brier,
        },
        "operating_point": {
            "threshold":     float(threshold),
            "alerts_per_day": float(alerts_per_day),
            "precision":     float(test_precision),
            "recall":        float(test_recall),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "business_impact": {
            "avg_fraud_amt_usd":  context["avg_fraud_amt"],
            "review_cost_usd":    REVIEW_COST_USD,
            "daily_capacity":     ANALYST_CAPACITY_DAILY,
            "daily_net_savings":  daily_net,
            "annual_net_savings": annual_net,
        },
    }
    with open(out_dir / "final_test_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    logger.info("Saved → reports/final_test_report.json")

    # ── MLflow ──
    mlflow.set_experiment("champion_model")
    with mlflow.start_run(run_name="final_test_evaluation"):
        mlflow.log_metric("test_roc_auc",       metrics["roc_auc"])
        mlflow.log_metric("test_pr_auc",        metrics["pr_auc"])
        mlflow.log_metric("test_precision_200", metrics["precision@k"])
        mlflow.log_metric("test_brier",         brier)
        mlflow.log_metric("test_precision_at_threshold", test_precision)
        mlflow.log_metric("test_recall_at_threshold",    test_recall)
        mlflow.log_metric("test_daily_net_savings",      daily_net)
        mlflow.log_metric("test_annual_net_savings",     annual_net)
        mlflow.log_artifact(str(out_dir / "figures" / "final_test_diagnostics.png"))
        mlflow.log_artifact(str(out_dir / "final_test_report.json"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()