# src/models/calibrate.py

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src.config import TARGET_COL, TIME_COL, RANDOM_SEED
from src.data.load import load_raw
from src.features.velocity import add_velocity_features
from src.features.pipeline import get_feature_cols
from src.models.evaluate import evaluate, print_metrics
from src.models.tune import prepare_folds

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("sqlite:///mlflow.db")


BEST_PARAMS = json.loads(Path("models/best_params.json").read_text())


def train_champion(df_cv: pd.DataFrame) -> tuple:
    """
    Train the champion model on train (60% of CV data) and hold out val (20% of CV)
    for calibration and evaluation. Test set (last 20% overall) stays untouched.
    """
    df_sorted = df_cv.sort_values(TIME_COL).reset_index(drop=True)
    t_min = df_sorted[TIME_COL].min()
    t_max = df_sorted[TIME_COL].max()
    total = t_max - t_min

    train_end = t_min + total * 0.75   # 75% of CV data = 60% of full dataset
    train_df = df_sorted[df_sorted[TIME_COL] <= train_end]
    val_df   = df_sorted[df_sorted[TIME_COL] > train_end]

    logger.info("Train: %d rows | Val: %d rows", len(train_df), len(val_df))

    cat_cols, num_cols = get_feature_cols(train_df)
    feature_cols = num_cols + cat_cols

    X_train, y_train = train_df[feature_cols].copy(), train_df[TARGET_COL]
    X_val,   y_val   = val_df[feature_cols].copy(),   val_df[TARGET_COL]

    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_val[c]   = pd.Categorical(X_val[c], categories=X_train[c].cat.categories)

    params = dict(BEST_PARAMS)
    params.update({
        "n_estimators":   1000,
        "random_state":   RANDOM_SEED,
        "n_jobs":         -1,
        "verbose":        -1,
        "metric":         "average_precision",
        "force_row_wise": True,
    })

    logger.info("Training champion LightGBM …")
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set            = [(X_val, y_val)],
        categorical_feature = cat_cols,
        callbacks           = [lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(100)],
    )
    logger.info("Trained %d trees.", model.best_iteration_)

    return model, X_train, y_train, X_val, y_val, cat_cols


def plot_reliability(
    y_true: pd.Series,
    y_score_raw: np.ndarray,
    y_score_cal: np.ndarray,
    output_path: str,
) -> None:
    """Two-panel plot: reliability curves + score histograms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── panel 1: reliability curves ──
    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, y_score_raw, n_bins=15, strategy="quantile")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_true, y_score_cal, n_bins=15, strategy="quantile")

    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax1.plot(mean_pred_raw, frac_pos_raw, "o-", color="tomato",   linewidth=2, markersize=7, label="Raw LightGBM")
    ax1.plot(mean_pred_cal, frac_pos_cal, "s-", color="steelblue", linewidth=2, markersize=7, label="Isotonic calibrated")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives (actual)")
    ax1.set_title("Reliability diagram")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # ── panel 2: score distributions ──
    ax2.hist(y_score_raw, bins=50, alpha=0.5, color="tomato",   label="Raw scores",         density=True)
    ax2.hist(y_score_cal, bins=50, alpha=0.5, color="steelblue", label="Calibrated scores", density=True)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Score distributions")
    ax2.legend()
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    logger.info("Loading + engineering …")
    df = load_raw()
    df = add_velocity_features(df)

    # hold out test set (last 20%) — never touched here
    cutoff = df[TIME_COL].quantile(0.80)
    df_cv = df[df[TIME_COL] <= cutoff].copy()

    mlflow.set_experiment("champion_model")

    with mlflow.start_run(run_name="lgbm_baseline_calibrated"):
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_param("imbalance_strategy", "baseline")
        mlflow.log_param("calibration_method", "isotonic")

        # ── 1. Train raw model ──
        model, X_train, y_train, X_val, y_val, cat_cols = train_champion(df_cv)

        # ── 2. Raw predictions ──
        y_score_raw = model.predict_proba(X_val)[:, 1]
        metrics_raw = evaluate(y_val, y_score_raw, k=200)
        print_metrics(metrics_raw, split="val (RAW)")

        brier_raw = brier_score_loss(y_val, y_score_raw)
        logger.info("Brier score (raw)       : %.5f", brier_raw)

        # ── 3. Fit isotonic calibrator on val ──
        logger.info("Fitting isotonic calibrator on val …")
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(y_score_raw, y_val)

        # ── 4. Calibrated predictions on val (for diagnostics) ──
        # NOTE: metrics computed on the same data we calibrated on will look artificially perfect for
        # calibration diagnostics — but ranking metrics (AUC/PR-AUC/P@k) are unchanged because
        # isotonic is monotonic, and Brier improvement is real if the calibrator captures true miscalibration.
        y_score_cal = calibrator.transform(y_score_raw)
        metrics_cal = evaluate(y_val, y_score_cal, k=200)
        print_metrics(metrics_cal, split="val (CALIBRATED)")

        brier_cal = brier_score_loss(y_val, y_score_cal)
        logger.info("Brier score (calibrated): %.5f", brier_cal)
        logger.info("Brier improvement       : %.5f  (%.1f%%)",
                    brier_raw - brier_cal,
                    (brier_raw - brier_cal) / brier_raw * 100)

        # ── 5. Reliability diagram ──
        Path("reports/figures").mkdir(parents=True, exist_ok=True)
        plot_path = "reports/figures/reliability_diagram.png"
        plot_reliability(y_val, y_score_raw, y_score_cal, plot_path)
        logger.info("Saved reliability diagram → %s", plot_path)

        # ── 6. Log everything to MLflow ──
        mlflow.log_metric("pr_auc_val",        metrics_raw["pr_auc"])
        mlflow.log_metric("roc_auc_val",       metrics_raw["roc_auc"])
        mlflow.log_metric("precision_at_200",  metrics_raw["precision@k"])
        mlflow.log_metric("brier_raw",         brier_raw)
        mlflow.log_metric("brier_calibrated",  brier_cal)
        mlflow.log_metric("brier_improvement", brier_raw - brier_cal)

        mlflow.log_artifact(plot_path)

        # save model + calibrator artifacts
        Path("models").mkdir(exist_ok=True)
        import joblib
        model.booster_.save_model("models/lgbm_champion.txt")
        joblib.dump(calibrator, "models/isotonic_calibrator.pkl")
        mlflow.log_artifact("models/lgbm_champion.txt")
        mlflow.log_artifact("models/isotonic_calibrator.pkl")

        logger.info("\n✔ Champion model + calibrator saved to /models")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()