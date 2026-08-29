import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from src.config import TARGET_COL, TIME_COL, RANDOM_SEED
from src.data.load import load_raw
from src.features.velocity import add_velocity_features
from src.models.evaluate import evaluate
from src.models.tune import prepare_folds

logger = logging.getLogger(__name__)

# Load tuned hyperparameters from Day 9
BEST_PARAMS = json.loads(Path("models/best_params.json").read_text())


def base_model_params() -> dict:
    """Tuned params + fixed settings."""
    p = dict(BEST_PARAMS)
    p.update({
        "n_estimators":   1000,
        "random_state":   RANDOM_SEED,
        "n_jobs":         -1,
        "verbose":        -1,
        "metric":         "average_precision",
        "force_row_wise": True,
    })
    return p


def train_fold(fold: dict, params: dict, strategy: str) -> dict:
    """Train one fold with a given imbalance strategy. Returns metrics dict."""
    X_train = fold["X_train"]
    y_train = fold["y_train"]

    # apply strategy to training data ONLY — val is always untouched
    if strategy == "baseline":
        pass
    
    elif strategy == "scale_pos_weight":
        params = dict(params)
        params["scale_pos_weight"] = (y_train == 0).sum() / (y_train == 1).sum()
    
    elif strategy == "undersample_5to1":
        # ratio 5 legit : 1 fraud
        sampler = RandomUnderSampler(
            sampling_strategy = 1/5,   # minority/majority = 0.2
            random_state      = RANDOM_SEED,
        )
        # SMOTE/undersampling can't handle NaN or pandas categorical → 
        # fallback: fill NaN, convert categoricals to codes
        X_train_arr = X_train.copy()
        for c in X_train_arr.select_dtypes(include=["category"]).columns:
            X_train_arr[c] = X_train_arr[c].cat.codes
        X_train_arr = X_train_arr.fillna(-999)
        X_train, y_train = sampler.fit_resample(X_train_arr, y_train)
    
    elif strategy == "smote":
        sampler = SMOTE(
            sampling_strategy = 1/5,
            random_state      = RANDOM_SEED,
            k_neighbors       = 5,
        )
        X_train_arr = X_train.copy()
        for c in X_train_arr.select_dtypes(include=["category"]).columns:
            X_train_arr[c] = X_train_arr[c].cat.codes
        X_train_arr = X_train_arr.fillna(-999)
        X_train, y_train = sampler.fit_resample(X_train_arr, y_train)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # for resampled data, val must also be numeric (no categoricals)
    X_val = fold["X_val"]
    if strategy in ("undersample_5to1", "smote"):
        X_val = X_val.copy()
        # match training-fold categorical encoding
        for c in X_val.select_dtypes(include=["category"]).columns:
            X_val[c] = X_val[c].cat.codes
        X_val = X_val.fillna(-999)
        cat_features = None
    else:
        cat_features = fold["cat_features"]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set            = [(X_val, fold["y_val"])],
        categorical_feature = cat_features,
        callbacks           = [lgb.early_stopping(50, verbose=False)],
    )

    y_score = model.predict_proba(X_val)[:, 1]
    metrics = evaluate(fold["y_val"], y_score, k=200)
    metrics["best_iter"] = model.best_iteration_
    return metrics


def run_experiment(strategy: str, folds: list[dict]) -> pd.DataFrame:
    """Run one strategy across all folds, return per-fold metrics."""
    logger.info("=" * 55)
    logger.info("STRATEGY: %s", strategy)
    logger.info("=" * 55)

    params = base_model_params()
    fold_results = []

    with mlflow.start_run(run_name=strategy):
        mlflow.log_param("strategy", strategy)
        mlflow.log_params({k: v for k, v in params.items()
                           if k in BEST_PARAMS})

        for fold in folds:
            metrics = train_fold(fold, params, strategy)
            metrics["fold"] = fold["fold"]
            fold_results.append(metrics)
            logger.info(
                "  Fold %d — PR-AUC=%.4f  P@200=%.4f  iter=%d",
                fold["fold"], metrics["pr_auc"],
                metrics["precision@k"], metrics["best_iter"],
            )

        df_results = pd.DataFrame(fold_results)

        # log mean/std of each metric — MLflow disallows '@' in metric names
        MLFLOW_NAME_MAP = {
            "roc_auc":     "roc_auc",
            "pr_auc":      "pr_auc",
            "precision@k": "precision_at_k",
        }
        for m, mlflow_name in MLFLOW_NAME_MAP.items():
            mlflow.log_metric(f"{mlflow_name}_mean", df_results[m].mean())
            mlflow.log_metric(f"{mlflow_name}_std",  df_results[m].std())

    return df_results


def main():
    logger.info("Loading + engineering …")
    df = load_raw()
    df = add_velocity_features(df)

    cutoff = df[TIME_COL].quantile(0.80)
    df_cv = df[df[TIME_COL] <= cutoff].copy()

    logger.info("Preparing folds …")
    folds = prepare_folds(df_cv, n_folds=5)

    mlflow.set_experiment("imbalance_strategies")

    all_results = []
    for strategy in ["baseline", "scale_pos_weight", "undersample_5to1", "smote"]:
        df_r = run_experiment(strategy, folds)
        df_r["strategy"] = strategy
        all_results.append(df_r)

    all_df = pd.concat(all_results, ignore_index=True)

    # summary table
    summary = (
        all_df.groupby("strategy")[["roc_auc", "pr_auc", "precision@k"]]
        .agg(["mean", "std"])
        .round(4)
    )

    print("\n" + "=" * 70)
    print("  IMBALANCE STRATEGY COMPARISON")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70)

    Path("reports").mkdir(exist_ok=True)
    all_df.to_csv("reports/imbalance_experiment.csv", index=False)
    summary.to_csv("reports/imbalance_summary.csv")
    logger.info("\nSaved → reports/imbalance_experiment.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()