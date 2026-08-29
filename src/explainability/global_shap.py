# src/explainability/global_shap.py

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
import mlflow

from src.config import TARGET_COL, TIME_COL
from src.data.load import load_raw
from src.features.velocity import add_velocity_features
from src.features.pipeline import get_feature_cols

logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("sqlite:///mlflow.db")


SAMPLE_SIZE = 5000    # SHAP on 5K rows is representative; full 100K+ is slow overkill


def prepare_data():
    df = load_raw()
    df = add_velocity_features(df)

    cutoff = df[TIME_COL].quantile(0.80)
    df_train_val = df[df[TIME_COL] <= cutoff].copy()

    # rebuild same split as training
    df_sorted = df_train_val.sort_values(TIME_COL).reset_index(drop=True)
    t_min, t_max = df_sorted[TIME_COL].min(), df_sorted[TIME_COL].max()
    train_end = t_min + (t_max - t_min) * 0.75
    train_df = df_sorted[df_sorted[TIME_COL] <= train_end]
    val_df   = df_sorted[df_sorted[TIME_COL] >  train_end]

    cat_cols, num_cols = get_feature_cols(train_df)
    feature_cols = num_cols + cat_cols

    X_val = val_df[feature_cols].copy()
    for c in cat_cols:
        X_val[c] = pd.Categorical(X_val[c], categories=train_df[c].astype("category").cat.categories)

    y_val = val_df[TARGET_COL]
    return X_val, y_val, feature_cols, cat_cols


def compute_shap_values(model, X_sample):
    """TreeExplainer is exact and fast for LightGBM."""
    logger.info("Building TreeExplainer …")
    explainer = shap.TreeExplainer(model)

    logger.info("Computing SHAP values on %d rows …", len(X_sample))
    # for LightGBM binary, shap_values returns raw log-odds contributions
    shap_values = explainer.shap_values(X_sample)

    # newer SHAP versions return array for binary; older return list of 2
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # positive class

    return explainer, shap_values


def plot_global_importance(shap_values, X_sample, out_path, top_n=25):
    """Global bar plot: mean absolute SHAP per feature."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    ordered = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n)[::-1],
            mean_abs[ordered],
            color="steelblue")
    ax.set_yticks(range(top_n)[::-1])
    ax.set_yticklabels([X_sample.columns[i] for i in ordered], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|  (impact on model output)")
    ax.set_title(f"Top {top_n} features by global SHAP importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_beeswarm(shap_values, X_sample, out_path, top_n=20):
    """Beeswarm shows feature effects in distribution — high/low values, direction."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        max_display=top_n, show=False,
    )
    plt.title(f"SHAP summary — top {top_n} features")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def compare_shap_vs_builtin(shap_values, model, X_sample, out_path):
    """Side-by-side of SHAP mean|value| vs LightGBM built-in gain importance."""
    shap_imp = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_sample.columns,
        name="shap",
    )

    gain_imp = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=X_sample.columns,
        name="gain",
    )

    df = pd.concat([shap_imp, gain_imp], axis=1)
    df["shap_rank"] = df["shap"].rank(ascending=False)
    df["gain_rank"] = df["gain"].rank(ascending=False)
    df["rank_diff"] = df["gain_rank"] - df["shap_rank"]

    df.sort_values("shap", ascending=False).head(30).to_csv(out_path)
    return df


def main():
    logger.info("Loading data + model …")
    X_val, y_val, feature_cols, cat_cols = prepare_data()

    # sample for SHAP (representative subset)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_val), size=min(SAMPLE_SIZE, len(X_val)), replace=False)
    X_sample = X_val.iloc[sample_idx].copy()
    logger.info("SHAP sample: %d rows", len(X_sample))

    model = lgb.Booster(model_file="models/lgbm_champion.txt")

    explainer, shap_values = compute_shap_values(model, X_sample)

    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    bar_path      = out_dir / "shap_global_importance.png"
    beeswarm_path = out_dir / "shap_beeswarm.png"
    compare_path  = Path("reports/shap_vs_gain.csv")

    logger.info("Rendering global importance …")
    plot_global_importance(shap_values, X_sample, bar_path)

    logger.info("Rendering beeswarm …")
    plot_beeswarm(shap_values, X_sample, beeswarm_path)

    logger.info("Comparing SHAP vs built-in gain …")
    comparison = compare_shap_vs_builtin(shap_values, model, X_sample, compare_path)

    # save explainer for Day 16 (reason codes) — avoids recomputing base_value
    import joblib
    joblib.dump(explainer, "models/shap_explainer.pkl")

    # log baseline (expected value = model output before any features)
    logger.info("Model base value (log-odds): %.4f", explainer.expected_value)

    # print top 15
    top15 = comparison.sort_values("shap", ascending=False).head(15)
    print("\n" + "=" * 70)
    print("  TOP 15 FEATURES BY SHAP (with gain-importance rank comparison)")
    print("=" * 70)
    print(top15[["shap", "gain", "shap_rank", "gain_rank", "rank_diff"]].to_string())
    print("=" * 70)
    print("\nInterpretation:")
    print("  rank_diff > 0  →  SHAP ranks feature HIGHER than gain does")
    print("  rank_diff < 0  →  SHAP ranks feature LOWER than gain does")

    # MLflow
    mlflow.set_experiment("champion_model")
    with mlflow.start_run(run_name="global_shap"):
        mlflow.log_param("shap_sample_size", len(X_sample))
        mlflow.log_metric("explainer_base_value", float(explainer.expected_value))
        mlflow.log_artifact(str(bar_path))
        mlflow.log_artifact(str(beeswarm_path))
        mlflow.log_artifact(str(compare_path))
        mlflow.log_artifact("models/shap_explainer.pkl")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()