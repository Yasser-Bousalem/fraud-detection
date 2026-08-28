# src/models/cv.py

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import TARGET_COL, TIME_COL, RANDOM_SEED
from src.features.pipeline import build_preprocessor, get_feature_cols
from src.models.evaluate import evaluate

logger = logging.getLogger(__name__)


def walk_forward_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    initial_train_fraction: float = 0.5,
    min_val_frauds: int = 500,
) -> tuple[list, pd.DataFrame]:
    """
    Walk-forward splits with auto-computed val_fraction based on n_folds.
    Validates that each val window has enough minority-class examples.
    """
    df_sorted = df.sort_values(TIME_COL).reset_index(drop=True)
    val_fraction = (1 - initial_train_fraction) / n_folds

    t_min = df_sorted[TIME_COL].min()
    t_max = df_sorted[TIME_COL].max()
    total_span = t_max - t_min
    val_span = total_span * val_fraction
    first_train_end = t_min + total_span * initial_train_fraction

    logger.info(
        "CV config: %d folds, val_fraction=%.3f (auto), initial_train=%.2f",
        n_folds,
        val_fraction,
        initial_train_fraction,
    )

    splits = []
    for i in range(n_folds):
        train_end = first_train_end + i * val_span
        val_end = train_end + val_span

        train_idx = df_sorted[df_sorted[TIME_COL] <= train_end].index
        val_idx = df_sorted[
            (df_sorted[TIME_COL] > train_end) & (df_sorted[TIME_COL] <= val_end)
        ].index

        n_val_frauds = df_sorted.loc[val_idx, TARGET_COL].sum()
        if n_val_frauds < min_val_frauds:
            logger.warning(
                "Fold %d has only %d frauds in val — below min_val_frauds=%d. "
                "Consider fewer folds.",
                i + 1,
                n_val_frauds,
                min_val_frauds,
            )

        splits.append((train_idx, val_idx))
        logger.info(
            "Fold %d: train %d rows (fraud %.2f%%) | val %d rows (fraud %.2f%%, n=%d)",
            i + 1,
            len(train_idx),
            df_sorted.loc[train_idx, TARGET_COL].mean() * 100,
            len(val_idx),
            df_sorted.loc[val_idx, TARGET_COL].mean() * 100,
            n_val_frauds,
        )

    return splits, df_sorted


def cross_validate(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Run walk-forward CV. Returns a DataFrame with metrics per fold."""
    splits, df_sorted = walk_forward_splits(df, n_folds=n_folds)

    results = []

    for fold_i, (train_idx, val_idx) in enumerate(splits, 1):
        logger.info("─" * 40)
        logger.info("FOLD %d/%d", fold_i, len(splits))
        logger.info("─" * 40)

        train_df = df_sorted.loc[train_idx].copy()
        val_df = df_sorted.loc[val_idx].copy()

        cat_cols, num_cols = get_feature_cols(train_df)
        feature_cols = num_cols + cat_cols

        X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
        X_val, y_val = val_df[feature_cols], val_df[TARGET_COL]

        preprocessor = build_preprocessor(cat_cols, num_cols)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
            metric="average_precision",
        )

        model.fit(
            X_train_proc,
            y_train,
            eval_set=[(X_val_proc, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        y_score = model.predict_proba(X_val_proc)[:, 1]
        metrics = evaluate(y_val, y_score, k=200)
        metrics["fold"] = fold_i
        metrics["train_rows"] = len(train_df)
        metrics["val_rows"] = len(val_df)
        metrics["best_iter"] = model.best_iteration_
        results.append(metrics)

        logger.info(
            "Fold %d done — ROC-AUC=%.4f PR-AUC=%.4f P@200=%.4f (iter=%d)",
            fold_i,
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["precision@k"],
            model.best_iteration_,
        )

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> None:
    metric_cols = ["roc_auc", "pr_auc", "precision@k"]
    print("\n" + "=" * 55)
    print("  WALK-FORWARD CV SUMMARY")
    print("=" * 55)
    print(
        results[
            ["fold", "train_rows", "val_rows", "best_iter"] + metric_cols
        ].to_string(index=False)
    )
    print("-" * 55)
    print(f"{'Metric':<15}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}")
    for m in metric_cols:
        v = results[m]
        print(
            f"{m:<15}{v.mean():>10.4f}{v.std():>10.4f}{v.min():>10.4f}{v.max():>10.4f}"
        )
    print("=" * 55)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.data.load import load_raw
    from src.features.velocity import add_velocity_features

    logger.info("Loading data …")
    df = load_raw()

    logger.info("Engineering velocity features …")
    df = add_velocity_features(df)

    # remove last 20% (test set) — walk-forward runs only on the first 80%
    from src.config import TIME_COL

    cutoff = df[TIME_COL].quantile(0.80)
    df_cv = df[df[TIME_COL] <= cutoff].copy()
    logger.info("CV data: %d rows (last 20%% held as test)", len(df_cv))

    results = cross_validate(df_cv, n_folds=5)
    summarize(results)

    results.to_csv("reports/cv_results.csv", index=False)
    logger.info("\nSaved to reports/cv_results.csv")
