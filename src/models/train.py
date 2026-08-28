import logging
import lightgbm as lgb

from src.config import TARGET_COL, RANDOM_SEED
from src.data.load import load_raw
from src.data.split import time_split
from src.features.pipeline import get_feature_cols, build_preprocessor
from src.models.evaluate import evaluate, print_metrics
from src.features.velocity import add_velocity_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def train():
    # ── 1. Load & split ───────────────────────────────────────────────
    logger.info("Loading data …")
    df = load_raw()
    train_df, val_df, test_df = time_split(df)

    logger.info("Engineering velocity features…")
    train_df = add_velocity_features(train_df)
    val_df = add_velocity_features(val_df)
    test_df = add_velocity_features(test_df)

    cat_cols, num_cols = get_feature_cols(train_df)
    feature_cols = num_cols + cat_cols

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]

    logger.info("Train: %d rows | Val: %d rows", len(X_train), len(X_val))

    # ── 2. Preprocess ─────────────────────────────────────────────────
    logger.info("Fitting preprocessor …")
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    # ── 3. Train ──────────────────────────────────────────────────────
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info("Class imbalance ratio (scale_pos_weight): %.1f", scale)

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        # scale_pos_weight=scale,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
        metric="average_precision",
    )

    logger.info("Training LightGBM baseline …")
    model.fit(
        X_train_proc,
        y_train,
        eval_set=[(X_val_proc, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(10)],
    )
    print(f"\nBest iteration: {model.best_iteration_}")
    print(f"Total trees fit: {model.n_estimators_}")

    # ── 4. Evaluate ───────────────────────────────────────────────────
    y_score = model.predict_proba(X_val_proc)[:, 1]
    metrics = evaluate(y_val, y_score, k=200)
    print_metrics(metrics, split="val")

    # ── 5. Feature importance ─────────────────────────────────────────
    import pandas as pd

    feature_names = preprocessor.get_feature_names_out()

    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "splits": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)

    print("\nTop 20 features by importance:")
    print(importances.head(20).to_string(index=False))

    print("\nVelocity features specifically:")
    velocity_names = [
        f
        for f in feature_names
        if any(k in f for k in ["count", "distinct", "seconds_since"])
    ]
    velocity_imp = importances[importances["feature"].isin(velocity_names)]
    print(velocity_imp.to_string(index=False))

    return model, preprocessor, metrics


if __name__ == "__main__":
    train()
