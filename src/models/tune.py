# src/models/tune.py

import logging
import json
from pathlib import Path

import optuna
import lightgbm as lgb
import pandas as pd

from src.config import TARGET_COL, TIME_COL, RANDOM_SEED
from src.data.load import load_raw
from src.features.velocity import add_velocity_features
from src.features.pipeline import get_feature_cols
from src.models.evaluate import evaluate
from src.models.cv import walk_forward_splits

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_folds(df_cv: pd.DataFrame, n_folds: int = 5) -> list[dict]:
    """
    Precompute per-fold train/val matrices ONCE.
    
    Preprocessing (feature selection, NaN handling) is done here — never inside
    the Optuna objective, because those steps don't depend on hyperparameters.
    Returns a list of dicts, one per fold, ready to feed to LightGBM.
    """
    splits, df_sorted = walk_forward_splits(df_cv, n_folds=n_folds)
    
    prepared = []
    for fold_i, (train_idx, val_idx) in enumerate(splits, 1):
        train_df = df_sorted.loc[train_idx]
        val_df   = df_sorted.loc[val_idx]
        
        cat_cols, num_cols = get_feature_cols(train_df)
        feature_cols = num_cols + cat_cols
        
        # LightGBM handles NaN natively — no imputation.
        # For categoricals, use pandas category dtype so LightGBM
        # treats them as categorical splits (faster + often better).
        X_train = train_df[feature_cols].copy()
        X_val   = val_df[feature_cols].copy()
        
        for c in cat_cols:
            X_train[c] = X_train[c].astype("category")
            # align val categories to train's — new-in-val become NaN
            X_val[c] = pd.Categorical(X_val[c], categories=X_train[c].cat.categories)
        
        prepared.append({
            "fold":         fold_i,
            "X_train":      X_train,
            "y_train":      train_df[TARGET_COL],
            "X_val":        X_val,
            "y_val":        val_df[TARGET_COL],
            "cat_features": cat_cols,
        })
        logger.info(
            "Fold %d prepared: train %d rows, val %d rows",
            fold_i, len(X_train), len(X_val),
        )
    
    return prepared


def objective(trial: optuna.Trial, folds: list[dict]) -> float:
    params = {
        "n_estimators":     1000,
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 255),
        "max_depth":        trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 0, 7),
        "lambda_l1":        trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2":        trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "random_state":     RANDOM_SEED,
        "n_jobs":           -1,
        "verbose":          -1,
        "metric":           "average_precision",
        "force_row_wise":   True,
    }

    fold_scores = []
    for fold in folds:
        model = lgb.LGBMClassifier(**params)
        model.fit(
            fold["X_train"], fold["y_train"],
            eval_set        = [(fold["X_val"], fold["y_val"])],
            categorical_feature = fold["cat_features"],
            callbacks       = [lgb.early_stopping(50, verbose=False)],
        )

        y_score = model.predict_proba(fold["X_val"])[:, 1]
        pr_auc  = evaluate(fold["y_val"], y_score, k=200)["pr_auc"]
        fold_scores.append(pr_auc)

        logger.info(
            "  Trial %d fold %d — PR-AUC=%.4f (iter=%d)",
            trial.number, fold["fold"], pr_auc, model.best_iteration_,
        )

        # aggressive pruning: kill bad trials after fold 1
        trial.report(sum(fold_scores) / len(fold_scores), fold["fold"])
        if trial.should_prune():
            raise optuna.TrialPruned()

    return sum(fold_scores) / len(fold_scores)


def tune(n_trials: int = 30, n_folds: int = 5):
    logger.info("Loading + engineering …")
    df = load_raw()
    df = add_velocity_features(df)

    cutoff = df[TIME_COL].quantile(0.80)
    df_cv = df[df[TIME_COL] <= cutoff].copy()

    logger.info("Preparing folds (preprocessed ONCE) …")
    folds = prepare_folds(df_cv, n_folds=n_folds)

    study = optuna.create_study(
        study_name     = "fraud_lgbm",
        direction      = "maximize",
        storage        = "sqlite:///optuna_studies.db",
        load_if_exists = True,
        pruner         = optuna.pruners.MedianPruner(n_startup_trials = 5,n_warmup_steps=2),
    )

    def _callback(study, trial):
        state = trial.state.name
        val = trial.value if trial.value is not None else float("nan")
        logger.info(
            "Trial %3d [%s] | PR-AUC=%.4f | best=%.4f",
            trial.number, state, val, study.best_value,
        )

    study.optimize(
        lambda t: objective(t, folds),
        n_trials  = n_trials,
        callbacks = [_callback],
    )

    print("\n" + "=" * 55)
    print("  OPTUNA RESULTS")
    print("=" * 55)
    print(f"Best PR-AUC : {study.best_value:.4f}")
    print(f"Best params :")
    for k, v in study.best_params.items():
        print(f"  {k:<20} {v}")
    print("=" * 55)

    Path("models").mkdir(exist_ok=True)
    with open("models/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info("Saved best params → models/best_params.json")

    return study


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tune(n_trials=30, n_folds=5)