import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def precision_at_k(y_true: pd.Series, y_score: np.ndarray, k: int = 200) -> float:
    """Of the top-k scored transactions, what fraction are actually fraud?"""
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(y_true.iloc[top_k_idx].mean())


def evaluate(y_true: pd.Series, y_score: np.ndarray, k: int = 200) -> dict:
    return {
        "roc_auc":     roc_auc_score(y_true, y_score),
        "pr_auc":      average_precision_score(y_true, y_score),
        "precision@k": precision_at_k(y_true, y_score, k),
        "k":           k,
    }


def print_metrics(metrics: dict, split: str = "val") -> None:
    print(f"\n{'─'*35}")
    print(f"  Results on {split} set")
    print(f"{'─'*35}")
    print(f"  ROC-AUC       : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC        : {metrics['pr_auc']:.4f}")
    print(f"  Precision@{metrics['k']:<3} : {metrics['precision@k']:.4f}")
    print(f"{'─'*35}\n")