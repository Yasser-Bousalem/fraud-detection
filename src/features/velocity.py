# src/features/velocity.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    logger.info("Computing card1 velocity features…")
    df = _add_card_velocity(df)
    return df


def _count_in_window(df, group_col, window_seconds, out_col):
    """Count of prior transactions per group in a trailing window (closed='left')."""
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    result = np.full(len(df), np.nan)

    for _, sub in df.groupby(group_col, sort=False, dropna=False):
        idx   = sub.index.to_numpy()
        times = sub["TransactionDT"].to_numpy()

        left = 0
        for right in range(len(sub)):
            t_now = times[right]
            while times[left] <= t_now - window_seconds:
                left += 1
            result[idx[right]] = right - left   # rows strictly before, within window

    df[out_col] = result
    return df


def _add_card_velocity(df: pd.DataFrame) -> pd.DataFrame:
    df = _count_in_window(df, "card1", 24*3600,    "card1_tx_count_24h")
    df = _count_in_window(df, "card1", 7*24*3600,  "card1_tx_count_7d")

    df = df.sort_values("TransactionDT").reset_index(drop=True)
    df["card1_seconds_since_last"] = (
        df.groupby("card1", sort=False)["TransactionDT"]
          .transform(lambda s: s - s.shift(1))
    )
    return df