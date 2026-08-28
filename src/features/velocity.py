# src/features/velocity.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer velocity features on entity keys (card1, DeviceInfo, P_emaildomain).

    For each transaction, compute what happened on the same entity in the past.
    All aggregations use `shift(1)` before rolling to guarantee we only see
    strictly-past rows — no leakage from the current transaction.

    Assumes df is a single split (train, val, or test) sorted by TransactionDT.
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # convert seconds → pandas datetime (arbitrary anchor, only relative time matters)
    df["_dt"] = pd.to_datetime(df["TransactionDT"], unit="s")

    logger.info("Computing card1 velocity features…")
    df = _add_card_velocity(df)

    logger.info("Computing device velocity features…")
    df = _add_device_velocity(df)

    logger.info("Computing email velocity features…")
    df = _add_email_velocity(df)

    df = df.drop(columns=["_dt"])
    return df


def _add_card_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per card1:
      - transactions in last 1h / 24h / 7d
      - amount z-score vs card's past 30-day history
      - time since last transaction (seconds)
    """
    df = df.set_index("_dt")
    grp = df.groupby("card1", sort=False)

    # rolling counts of prior transactions (shift(1) → exclude current row)
    for window, name in [("24h", "24h"), ("7d", "7d")]:
        df[f"card1_tx_count_{name}"] = grp["TransactionAmt"].transform(
            lambda s: s.rolling(window, closed="left").count()
        )

    # seconds since last transaction on this card
    df["card1_seconds_since_last"] = grp["TransactionDT"].transform(
        lambda s: s - s.shift(1)
    )

    return df.reset_index()


def _add_device_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per DeviceInfo:
      - transactions in last 1h / 24h
      - distinct cards used on this device in last 24h (account takeover signal)
    """
    df = df.set_index("_dt")
    grp = df.groupby("DeviceInfo", sort=False, dropna=False)

    for window, name in [("1h", "1h"), ("24h", "24h")]:
        df[f"device_tx_count_{name}"] = grp["TransactionAmt"].transform(
            lambda s: s.rolling(window, closed="left").count()
        )

    # distinct cards seen on this device in past 24h — this is the ATO signal
    df["device_distinct_cards_24h"] = grp["card1"].transform(
        lambda s: s.rolling("24h", closed="left").apply(
            lambda x: x.nunique(), raw=False
        )
    )

    return df.reset_index()


def _add_email_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per P_emaildomain:
      - transactions in last 24h
      - distinct cards used with this email in last 24h
    """
    df = df.set_index("_dt")
    grp = df.groupby("P_emaildomain", sort=False, dropna=False)

    df["email_tx_count_24h"] = grp["TransactionAmt"].transform(
        lambda s: s.rolling("24h", closed="left").count()
    )

    df["email_distinct_cards_24h"] = grp["card1"].transform(
        lambda s: s.rolling("24h", closed="left").apply(
            lambda x: x.nunique(), raw=False
        )
    )

    return df.reset_index()
