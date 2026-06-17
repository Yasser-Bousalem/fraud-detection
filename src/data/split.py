import pandas as pd
from src.config import TIME_COL


def time_split(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by TransactionDT (chronological). No shuffling — ever.

    train : first 60% of the time range
    val   : next  20%
    test  : last  20%
    """
    t_min = df[TIME_COL].min()
    t_max = df[TIME_COL].max()
    t_range = t_max - t_min

    train_end = t_min + t_range * train_frac
    val_end = t_min + t_range * (train_frac + val_frac)

    train = df[df[TIME_COL] <= train_end].copy()
    val = df[(df[TIME_COL] > train_end) & (df[TIME_COL] <= val_end)].copy()
    test = df[df[TIME_COL] > val_end]

    _report(train, val, test)
    return train, val, test


def _report(train, val, test):
    for name, split in [("train", train), ("val", val), ("test", test)]:
        n = len(split)
        fraud_rate = split["isFraud"].mean() * 100
        print(f"{name:>5}: {n:>7,} rows | fraud rate: {fraud_rate:.2f}%")
