from src.data.load import load_raw
from src.data.split import time_split


def test_no_temporal_leakage():
    df = load_raw(nrows=10_000)
    train, val, test = time_split(df)

    assert train["TransactionDT"].max() <= val["TransactionDT"].min(), \
        "Leakage: train contains timestamps after val start"
    assert val["TransactionDT"].max() <= test["TransactionDT"].min(), \
        "Leakage: val contains timestamps after test start"


def test_split_sizes():
    df = load_raw(nrows=10_000)
    train, val, test = time_split(df)

    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert len(train) > len(val)   # 60% > 20%
    assert len(val) >= len(test)   # 20% == 20%