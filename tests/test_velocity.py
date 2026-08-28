def test_gap_larger_than_window_gives_zero_count():
    import pandas as pd
    from src.features.velocity import add_velocity_features
    
    df = pd.DataFrame({
        "TransactionID":   [1, 2],
        "TransactionDT":   [36_000, 36_000 + 90 * 60],   # 10:00 and 11:30
        "TransactionAmt":  [10.0, 80.0],
        "card1":           [42, 42],
        "isFraud":         [0, 0],
        "DeviceInfo":      ["ios", "ios"],
        "P_emaildomain":   ["gmail.com", "gmail.com"],
    })
    
    out = add_velocity_features(df)
    
    # second transaction's 1h count must be 0 or NaN (both mean "no prior activity")
    # what it must NOT be is 1 — that would be the shifted-value carryover bug
    value = out["card1_tx_count_1h"].iloc[1]
    assert value != 1, "Shifted-value carryover bug is back"
    assert pd.isna(value) or value == 0, f"Expected 0 or NaN, got {value}"