def test_gap_larger_than_window_gives_zero_count():
    import pandas as pd
    from src.features.velocity import add_velocity_features

    # single card1, two transactions 25 hours apart (gap > 24h window)
    df = pd.DataFrame({
        "TransactionID":  [1, 2],
        "TransactionDT":  [86400, 86400 + 25*3600],
        "TransactionAmt": [10.0, 80.0],
        "card1":          [42, 42],
        "isFraud":        [0, 0],
        "DeviceInfo":     ["ios", "ios"],
        "P_emaildomain":  ["gmail.com", "gmail.com"],
    })
    out = add_velocity_features(df)
    # second tx: prior tx was 25h ago, outside 24h window → count 0
    assert out["card1_tx_count_24h"].iloc[1] == 0, \
        "Gap of 25h should give 24h-count of 0"
    # but within 7d window → count 1
    assert out["card1_tx_count_7d"].iloc[1] == 1, \
        "Gap of 25h should give 7d-count of 1"