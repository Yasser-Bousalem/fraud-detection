from tests.conftest import requires_data

@requires_data
def test_velocity_counts_are_bounded():
    """
    Regression test for the coarse-entity bug: card1 velocity counts must be
    bounded and reasonable, not accumulating into the hundreds/thousands 
    that indicated a broken window or coarse entity key.
    """
    from src.data.load import load_raw
    from src.features.velocity import add_velocity_features

    df = load_raw(nrows=50_000)
    df = add_velocity_features(df)

    # a card1 segment shouldn't show absurd 24h counts (the bug produced 700+)
    # median should be small; we allow a generous ceiling on the median
    median_24h = df["card1_tx_count_24h"].median()
    assert median_24h < 50, (
        f"card1_tx_count_24h median={median_24h} is implausibly high — "
        "possible coarse-entity or windowing regression"
    )