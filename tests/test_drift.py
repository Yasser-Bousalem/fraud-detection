import numpy as np
import pandas as pd
from src.monitoring.drift import compute_psi


def test_psi_identical_distributions_is_zero():
    """Same distribution → PSI ≈ 0."""
    rng = np.random.RandomState(42)
    data = pd.Series(rng.normal(0, 1, 10000))
    psi = compute_psi(data, data)
    assert psi < 0.01, f"Identical distributions should give PSI≈0, got {psi}"


def test_psi_shifted_distribution_is_high():
    """Clearly shifted distribution → high PSI."""
    rng = np.random.RandomState(42)
    ref = pd.Series(rng.normal(0, 1, 10000))
    cur = pd.Series(rng.normal(3, 1, 10000))   # shifted mean by 3 std
    psi = compute_psi(ref, cur)
    assert psi > 0.25, f"Large shift should give PSI>0.25, got {psi}"


def test_psi_handles_nan():
    """NaN values should be dropped, not crash."""
    ref = pd.Series([1.0, 2.0, np.nan, 3.0, 4.0] * 100)
    cur = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0] * 100)
    psi = compute_psi(ref, cur)
    assert not np.isnan(psi)