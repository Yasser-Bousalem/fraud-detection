import os
import pytest

requires_data = pytest.mark.skipif(
    not os.path.exists("data/raw/train_transaction.csv"),
    reason="requires IEEE-CIS raw data (not available in CI)",
)