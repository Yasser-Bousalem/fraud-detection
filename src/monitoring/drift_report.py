# src/monitoring/drift_report.py

import logging
from pathlib import Path

import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

from src.config import TARGET_COL, TIME_COL
from src.data.load import load_raw
from src.features.velocity import add_velocity_features

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Loading data …")
    df = load_raw().pipe(add_velocity_features)

    t60 = df[TIME_COL].quantile(0.60)
    t80 = df[TIME_COL].quantile(0.80)

    reference = df[df[TIME_COL] <= t60].copy()
    current   = df[df[TIME_COL] >  t80].copy()

    report_cols = [
        "TransactionAmt", "card1_tx_count_7d", "card1_tx_count_24h",
        "device_distinct_cards_24h", "email_distinct_cards_24h",
        "C1", "C13", "C14", "D1", "D2", "D15", "addr1", "dist1",
        TARGET_COL,
    ]
    report_cols = [c for c in report_cols if c in df.columns]

    ref = reference[report_cols].copy()
    cur = current[report_cols].copy()

    logger.info("Wrapping in Evidently Datasets …")
    # 0.7.x requires a DataDefinition + Dataset wrapper
    data_def = DataDefinition()   # auto-infers column types

    ref_ds = Dataset.from_pandas(ref, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur, data_definition=data_def)

    logger.info("Generating drift report …")
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_ds, current_data=cur_ds)

    Path("reports").mkdir(exist_ok=True)
    out_path = "reports/drift_report_evidently.html"
    result.save_html(out_path)

    logger.info("Saved → %s", out_path)
    print(f"\nOpen {out_path} in a browser.")


if __name__ == "__main__":
    main()