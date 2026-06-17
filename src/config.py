from pathlib import Path

ROOT_DIR     = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
INTERIM_DIR  = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR   = ROOT_DIR / "models"
REPORTS_DIR  = ROOT_DIR / "reports"

RANDOM_SEED = 42
TARGET_COL  = "isFraud"
ID_COL      = "TransactionID"
TIME_COL    = "TransactionDT"