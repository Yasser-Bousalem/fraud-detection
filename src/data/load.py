import pandas as pd
import logging


from src.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)


def load_raw(
    transaction_file: str = "train_transaction.csv",
    identity_file: str = "train_identity.csv",
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load and join the IEEE-CIS transaction and identity tables.

    The identity table is optional — only ~144K of the 590K transactions
    have identity rows. We left-join so we keep all transactions.

    Parameters
    ----------
    transaction_file : name of the transaction CSV under RAW_DATA_DIR
    identity_file    : name of the identity CSV under RAW_DATA_DIR
    nrows            : if set, read only the first N rows (dev mode)

    Returns
    -------
    pd.DataFrame with shape (~590K, ~434) after join
    """
    trans_path = RAW_DATA_DIR / transaction_file
    id_path = RAW_DATA_DIR / identity_file

    logger.info("Loading transactions from %s", trans_path)
    transactions = pd.read_csv(trans_path, nrows=nrows)
    logger.info("Transactions loaded: %s rows, %s cols", *transactions.shape)

    logger.info("Loading identity from %s", id_path)
    identity = pd.read_csv(id_path, nrows=nrows)
    logger.info("Identity loaded: %s rows, %s cols", *identity.shape)

    logger.info("Left-joining on TransactionID …")
    df = transactions.merge(identity, on="TransactionID", how="left")
    logger.info(
        "After join: %s rows, %s cols  |  identity match rate: %.1f%%",
        *df.shape,
        df["id_01"].notna().mean() * 100,  # id_01 is a reliable proxy for join hit
    )
    df["has_identity"] = df["id_01"].notna().astype(int)
    logger.info("has_identity feature added")

    return df


def describe_target(df: pd.DataFrame) -> None:
    """Print a quick fraud-rate summary — call this right after load_raw."""
    n_total = len(df)
    n_fraud = df["isFraud"].sum()
    print(f"Total transactions : {n_total:,}")
    print(f"Fraud              : {n_fraud:,}  ({n_fraud / n_total * 100:.2f}%)")
    print(f"Legit              : {n_total - n_fraud:,}")
    print(
        f"Identity match     : {df['has_identity'].sum():,}  ({df['has_identity'].mean()*100:.1f}%)"
    )
    print(
        f"TransactionDT range: {df['TransactionDT'].min():,} → {df['TransactionDT'].max():,}  (seconds)"
    )
