import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.config import TARGET_COL, ID_COL

# columns that are never features
_DROP = {TARGET_COL, ID_COL}


def get_feature_cols(df: pd.DataFrame) -> tuple[list, list]:
    """Return (cat_cols, num_cols) — columns used as model features."""
    cat_cols = [
        c
        for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in _DROP
    ]
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in _DROP
    ]
    return cat_cols, num_cols


def build_preprocessor(cat_cols: list, num_cols: list) -> ColumnTransformer:
    """
    Baseline preprocessor:
      - numerics    : mean imputation (LightGBM handles NaN natively but
                      we keep this explicit for the baseline)
      - categoricals: fill NaN → ordinal encode
    """
    num_transformer = SimpleImputer(strategy="mean")

    cat_transformer = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    preprocessor.set_output(transform="pandas")

    return preprocessor
