"""Deterministic baseline features and train-fitted preprocessing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ("floor_area_sqm", "flat_age", "month_index")
CATEGORICAL_FEATURES = ("TOWN", "FLAT_TYPE")
BASELINE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_REQUIRED_COLUMNS = (
    "RENT_APPROVAL_DATE",
    "FLOOR_AREA_SQM",
    "LEASE_COMMENCE_DATE",
    "TOWN",
    "FLAT_TYPE",
)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the five deterministic baseline features while preserving the index."""
    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for baseline features: {missing}")

    dates = pd.to_datetime(
        df["RENT_APPROVAL_DATE"], format="%Y-%m", errors="coerce"
    )
    if dates.isna().any():
        bad_count = int(dates.isna().sum())
        raise ValueError(
            f"RENT_APPROVAL_DATE contains {bad_count} missing or invalid value(s)."
        )

    features = pd.DataFrame(index=df.index)
    features["floor_area_sqm"] = pd.to_numeric(
        df["FLOOR_AREA_SQM"], errors="coerce"
    )

    lease_year = pd.to_numeric(df["LEASE_COMMENCE_DATE"], errors="coerce")
    flat_age = dates.dt.year - lease_year
    features["flat_age"] = flat_age.mask(flat_age < 0)
    features["month_index"] = (
        (dates.dt.year - 2021) * 12 + dates.dt.month - 1
    )

    for column in CATEGORICAL_FEATURES:
        values = df[column].astype("string")
        features[column] = values.astype(object).where(values.notna(), np.nan)

    return features.replace([np.inf, -np.inf], np.nan)


def make_preprocessor(
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Return a new, unfitted preprocessor for baseline or extended features.

    Keep this object inside a model Pipeline and fit that Pipeline on training
    data only. Extra feature experiments can pass their own column lists without
    changing ``make_features`` or the baseline constants.
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="MISSING"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ]
    )
