"""Loading and fixed temporal validation split for shared experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cleaning import clean_listings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"

TRAIN_START = pd.Timestamp("2021-01-01")
TRAIN_END = pd.Timestamp("2024-04-01")
VALID_START = pd.Timestamp("2024-05-01")
VALID_END = pd.Timestamp("2025-03-01")


def temporal_train_valid_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into the project's fixed, non-overlapping month ranges."""
    if "RENT_APPROVAL_DATE" not in df.columns:
        raise ValueError("Missing split column: RENT_APPROVAL_DATE")

    dates = pd.to_datetime(
        df["RENT_APPROVAL_DATE"], format="%Y-%m", errors="coerce"
    )
    if dates.isna().any():
        raise ValueError("RENT_APPROVAL_DATE contains missing or invalid values.")

    train_mask = dates.between(TRAIN_START, TRAIN_END, inclusive="both")
    valid_mask = dates.between(VALID_START, VALID_END, inclusive="both")
    assigned_mask = train_mask | valid_mask

    if not assigned_mask.all():
        outside = sorted(df.loc[~assigned_mask, "RENT_APPROVAL_DATE"].unique())
        raise ValueError(
            "Rows outside the fixed 2021-01 to 2025-03 experiment window: "
            f"{outside}"
        )
    if (train_mask & valid_mask).any():
        raise RuntimeError("Temporal split ranges overlap.")

    # Boolean .loc keeps the source row order and original index labels.
    return df.loc[train_mask].copy(), df.loc[valid_mask].copy()


def load_data(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train.csv, apply shared cleaning, and return train/validation rows."""
    raw = pd.read_csv(Path(train_path))
    cleaned = clean_listings(raw)
    return temporal_train_valid_split(cleaned)
