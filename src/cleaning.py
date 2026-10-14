"""Shared cleaning rules for the HDB rental listing tables."""

from __future__ import annotations

import pandas as pd


TEXT_COLUMNS = ("TOWN", "STREET", "BLOCK", "FLAT_MODEL")
CONSTANT_COLUMNS = ("FURNISHED", "FEE")


def _normalise_text(series: pd.Series) -> pd.Series:
    """Return lowercase, stripped text with repeated whitespace collapsed."""
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _normalise_flat_type(series: pd.Series) -> pd.Series:
    """Map values such as ``3-room`` and ``3 room`` to ``3 room``."""
    return (
        series.astype("string")
        .str.lower()
        .str.replace("-", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a listing table without removing, reordering, or reindexing rows.

    The function intentionally keeps exact duplicate records because the source
    data has no flat unit number; identical rows can represent real transactions.
    """
    out = df.copy()

    for column in TEXT_COLUMNS:
        if column in out.columns:
            out[column] = _normalise_text(out[column])

    if "FLAT_TYPE" in out.columns:
        out["FLAT_TYPE"] = _normalise_flat_type(out["FLAT_TYPE"])

    return out.drop(
        columns=[column for column in CONSTANT_COLUMNS if column in out.columns]
    )
