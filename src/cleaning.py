"""Cleaning rules migrated from ``scripts/cleaning.ipynb``."""

from __future__ import annotations

import numpy as np
import pandas as pd


CONSTANT_COLS = ["FURNISHED", "FEE"]
MONTH_TO_NUM = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def normalise_flat_type(series: pd.Series) -> pd.Series:
    """Normalize flat types such as ``3-room`` and ``3 room``."""
    return (
        series.astype(str)
        .str.lower()
        .str.replace("-", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def normalise_text(series: pd.Series) -> pd.Series:
    """Lowercase, strip, and collapse repeated whitespace."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def clean_hdb(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the HDB block auxiliary table."""
    out = df.copy()
    for column in (
        "TOWN",
        "BLOCK",
        "STREET",
        "SUBZONE",
        "PLANNING_AREA",
        "REGION",
    ):
        out[column] = normalise_text(out[column])
    postal = out["POSTAL_CODE"].astype(str).str.strip()
    out["POSTAL_CODE"] = postal.where(~postal.isin(["NIL", "nil"]), np.nan)
    out["YEAR_COMPLETED"] = out["YEAR_COMPLETED"].replace(-1, np.nan)
    out["MAX_FLOOR"] = out["MAX_FLOOR"].replace(-1, np.nan)
    return out


def clean_mrt(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the MRT auxiliary table and mark open stations."""
    out = df.copy()
    for column in (
        "CODE",
        "NAME",
        "STATUS",
        "SUBZONE",
        "PLANNING_AREA",
        "REGION",
    ):
        out[column] = normalise_text(out[column])
    out["IS_OPEN"] = out["STATUS"] == "open"
    return out


def expand_school_street(series: pd.Series) -> pd.Series:
    """Expand the street abbreviations used by the schools table."""
    out = normalise_text(series)
    out = out.str.replace(r"\bave\b", "avenue", regex=True)
    out = out.str.replace(r"\brd\b", "road", regex=True)
    out = out.str.replace(r"\bdr\b", "drive", regex=True)
    out = out.str.replace(r"\blor\b", "lorong", regex=True)
    out = out.str.replace(r"\bst(?=\s+\d)", "street", regex=True)
    return out


def clean_schools(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the schools auxiliary table."""
    out = df.copy()
    if "NATRUE_CODE" in out.columns:
        out = out.rename(columns={"NATRUE_CODE": "NATURE_CODE"})
    for column in (
        "NAME",
        "PLANNING_AREA",
        "REGION",
        "TYPE_CODE",
        "NATURE_CODE",
        "SESSION_CODE",
        "MAINLEVEL_CODE",
    ):
        if column in out.columns:
            out[column] = normalise_text(out[column])
    out["PLANNING_AREA"] = out["PLANNING_AREA"].replace(
        {"seng kang": "sengkang"}
    )
    out["TYPE_CODE"] = out["TYPE_CODE"].replace(
        {"government-aided sch": "government-aided school"}
    )
    out["STREET"] = expand_school_street(out["STREET"])
    out["POSTAL_CODE"] = (
        out["POSTAL_CODE"]
        .astype(str)
        .str.replace(r"\.0$", "", n=1, regex=True)
        .str.zfill(6)
    )
    if "MRT_STATIONS" in out.columns:
        out["MRT_STATIONS"] = normalise_text(out["MRT_STATIONS"])
    if "BUS_LINES" in out.columns:
        out["BUS_LINES"] = normalise_text(out["BUS_LINES"])
    out["SUSPECT_DUPLICATE_COORDS"] = out.duplicated(
        ["LATITUDE", "LONGITUDE"], keep=False
    )
    return out


def clean_malls(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the shopping-mall auxiliary table."""
    out = df.copy()
    out["NAME"] = normalise_text(out["NAME"])
    return out


def parse_numeric_string(series: pd.Series) -> pd.Series:
    """Parse currency/count strings containing dollar signs or commas."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )


def clean_coe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the COE auxiliary table and create its listing join month."""
    out = df.copy()
    out["CATEGORY"] = normalise_text(out["CATEGORY"])
    out["YEAR_MONTH"] = (
        out["YEAR"].astype(str)
        + "-"
        + normalise_text(out["MONTH"])
        .map(MONTH_TO_NUM)
        .astype(str)
        .str.zfill(2)
    )
    for column in (
        "QUOTA_PREMIUM",
        "PREVAILING_QUOTA_PREMIUM",
        "QUOTA",
        "BIDS_RECEIVED",
    ):
        out[column] = parse_numeric_string(out[column])
    return out.drop(columns=["YEAR", "MONTH"])


def clean_listings(
    df: pd.DataFrame, drop_exact_duplicates: bool = False
) -> pd.DataFrame:
    """Clean a listing table and optionally remove exact duplicates."""
    out = df.copy()
    for column in ("TOWN", "BLOCK", "STREET", "FLAT_MODEL"):
        if column in out.columns:
            out[column] = normalise_text(out[column])
    if "FLAT_TYPE" in out.columns:
        out["FLAT_TYPE"] = normalise_flat_type(out["FLAT_TYPE"])
    out = out.drop(columns=[column for column in CONSTANT_COLS if column in out])
    if drop_exact_duplicates:
        out = out.drop_duplicates(keep="first").reset_index(drop=True)
    return out


def clean_auxiliary(
    hdb_df: pd.DataFrame,
    mrt_df: pd.DataFrame,
    schools_df: pd.DataFrame,
    malls_df: pd.DataFrame,
    coe_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Clean all auxiliary tables using the notebook's shared rules."""
    mrt_clean = clean_mrt(mrt_df)
    return {
        "hdb": clean_hdb(hdb_df),
        "mrt": mrt_clean,
        "mrt_open": mrt_clean[mrt_clean["IS_OPEN"]].copy(),
        "schools": clean_schools(schools_df),
        "malls": clean_malls(malls_df),
        "coe": clean_coe(coe_df),
    }
