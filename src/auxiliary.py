"""Load cleaned auxiliary tables and attach them to listing feature rows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.cleaning import clean_auxiliary
from src.validation import PROJECT_ROOT


DEFAULT_AUX_DIR = PROJECT_ROOT / "data" / "auxiliary"
EARTH_RADIUS_KM = 6371.0
SCHOOL_COUNT_RADII_KM = (1, 2)

HDB_KEYS = ["TOWN", "BLOCK", "STREET"]
_COORDS = ["LATITUDE", "LONGITUDE"]
_AUX_FILES = {
    "hdb": "sg-hdb-block.csv",
    "mrt": "sg-mrt-stations.csv",
    "schools": "sg-schools.csv",
    "malls": "sg-shopping-malls.csv",
    "coe": "sg-coe-prices.csv",
    "stock": "sg-stock-prices.csv",
}


def load_auxiliary(aux_dir: str | Path = DEFAULT_AUX_DIR) -> dict[str, pd.DataFrame]:
    """Read and clean the six auxiliary CSVs. Listing tables are not loaded."""
    folder = Path(aux_dir)
    frames = {
        name: pd.read_csv(folder / filename) for name, filename in _AUX_FILES.items()
    }
    cleaned = clean_auxiliary(
        frames["hdb"],
        frames["mrt"],
        frames["schools"],
        frames["malls"],
        frames["coe"],
    )
    # Stock prices have no cleaning rule; only rows without a price are dropped.
    cleaned["stock"] = frames["stock"].dropna(subset=["ADJUSTED_CLOSE"])
    return cleaned


def attach_auxiliary(
    X: pd.DataFrame, auxiliary: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Left-join the tables from ``make_auxiliary`` onto ``make_listing`` output.

    Only the keys present in ``auxiliary`` are attached, so a table-level
    ablation is just a smaller dict. MRT, schools, and malls need the HDB
    coordinates, so they require ``"hdb"``. Row count and index are preserved.
    """
    poi = [name for name in ("mrt", "schools", "malls") if name in auxiliary]
    if poi and "hdb" not in auxiliary:
        raise ValueError(f"{poi} need HDB coordinates; include 'hdb' as well.")

    out = X.copy()
    if "hdb" in auxiliary:
        out = out.join(auxiliary["hdb"].set_index(HDB_KEYS), on=HDB_KEYS)
    for name in ("coe", "stock"):
        if name in auxiliary:
            out = out.join(auxiliary[name].set_index("month_index"), on="month_index")

    if "mrt" in auxiliary:
        mrt = auxiliary["mrt"]
        distance, nearest = _nearest(out, mrt)
        out["mrt_distance_km"] = distance
        out["mrt_name"] = _take(mrt["NAME"], nearest)
        out["mrt_line_count"] = _take(mrt["LINE_COUNT"], nearest)
    if "schools" in auxiliary:
        schools = auxiliary["schools"]
        out["primary_school_distance_km"] = _nearest(out, schools[schools["IS_PRIMARY"]])[0]
        out["secondary_school_distance_km"] = _nearest(out, schools[schools["IS_SECONDARY"]])[0]
        for radius in SCHOOL_COUNT_RADII_KM:
            out[f"schools_within_{radius}km"] = _count_within(out, schools, radius)
    if "malls" in auxiliary:
        malls = auxiliary["malls"]
        distance, nearest = _nearest(out, malls)
        out["mall_distance_km"] = distance
        out["mall_name"] = _take(malls["NAME"], nearest)

    if len(out) != len(X) or not out.index.equals(X.index):
        raise ValueError("Auxiliary join changed the listing rows.")
    return out


def _tree_and_points(listings: pd.DataFrame, poi: pd.DataFrame):
    points = np.radians(listings[_COORDS].to_numpy(dtype=float))
    valid = np.isfinite(points).all(axis=1)
    tree = BallTree(np.radians(poi[_COORDS].to_numpy(dtype=float)), metric="haversine")
    return tree, points, valid


def _nearest(listings: pd.DataFrame, poi: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Distance (km) to and positional index of the nearest POI; NaN / -1 if no coords."""
    tree, points, valid = _tree_and_points(listings, poi)
    distance = np.full(len(points), np.nan)
    nearest = np.full(len(points), -1)
    if valid.any():
        radius, index = tree.query(points[valid], k=1)
        distance[valid] = radius[:, 0] * EARTH_RADIUS_KM
        nearest[valid] = index[:, 0]
    return distance, nearest


def _count_within(listings: pd.DataFrame, poi: pd.DataFrame, radius_km: float) -> np.ndarray:
    tree, points, valid = _tree_and_points(listings, poi)
    counts = np.full(len(points), np.nan)
    if valid.any():
        counts[valid] = tree.query_radius(
            points[valid], r=radius_km / EARTH_RADIUS_KM, count_only=True
        )
    return counts


def _take(values: pd.Series, positions: np.ndarray) -> np.ndarray:
    taken = values.to_numpy()[np.maximum(positions, 0)]
    return np.where(positions >= 0, taken, np.nan)
