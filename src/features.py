"""Deterministic listing/auxiliary features and train-fitted encoders.

Call chain::

    load_data -> make_listing -> [load_auxiliary -> make_auxiliary -> attach_auxiliary]
              -> [scale_numerical] -> [onehot_encoding] -> encode_high_card

The fitted steps take the training frame first and any number of frames to
transform with it (validation, test). They are fitted on the first frame only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.auxiliary import HDB_KEYS


NUMERIC_FEATURES = (
    "FLOOR_AREA_SQM",
    "flat_age",
    "month_index",
    "LATITUDE",
    "LONGITUDE",
    "MAX_FLOOR",
    "mrt_distance_km",
    "mrt_line_count",
    "primary_school_distance_km",
    "secondary_school_distance_km",
    "schools_within_1km",
    "schools_within_2km",
    "mall_distance_km",
    "coe_sum",
    "stock_index",
)
LOW_CARD_FEATURES = (
    "TOWN",
    "FLAT_TYPE",
    "FLAT_MODEL",
    "PLANNING_AREA",
)
HIGH_CARD_FEATURES = (
    "BLOCK",
    "STREET",
    "SUBZONE",
    "mrt_name",
    "mall_name",
)
ALL_FEATURES = NUMERIC_FEATURES + LOW_CARD_FEATURES + HIGH_CARD_FEATURES

# YEAR_COMPLETED tracks flat_age; REGION is a function of TOWN.
_HDB_COLUMNS = ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "SUBZONE", "PLANNING_AREA"]
_MISSING_CATEGORY = "MISSING"


def make_listing(df: pd.DataFrame) -> pd.DataFrame:
    """Build listing features (dates -> month_index / flat_age), preserving the index."""
    dates = pd.to_datetime(df["RENT_APPROVAL_DATE"], format="%Y-%m", errors="coerce")
    features = pd.DataFrame(index=df.index)
    features["month_index"] = (dates.dt.year - 2021) * 12 + dates.dt.month - 1
    flat_age = dates.dt.year - pd.to_numeric(df["LEASE_COMMENCE_DATE"], errors="coerce")
    features["flat_age"] = flat_age.mask(flat_age < 0)

    for column in ALL_FEATURES:
        if column in df.columns and column not in features.columns:
            features[column] = df[column]
    return order_features(features)


def make_auxiliary(
    df_coe: pd.DataFrame | None = None,
    df_mrt: pd.DataFrame | None = None,
    df_schools: pd.DataFrame | None = None,
    df_hdb: pd.DataFrame | None = None,
    df_malls: pd.DataFrame | None = None,
    df_stock: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Turn cleaned auxiliary tables into join-ready tables for ``attach_auxiliary``.

    Only the tables passed in are returned, keyed ``coe``/``mrt``/``schools``/
    ``hdb``/``malls``/``stock``.
    """
    out: dict[str, pd.DataFrame] = {}

    if df_coe is not None:
        # Mean of the two monthly rounds per category, then summed into one index.
        monthly = df_coe.pivot_table(
            index="YEAR_MONTH", columns="CATEGORY", values="QUOTA_PREMIUM", aggfunc="mean"
        )
        out["coe"] = _monthly_table(
            monthly.index, coe_sum=monthly.sum(axis=1, min_count=monthly.shape[1])
        )

    if df_stock is not None:
        # Equal-weight index of SGX stocks priced since the first month, each scaled to 1.0 there.
        monthly = df_stock.pivot_table(
            index=df_stock["DATE"].str[:7],
            columns="SYMBOL",
            values="ADJUSTED_CLOSE",
            aggfunc="mean",
        ).sort_index()
        full = monthly.columns[monthly.iloc[0].notna() & monthly.columns.str.endswith(".SI")]
        relative = monthly[full] / monthly[full].iloc[0]
        out["stock"] = _monthly_table(monthly.index, stock_index=relative.mean(axis=1))

    if df_mrt is not None:
        # Interchanges have one row per line: collapse to one row per station.
        mrt = df_mrt[df_mrt["STATUS"] == "open"]
        lines = mrt["CODE"].str.extract(r"^([a-z]+)", expand=False)
        out["mrt"] = (
            mrt.assign(LINE=lines)
            .groupby("NAME", as_index=False)
            .agg(
                LATITUDE=("LATITUDE", "first"),
                LONGITUDE=("LONGITUDE", "first"),
                LINE_COUNT=("LINE", "nunique"),
            )
        )

    if df_schools is not None:
        level = df_schools["MAINLEVEL_CODE"]
        out["schools"] = pd.DataFrame(
            {
                "LATITUDE": df_schools["LATITUDE"],
                "LONGITUDE": df_schools["LONGITUDE"],
                "IS_PRIMARY": level.str.contains(r"primary|p1", regex=True),
                "IS_SECONDARY": level.str.contains(r"secondary|s1", regex=True),
            }
        ).reset_index(drop=True)

    if df_hdb is not None:
        out["hdb"] = df_hdb.drop_duplicates(HDB_KEYS)[HDB_KEYS + _HDB_COLUMNS]

    if df_malls is not None:
        out["malls"] = df_malls[["NAME", "LATITUDE", "LONGITUDE"]].reset_index(drop=True)

    return out


def _monthly_table(year_month: pd.Index, **values: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(year_month, format="%Y-%m")
    table = pd.DataFrame({name: series.to_numpy() for name, series in values.items()})
    table.insert(0, "month_index", (dates.year - 2021) * 12 + dates.month - 1)
    return table


def order_features(X: pd.DataFrame) -> pd.DataFrame:
    """Keep known feature columns only, in ``ALL_FEATURES`` order."""
    return X.loc[:, [column for column in ALL_FEATURES if column in X.columns]]


def feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return the numeric, low-card, and high-card columns present in ``X``."""
    numeric = [column for column in NUMERIC_FEATURES if column in X.columns]
    low_card = [column for column in LOW_CARD_FEATURES if column in X.columns]
    high_card = [column for column in HIGH_CARD_FEATURES if column in X.columns]
    return numeric, low_card, high_card


def scale_numerical(
    X_train: pd.DataFrame, *X_others: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, ...]:
    """Median-impute and standardise numeric columns (Ridge / linear models)."""
    columns = feature_groups(X_train)[0] if columns is None else columns
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    scaler.fit(imputer.fit_transform(X_train[columns]))
    return tuple(
        X.assign(**dict(zip(columns, scaler.transform(imputer.transform(X[columns])).T)))
        for X in (X_train, *X_others)
    )


def onehot_encoding(
    X_train: pd.DataFrame, *X_others: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, ...]:
    """One-hot encode low-card categoricals; unseen categories become all zeros."""
    columns = feature_groups(X_train)[1] if columns is None else columns
    encoder = OneHotEncoder(
        handle_unknown="ignore", drop="first", sparse_output=False
    ).set_output(transform="pandas")
    encoder.fit(_fill_categories(X_train, columns))
    return tuple(
        pd.concat(
            [X.drop(columns=columns), encoder.transform(_fill_categories(X, columns))],
            axis=1,
        )
        for X in (X_train, *X_others)
    )


def ordinal_encoding(
    X_train: pd.DataFrame, *X_others: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, ...]:
    """Map categoricals to integer codes; unseen categories become -1."""
    _, low_card, high_card = feature_groups(X_train)
    columns = low_card + high_card if columns is None else columns
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64
    )
    encoder.fit(_fill_categories(X_train, columns))
    return tuple(
        X.assign(**dict(zip(columns, encoder.transform(_fill_categories(X, columns)).T)))
        for X in (X_train, *X_others)
    )


_HIGH_CARD_ENCODERS = {"ordinal": ordinal_encoding}


def encode_high_card(
    X_train: pd.DataFrame,
    *X_others: pd.DataFrame,
    method: str = "ordinal",
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, ...]:
    """Encode high-card categoricals. Register new methods in ``_HIGH_CARD_ENCODERS``."""
    if method not in _HIGH_CARD_ENCODERS:
        raise ValueError(
            f"Unknown high-card encoding {method!r}; expected one of {list(_HIGH_CARD_ENCODERS)}."
        )
    columns = feature_groups(X_train)[2] if columns is None else columns
    return _HIGH_CARD_ENCODERS[method](X_train, *X_others, columns=columns)


def _fill_categories(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return X[columns].astype(object).where(X[columns].notna(), _MISSING_CATEGORY)
