"""One controlled E6 follow-up: damped market trend + CatBoost residuals.

The CatBoost residual model has no long-term time variable.  It keeps only
calendar_month for seasonality, plus listing, full-address, and auxiliary
features.  Existing comparator predictions are reused without refitting.

Run: .venv/bin/python -m experiments.e6.market_regression.catboost_trend
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from experiments.e6.market_regression.run import (
    CAT_PARAMS,
    address_features,
    columns,
    extrapolate,
    fit_market,
)
from experiments.e6.run_e6 import digest, score
from experiments.error_analysis.e5 import FOLDS, split_at
from experiments.feature_engineering.utils.core import ROOT
from src.auxiliary import load_auxiliary
from src.features import make_auxiliary
from src.validation import load_data


DEST = Path(__file__).resolve().parent / "catboost_trend_results"
NEW_MODEL = "Trend_damped_CatBoost_no_long_time"
COMPARATORS = {
    "B_CatBoost": "B_address_catboost",
    "Trend_damped_LightGBM": "Trend_damped_LGBM",
}
ENSEMBLE = "B_CatBoost_plus_Trend_LightGBM_equal"


def residual_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove long-run time while retaining calendar-month seasonality."""
    out = frame.drop(columns=[c for c in ("month_index", "year") if c in frame])
    if "calendar_month" not in out:
        raise ValueError("calendar_month is required for seasonal residual effects")
    if {"month_index", "year"}.intersection(out.columns):
        raise RuntimeError("Long-term time variable remained in residual features")
    return out


def catboost_frames(train: pd.DataFrame, valid: pd.DataFrame):
    low, high = columns(train)
    categorical = low + high
    x, z = train.copy(), valid.copy()
    for column in categorical:
        x[column] = x[column].fillna("MISSING").astype(str)
        z[column] = z[column].fillna("MISSING").astype(str)
    for column in x.columns.difference(categorical):
        median = x[column].median()
        x[column] = x[column].fillna(median)
        z[column] = z[column].fillna(median)
    return x, z, categorical


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    train, valid = load_data()
    raw = load_auxiliary()
    auxiliary = make_auxiliary(
        df_hdb=raw["hdb"], df_mrt=raw["mrt"], df_malls=raw["malls"]
    )
    folds = [("formal_2024_04", train, valid)] + [
        (name, *split_at(train, cutoff, horizon))
        for name, cutoff, horizon in FOLDS
    ]
    protocol = {
        "new_model": NEW_MODEL,
        "single_new_candidate": True,
        "trend": "reuse adjusted monthly effects; recent-12-month slope; last-3-month anchor; six-month exponential damping",
        "residual_model": "CatBoost on residual rent after subtracting fitted training-month effect",
        "removed_residual_features": ["month_index", "year"],
        "retained_seasonal_feature": "calendar_month",
        "catboost_params": CAT_PARAMS,
        "comparators": COMPARATORS,
        "ensemble": "fixed 0.5 B CatBoost + 0.5 damped-trend LightGBM",
        "outer_validation_used_for_fitting": False,
        "test_csv_used": False,
        "train_sha256": digest(ROOT / "data/train.csv"),
        "script_sha256": digest(Path(__file__)),
    }
    (DEST / "protocol.json").write_text(json.dumps(protocol, indent=2))

    rows, grouped, trend_rows = [], [], []
    for fold, fit, evaluate in folds:
        full_x = address_features(fit, auxiliary, "C")
        full_z = address_features(evaluate, auxiliary, "C")
        months_x = pd.Series(
            pd.PeriodIndex(fit.RENT_APPROVAL_DATE, freq="M").asi8, index=fit.index
        )
        months_z = pd.Series(
            pd.PeriodIndex(evaluate.RENT_APPROVAL_DATE, freq="M").asi8,
            index=evaluate.index,
        )
        train_effect, effects, counts, _ = fit_market(
            full_x, full_z, fit.MONTHLY_RENT, months_x, months_z
        )
        future_effect, trend = extrapolate(effects, counts, months_z, "damped")
        trend_rows.append({"fold": fold, **trend})

        x = residual_features(full_x)
        z = residual_features(full_z)
        x, z, categorical = catboost_frames(x, z)
        residual = fit.MONTHLY_RENT - train_effect
        order = months_x.sort_values(kind="stable").index
        started = time.perf_counter()
        model = CatBoostRegressor(**CAT_PARAMS).fit(
            x.loc[order], residual.loc[order], cat_features=categorical
        )
        predicted = future_effect + model.predict(z)
        seconds = time.perf_counter() - started
        if not np.isfinite(predicted).all():
            raise RuntimeError("Non-finite CatBoost trend prediction")

        source = pd.read_csv(
            ROOT
            / "experiments/e6/market_regression/results"
            / f"{fold}_predictions.csv.gz",
            index_col="listing_index",
        )
        np.testing.assert_allclose(
            source.loc[evaluate.index, "actual"], evaluate.MONTHLY_RENT
        )
        predictions = pd.DataFrame({"actual": evaluate.MONTHLY_RENT})
        for public_name, source_name in COMPARATORS.items():
            predictions[public_name] = source.loc[evaluate.index, source_name]
        predictions[ENSEMBLE] = 0.5 * (
            predictions["B_CatBoost"] + predictions["Trend_damped_LightGBM"]
        )
        predictions[NEW_MODEL] = predicted

        for name in [*COMPARATORS, ENSEMBLE, NEW_MODEL]:
            extra = {}
            if name == NEW_MODEL:
                extra = {
                    "fit_seconds": seconds,
                    "residual_train_rmse": float(
                        model.get_evals_result()["learn"]["RMSE"][-1]
                    ),
                }
            rows.append(
                {
                    "fold": fold,
                    "configuration": name,
                    "train_rows": len(fit),
                    "validation_rows": len(evaluate),
                    **extra,
                    **score(predictions.actual, predictions[name]),
                }
            )
            errors = predictions[name] - predictions.actual
            dimensions = [
                ("month", evaluate.RENT_APPROVAL_DATE),
                (
                    "rent_band",
                    pd.cut(
                        evaluate.MONTHLY_RENT,
                        [-np.inf, 2000, 3600, np.inf],
                    ).astype(str),
                ),
            ]
            for dimension, labels in dimensions:
                for label, index in labels.groupby(labels).groups.items():
                    grouped.append(
                        {
                            "fold": fold,
                            "configuration": name,
                            "dimension": dimension,
                            "group": str(label),
                            "rows": len(index),
                            "sse_share": float(
                                (errors.loc[index] ** 2).sum() / (errors**2).sum()
                            ),
                            **score(
                                predictions.loc[index, "actual"],
                                predictions.loc[index, name],
                            ),
                        }
                    )
        predictions.to_csv(
            DEST / f"{fold}_predictions.csv.gz", index_label="listing_index"
        )
        pd.DataFrame(rows).to_csv(DEST / "overall.csv", index=False)
        pd.DataFrame(grouped).to_csv(DEST / "group_errors.csv", index=False)
        pd.DataFrame(trend_rows).to_csv(DEST / "trend_parameters.csv", index=False)
        print(pd.DataFrame(rows).tail(4).to_string(index=False), flush=True)
    print("Complete", flush=True)


if __name__ == "__main__":
    main()
