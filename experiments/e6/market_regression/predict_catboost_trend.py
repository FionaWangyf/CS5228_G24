"""Refit the selected damped-trend + CatBoost residual model on all labels.

Run: .venv/bin/python -m experiments.e6.market_regression.predict_catboost_trend
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from experiments.e6.market_regression.catboost_trend import (
    catboost_frames,
    residual_features,
)
from experiments.e6.market_regression.run import (
    CAT_PARAMS,
    address_features,
    extrapolate,
    fit_market,
)
from experiments.e6.predict_test import validate_submission
from experiments.e6.run_e6 import DEST, ROOT, digest
from src.auxiliary import load_auxiliary
from src.cleaning import clean_listings
from src.features import make_auxiliary


NAME = "damped_trend_catboost_no_long_time"


def main():
    output = DEST / "submissions"
    output.mkdir(exist_ok=True)
    raw_train = pd.read_csv(ROOT / "data/train.csv")
    raw_test = pd.read_csv(ROOT / "data/test.csv")
    sample = pd.read_csv(ROOT / "data/example-submission.csv")
    train = clean_listings(raw_train, drop_exact_duplicates=True)
    test = clean_listings(raw_test, drop_exact_duplicates=False)
    if not test.index.equals(raw_test.index) or len(test) != len(raw_test):
        raise RuntimeError("Test order or row count changed during cleaning")

    raw = load_auxiliary()
    auxiliary = make_auxiliary(
        df_hdb=raw["hdb"], df_mrt=raw["mrt"], df_malls=raw["malls"]
    )
    full_x = address_features(train, auxiliary, "C")
    full_z = address_features(test, auxiliary, "C")
    months_x = pd.Series(
        pd.PeriodIndex(train.RENT_APPROVAL_DATE, freq="M").asi8,
        index=train.index,
    )
    months_z = pd.Series(
        pd.PeriodIndex(test.RENT_APPROVAL_DATE, freq="M").asi8,
        index=test.index,
    )
    train_effect, effects, counts, _ = fit_market(
        full_x, full_z, train.MONTHLY_RENT, months_x, months_z
    )
    future_effect, trend = extrapolate(effects, counts, months_z, "damped")

    x = residual_features(full_x)
    z = residual_features(full_z)
    if "year" in x or "month_index" in x:
        raise RuntimeError("Long-term time feature reached CatBoost")
    x, z, categorical = catboost_frames(x, z)
    residual = train.MONTHLY_RENT - train_effect
    order = months_x.sort_values(kind="stable").index
    model = CatBoostRegressor(**CAT_PARAMS).fit(
        x.loc[order], residual.loc[order], cat_features=categorical
    )
    predictions = future_effect + model.predict(z)
    submission = pd.DataFrame(
        {"Id": sample.Id.copy(), "Predicted": predictions}
    )
    validate_submission(submission, sample, test)

    path = output / f"submission_{NAME}_fulltrain.csv"
    submission.to_csv(path, index=False)
    reread = pd.read_csv(path)
    validate_submission(reread, sample, test)
    np.testing.assert_allclose(reread.Predicted, predictions, rtol=0, atol=1e-9)

    monthly = pd.DataFrame(
        {
            "month": test.RENT_APPROVAL_DATE,
            "market_effect": future_effect,
            "predicted": predictions,
        }
    ).groupby("month").agg(
        rows=("predicted", "size"),
        market_effect=("market_effect", "first"),
        prediction_mean=("predicted", "mean"),
        prediction_std=("predicted", "std"),
        prediction_min=("predicted", "min"),
        prediction_max=("predicted", "max"),
    )
    monthly_path = output / f"{NAME}_test_monthly_predictions.csv"
    monthly.to_csv(monthly_path)

    sources = [
        Path(__file__),
        Path(__file__).with_name("catboost_trend.py"),
        Path(__file__).with_name("run.py"),
        ROOT / "src/cleaning.py",
        ROOT / "src/features.py",
        ROOT / "src/auxiliary.py",
    ]
    inputs = [
        ROOT / "data/train.csv",
        ROOT / "data/test.csv",
        ROOT / "data/example-submission.csv",
        *sorted((ROOT / "data/auxiliary").glob("*.csv")),
    ]
    audit = {
        "model": NAME,
        "selection": "Lowest formal validation RMSE among the requested controlled comparison; historical windows retained as robustness evidence.",
        "train_rows": len(train),
        "raw_train_rows": len(raw_train),
        "test_rows": len(test),
        "train_period": [train.RENT_APPROVAL_DATE.min(), train.RENT_APPROVAL_DATE.max()],
        "test_period": [test.RENT_APPROVAL_DATE.min(), test.RENT_APPROVAL_DATE.max()],
        "test_deduplicated": False,
        "residual_features": list(x.columns),
        "categorical_features": categorical,
        "excluded_long_term_residual_features": ["month_index", "year"],
        "retained_seasonal_feature": "calendar_month",
        "trend": trend,
        "catboost_params": CAT_PARAMS,
        "residual_training_rmse": float(
            model.get_evals_result()["learn"]["RMSE"][-1]
        ),
        "prediction_summary": pd.Series(predictions).describe().to_dict(),
        "checks": {
            "rows_and_ids_match_sample": True,
            "all_predictions_finite": True,
            "test_order_preserved": True,
            "training_only_fit": True,
        },
        "input_hashes": {
            str(item.relative_to(ROOT)): digest(item) for item in inputs
        },
        "source_hashes": {
            str(item.relative_to(ROOT)): digest(item) for item in sources
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in [
                "numpy",
                "pandas",
                "scikit-learn",
                "catboost",
            ]
        },
        "submission_sha256": digest(path),
        "uploaded": False,
        "kaggle_score": None,
    }
    audit_path = output / f"{NAME}_submission_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    print(
        json.dumps(
            {
                "submission": str(path),
                "monthly": str(monthly_path),
                "audit": str(audit_path),
                "trend": trend,
                "prediction_summary": audit["prediction_summary"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
