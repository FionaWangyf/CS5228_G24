# E5 — Validation and Error Analysis

E5 evaluates fixed LightGBM feature configurations without E4, tuning, calibration,
or test.csv. See [Chinese findings](../../analysis/e5_results_zh.md) for interpretation
and the integration recommendation.

Open `e5_validation_error_analysis.ipynb` to inspect saved tables and figures
without training models again.

## Reproduce

From the repository root, using the environment for E2/E3:

```bash
.venv/bin/python -m experiments.error_analysis.e5
.venv/bin/python -m unittest discover -s tests -v
```

Dependencies are the root requirements plus
`experiments/feature_engineering/requirements.txt`. Existing E2, E3 and incremental
prediction CSVs are required for the formal-period analysis. No notebooks need to
be rerun. `results/audit.json` records versions, data hashes, exact feature lists,
model parameters and fold sizes. All pre-existing feature-engineering result and
prediction CSV hashes are checked unchanged at completion.

## Protocol fixed before execution

| Origin | Training | Evaluation | Purpose |
|---|---|---|---|
| 2022-06 | 2021-01–2022-06 | 2022-07–2023-10 (16 months) | Historical robustness check |
| 2022-12 | 2021-01–2022-12 | 2023-01–2024-04 (16 months) | Historical robustness check |
| 2024-04 | 2021-01–2024-04 | 2024-05–2025-03 (11 months) | Reuse official aligned predictions |

The historical checks stay inside the original training period. Each model is
fitted once per origin; rental labels from the forecast period never update it.
The historical validation windows overlap. These are retrospective checks after
inspecting formal results, not independent folds or a new untouched test set.
Differences across origins also reflect training-history length and market regime.

All nine configurations use `month_index`, numeric ordinal high-cardinality
encoding, one-hot low-cardinality encoding and E2/E3's LightGBM parameters. Encoders
are fitted only on each fold's training rows. Existing choice of encoding is held
fixed, not reselected or claimed to have been selected independently of formal
validation.

| Name | Feature set |
|---|---|
| F0 | Five baseline features |
| Core | E2.5 / E3.0 eight listing features |
| HDB | Core + HDB |
| HDB_MRT | Core + HDB + MRT |
| Schools | E3.3: Core + HDB + MRT + Schools |
| Malls | E3.3 + Malls (D1) |
| COE | E3.3 + COE (D2) |
| Stock | E3.3 + Stock (D3) |
| All | E3.4 |

Macro configurations retain a one-calendar-month lag and assume supplied
forecast-period covariates may be used, as provisionally described by the user.
This is a competition-covariate scenario, not strict information availability at
the forecast origin. Geographic snapshots are static. Neither assumption is
resolved by a time split.

## Outputs

- `results/overall.csv`: nine configurations at each origin, four metrics.
- `results/paired_comparisons.csv`: RMSE/MAE changes and number of months improved
  against a specified control. Positive improvement means lower error.
- `results/group_errors.csv`: month, forecast-horizon band, town, flat type, floor
  area, training address support, and actual-rent band. Includes sample share and
  squared-error share; groups below 100 rows are flagged.
- `results/oracle_diagnostics.csv`: hypothetical perfect global/monthly additive
  offsets and error concentration. These use validation labels and are diagnostic
  quantities only, never corrected predictions or deployable performance.
- `predictions/*.csv.gz`: historical predictions with cleaned listing indices and
  actual labels. Formal predictions remain in their original directory.
- `figures/`: fold RMSE, paired feature gains, horizon bias, town sample/error shares.

Area edges are fixed (60, 90, 110 sqm); address support is based on the composite
TOWN/BLOCK/STREET key and training counts (unseen, 1–5, 6–20, >20). Rent edges use
training-label quantiles. Actual-price bins and the largest-error 1% are post-hoc
analysis, not prediction inputs or rules for dropping rows. R² is undefined for
singleton groups. Horizon band 09–12 in the formal period contains only months
9–11; there is no formal 13–16 result.

No final model is selected by pooling these overlapping folds into one score.
This module completes a focused E5, not E6 integration or final Kaggle inference.
