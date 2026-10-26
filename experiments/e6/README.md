# E6 — Evidence-based integration and ablation

Start with [TEAM_EXPERIMENT_SUMMARY_zh.md](TEAM_EXPERIMENT_SUMMARY_zh.md) for the
complete E1–E6 narrative, model designs, results, error analysis, and current
recommendation.

The predeclared [PROTOCOL.md](PROTOCOL.md) defines the original A–D integration
and its decision principles. Later subdirectories preserve follow-up experiments
without changing that original protocol.

## Experiment map

| Entry | Purpose |
|---|---|
| `run_e6.py` | Original A–D LightGBM integration and three-origin evaluation. |
| `e6_results.ipynb` | Inspect the original E6 saved tables and figures without retraining. |
| `followup/clean_similar.py` | Training-only similar-listing outlier flags. |
| `followup/run.py` | Outlier cleaning, complete address, alternative encoding, and historical-rent features. |
| `market_regression/run.py` | CatBoost, time-safe encodings, explicit market trends, and simple ensembles. |
| `market_regression/catboost_trend.py` | Final controlled comparison for damped trend plus CatBoost residuals. |
| `predict_test.py` | Reproduce the earlier E6-C full-train submission. |
| `market_regression/predict_catboost_trend.py` | Reproduce the latest full-train test prediction. |
| `submissions/` | Kaggle-ready CSVs, monthly summaries, leaderboard observation, and audit metadata. |

## Reproduce

Run commands from the repository root. Reproduce only the stage needed; later
stages use the same project environment and saved prerequisites.

```bash
# Original E6 A–D integration
.venv/bin/python -m experiments.e6.run_e6

# Follow-up cleaning/encoding/history experiments
.venv/bin/python -m experiments.e6.followup.run

# CatBoost, trend, and ensemble comparison
.venv/bin/python -m experiments.e6.market_regression.run

# Latest controlled model comparison
.venv/bin/python -m experiments.e6.market_regression.catboost_trend

# Latest full-train Kaggle prediction
.venv/bin/python -m experiments.e6.market_regression.predict_catboost_trend

.venv/bin/python -m unittest discover -s tests -v
```

Dependencies are the root requirements plus feature-engineering plotting
requirements. Actual runtime versions and data/source hashes are in
`results/audit.json`. Original E4's Python/version/source assertions are not used:
C implements E4.1's representation inside the E2/E3 pipeline. It is explicitly
not a rerun of the original E4 notebook.

Required inputs include raw training/auxiliary data, existing E3/D1 predictions
and audit files, and E5's historical predictions, metrics and audit. To rebuild
these prerequisites, follow `experiments/feature_engineering/README.md`, then run
`.venv/bin/python -m experiments.error_analysis.e5`. E6 does not modify them.

The original E6 runner trains B/C/D at three origins and verifies A by refitting
it on the formal split. It does not use `test.csv`. Only the two prediction
scripts under this directory read `test.csv`; they fit preprocessing and models
using labeled training data and preserve test order.

## Artifact map

| Artifact | Purpose |
|---|---|
| `e6_results.ipynb` | View saved tables/figures without retraining |
| `results/overall.csv` | Four metrics for each configuration and origin |
| `results/paired_errors.csv` | Same-row comparisons, overall/month/horizon/subgroups |
| `results/monthly_consistency.csv` | Counts of months with lower RMSE/MAE |
| `results/group_errors.csv` | Group sizes, errors, sample and squared-error shares |
| `results/oracle_diagnostics.csv` | Diagnostic monthly-offset headroom, tail-error concentration |
| `results/audit.json` | Configurations, exact feature lists, hashes, versions, checks |
| `results/selected_configuration.json` | Documented final choice and its scope |
| `predictions/*.csv.gz` | Index-aligned actual values and A/B/C/D predictions |
| `figures/*.png` | Paired gains, monthly errors and horizon bias |
| `followup/results/` | Cleaning, complete-address, encoding, and historical-feature comparisons |
| `market_regression/results/` | CatBoost, trend, and ensemble comparison artifacts |
| `market_regression/catboost_trend_results/` | Latest controlled model metrics, predictions, groups, and trend parameters |
| `submissions/` | Submission CSVs and reproducibility audits |

For paired metrics, **positive RMSE/MAE/MSE/SSE improvement means the candidate
is better**. `mean_error_change` is candidate bias minus control bias; it is not
an improvement score. `abs_bias_improvement` is positive when absolute bias falls.
Fraction of rows improved counts strictly lower absolute error; ties are not
wins. SSE changes sum across disjoint groups; RMSE differences do not.

Historical evaluation windows overlap. Each origin has different training
length and market regime, so compare candidates within each origin, not raw RMSE
between origins as a measure of algorithm improvement. The formal period was
previously inspected; no independent-test or significance claim is made.

Groups below 100 rows are flagged. The formal 09–12 horizon group contains only
months 9–11. Actual-rent bins use training quantiles but classify validation rows
by their actual labels; these are explanatory diagnostics, never usable inference
features or rules for deleting outliers. Oracle scores use validation labels and
are not deployable performance. Static auxiliary geography is not historically
reconstructed. No claim of causal effects is supported.
