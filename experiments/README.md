# Experiments

For the full experiment narrative, model designs, results, error analysis, and
current recommendation, start with the
[E1–E6 team experiment summary](e6/TEAM_EXPERIMENT_SUMMARY_zh.md).

| Stage | Entry | Purpose |
|---|---|---|
| E1 | [model_comparison.ipynb](model_comparison.ipynb) | Original model comparison; CSV and Markdown exports remain beside the notebook. |
| E2/E3 | [feature_engineering/](feature_engineering/README.md) | Listing features, encoding, auxiliary ablations, diagnostics, and saved results. |
| E4 | [e4/](e4/README.md) | Controlled comparison of temporal representations. |
| E5 | [error_analysis/](error_analysis/README.md) | Historical fixed-origin backtests and subgroup/error analysis. |
| E6 | [e6/](e6/README.md) | Evidence-based integration, follow-up encoding/cleaning experiments, market trend models, and Kaggle submissions. |
| Shared example | [example_baseline.ipynb](example_baseline.ipynb) | Minimal workflow using the shared preprocessing and validation code. |

Detailed E2/E3 interpretation is in
[feature_engineering_results_zh.md](../analysis/feature_engineering_results_zh.md);
pipeline documentation is in
[feature_engineering_zh.md](../analysis/feature_engineering_zh.md). E5 findings
are in [e5_results_zh.md](../analysis/e5_results_zh.md).

Keep stage-specific code and generated artifacts inside their stage directory;
do not add new experiment outputs to this directory's root.
