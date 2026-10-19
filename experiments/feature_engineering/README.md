# Feature Engineering Experiments

Start with the [final analysis](../../analysis/feature_engineering_results_zh.md). Existing results are saved; reading them does not require training models.

```text
feature_engineering/
├── encoding_comparison.ipynb       # E2.5 encoding selection
├── e2_listing_features.ipynb       # E2.0–E2.5
├── e3_auxiliary_features.ipynb      # E3.0–E3.4; macro lag = 1 month
├── diagnostics/                    # Three focused diagnostic notebooks
├── utils/                          # core.py, auxiliary.py, diagnostics.py
├── results/                        # Encoding/E2/E3 CSVs
│   └── diagnostics/                # Monthly errors, D1–D3, feature profiles
├── predictions/                    # Index-aligned validation predictions
├── figures/                        # Report and diagnostic figures
├── audits/                         # Parameters/selection, data alignment, missingness
└── requirements.txt                # Extra notebook/plotting dependencies
```

## Results to reuse

| Question | Result |
|---|---|
| Which encoding was selected? | [encoding_comparison.csv](results/encoding_comparison.csv), [selection](audits/selected_encoding.json) |
| What did listing features add? | [e2_results.csv](results/e2_results.csv) |
| What did auxiliary groups add? | [e3_results.csv](results/e3_results.csv) |
| What did Malls/COE/Stock add separately? | [incremental_results.csv](results/diagnostics/incremental_results.csv) |
| How did error change by month? | [monthly_errors.csv](results/diagnostics/monthly_errors.csv) |
| Is Native categorical actually active? | [encoding_audit.json](audits/encoding_audit.json), [tree split audit](results/diagnostics/native_split_audit.csv) |
| What can month_index do beyond training dates? | [temporal_audit.json](audits/temporal_audit.json), [feature importance](results/diagnostics/e33_feature_importance.csv) |

## Reproduction

Use the existing project environment, or install repository `requirements.txt` and then this directory's `requirements.txt`. Notebooks discover the project root, so they can be opened from their own folders.

Run encoding comparison → E2 → E3 only when reproducing the formal experiments. The three notebooks in `diagnostics/` reuse official results and run only their stated audit/incremental fits. Running them does not change formal CSVs or predictions. Source helpers under `src/` retain their existing interfaces.

Numerical NaNs remain for LightGBM. Low-cardinality categories use One-Hot; high-cardinality categories use numeric Ordinal in formal experiments. Train/validation periods and cleaning are fixed. COE/Stock use previous-month indicators without future filling; geographic tables use static approximations. Exact publication timestamps and historical POI states remain unverified.

Tests: `.venv/bin/python -m unittest discover -s tests -v` from the project root.

## Cleanup policy

Keep source notebooks, essential helper code, canonical result CSVs, aligned predictions, figures, and data/model-behavior audit summaries. Do not keep duplicate copies of result tables, generated report drafts, Python caches, or reconstructible model dumps. The previous E2/E3/diagnostic report drafts have been consolidated into the final analysis; no experiment metric or saved prediction was altered during the reorganization.
