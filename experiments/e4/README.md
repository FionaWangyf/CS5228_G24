# E4: Temporal Feature Representation

Compare five approval-time representations for HDB rental prediction under one
fixed LightGBM model and chronological validation protocol. The goal is to assess
accuracy and temporal generalization, not to tune for the lowest validation score.

## Controlled experiment

All configurations retain numeric `floor_area_sqm`, `flat_age` and categorical
`TOWN`, `FLAT_TYPE`. Only the explicit approval-time inputs change:

| Experiment | Numeric time inputs | Categorical time inputs |
| --- | --- | --- |
| E4.0 | `month_index` | None |
| E4.1 | `year` | `calendar_month` |
| E4.2 | `month_index` | `calendar_month` |
| E4.3 | `year` | `calendar_quarter` |
| E4.4 | `month_index`, `year` | `calendar_month` |

`month_index = (approval_year - 2021) * 12 + approval_month - 1`.
Year, month (1–12), and quarter (1–4) come from `RENT_APPROVAL_DATE`.
Approximate `flat_age = approval_year - LEASE_COMMENCE_DATE`; negative values
become missing without deleting records. Flat age itself contains time information
and is held fixed across configurations. Unused temporal columns are excluded.

Shared functions in `src/` supply cleaning, features, preprocessing and evaluation:
`load_data()`, `make_features()`, `make_preprocessor()`, `evaluate()`.
Labels are normalized and exact duplicates removed before the fixed split:

| Partition | Approval months | Cleaned rows |
| --- | --- | ---: |
| Training | 2021-01–2024-04 | 117,777 |
| Validation | 2024-05–2025-03 | 31,582 |

Each configuration uses a fresh pipeline. Numerical median imputation and scaling,
and categorical constant imputation and one-hot encoding (`drop="first"`,
`handle_unknown="ignore"`), are fitted on training rows only. Scaling is retained
for consistency with the shared baseline, although trees do not require it.
There is no test.csv use, auxiliary integration, target encoding, hyperparameter
tuning or validation-based early stopping.

LightGBM settings: `n_estimators=500`, `learning_rate=0.05`, `num_leaves=31`,
`random_state=42`, `n_jobs=-1`, `verbosity=-1`. Remaining parameters are the
LightGBM 4.6.0 defaults; complete parameter dictionaries are saved in `run.json`.

## Approved results

Errors are in SGD. Mean Error is prediction minus actual; negative means
underprediction. Fit time includes preprocessing and model fitting, not prediction
or export, and is hardware dependent.

| Experiment | RMSE | MAE | R² | Mean Error | Fit time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| E4.0 | 485.3581 | 361.3151 | 0.4571 | -45.0954 | 2.276 |
| E4.1 | 484.5177 | 360.9596 | 0.4589 | -43.3791 | 2.240 |
| E4.2 | 484.6909 | 361.1297 | 0.4586 | -46.9347 | 2.250 |
| E4.3 | 484.9714 | 362.4987 | 0.4579 | -51.9187 | 2.222 |
| E4.4 | 484.6909 | 361.1297 | 0.4586 | -46.9347 | 2.199 |

- E4.1 has the lowest overall RMSE/MAE, but gains are small: RMSE improves by
  0.8404 SGD (0.173%) and MAE by 0.3555 SGD (0.098%) versus E4.0.
- E4.1 improves monthly MAE in 6 of 11 months, but is worse in all three 2025
  validation months. Best overall accuracy does not imply consistent later gains.
- E4.2 and E4.4 have identical saved predictions; adding year supplied no additional
  benefit in this run. E4.4 contains redundant temporal information.
- E4.3 slightly improves RMSE while worsening MAE and mean underprediction.

These are descriptive results on one previously examined holdout, without a
significance test or independent future evaluation. Small gains do not establish
stable seasonality or improved extrapolation. No direct temporal saturation
response diagnostic was performed on these E4 models; E2/E3 diagnostics are not
substitutes. Multithreaded training may produce tiny cross-platform numerical
variation. Calendar effects and changing listing composition cannot be separated
from monthly averages alone.

## Environment and installation

Run all commands below from the **repository root**. Use **Python 3.12.10**.
The notebook requires the project-local `.venv` and enforces these exact versions
when comparing with the preserved approved baseline:

| Package | Version | Installed through |
| --- | --- | --- |
| NumPy | 2.2.4 | Root requirements |
| pandas | 2.2.3 | Root requirements |
| scikit-learn | 1.6.1 | Root requirements |
| LightGBM | 4.6.0 | Root requirements |
| matplotlib | 3.11.1 | E4 requirements |
| ipykernel | 6.29.5 | Root requirements |

Root requirements also pin JupyterLab 4.3.5 and CatBoost 1.2.8 and constrain tqdm;
these are not duplicated in E4 requirements. CatBoost is not used by E4.
Phase 1 checks the four core ML package versions; Phase 2 additionally requires
exact equality of Python and all six recorded package versions above. The root
README's broader Python 3.12 recommendation is not sufficient for these assertions.

First ensure `python3.12 --version` prints `Python 3.12.10`; install/select that
interpreter if necessary. Do not replace an existing project environment blindly.
For a fresh clone without `.venv` (macOS/Linux):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r experiments/e4/requirements-e4.txt
python -m pip check
```

For an existing matching `.venv`, activate it and verify versions rather than
recreating it. To inspect the interpreter and packages:

```bash
python --version
python -c "import sys; print(sys.executable)"
python -m pip show numpy pandas scikit-learn lightgbm matplotlib ipykernel
```

**macOS:** LightGBM's default build uses OpenMP. If importing LightGBM reports a
missing `libomp.dylib`, install the system runtime with Homebrew, then retry:

```bash
brew install libomp
```

This is a system dependency, not a pip requirement. See the
[LightGBM 4.6.0 installation guide](https://lightgbm.readthedocs.io/en/v4.6.0/Installation-Guide.html#macos).
No Homebrew change is needed if LightGBM already imports successfully.

## Run from a fresh kernel

1. Open the repository in VS Code with Python and Jupyter extensions installed.
2. Use **Python: Select Interpreter** to choose the repository's `.venv/bin/python`.
3. Open [the E4 notebook](../e4_temporal_representation.ipynb). In **Select Kernel →
   Python Environments**, select that same interpreter; the display name alone is
   not sufficient. The notebook checks `sys.prefix` against the project `.venv`.
4. Use **Restart Kernel**, then **Run All** in order. The kernel working directory
   must be the repository root or `experiments/`, not `experiments/e4/`.
   Alternatively, start `jupyter lab` from the activated repository-root terminal.
5. Keep the original baseline directory below present. New runs use unique UTC
   directories under `experiments/e4/results/` and never replace approved artifacts.

A full run first refits E4.0 into a **new** directory, then fits E4.1–E4.4.
Phase 2 deliberately compares against the **original approved E4.0**, not the newly
refitted one. It verifies source/data hashes, versions, validation indices and
actual targets before comparison. Changes to shared source or root requirements
cause this guard to stop: do not bypass it to mix different experiment protocols.
No prior kernel state or notebook outside this repository is needed.

A non-fatal LightGBM feature-name warning was captured during the approved runs;
feature alignment and saved metrics were checked independently. A matplotlib cache
permission message may appear in restricted environments; plotting uses a temporary
cache directory and does not require editing teammate files.

## Approved artifacts and submission scope

- [Original E4.0](results/e40_20261021T024110526575Z/): `metrics.csv`, `run.json`,
  `validation_predictions.csv`. **All three must be preserved and included**:
  Phase 2 depends on this exact directory even after a fresh full rerun.
- [Approved Phase 2](results/phase2_20261021T024620202600Z/): `comparison.csv`,
  `run.json`, `protocol.md`, `monthly_metrics.csv`, four `E41`–`E44` validation
  prediction CSVs, `monthly_mae.png`, and `monthly_average_rent.png`.
- Predictions retain `listing_index`, approval month, actual rent, predicted rent
  and signed error. Metadata records exact inputs, parameters, versions, source
  hashes and checks. The monthly table contains 55 experiment–month combinations.
- E4.2/E4.4 curves overlap because their predictions are identical; this should be
  stated when reusing the figures in the report.

The incomplete first attempt `results/phase2_20261021T024544608090Z/` is retained
locally but excluded by [the scoped ignore rule](results/.gitignore). It is not an
approved result. Do not include repeated exploratory runs or environment caches
in the submission. The intended review branch is `e4-temporal-features`; branch
creation, staging and upload are separate actions, not part of these instructions.
