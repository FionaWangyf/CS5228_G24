# CS5228 HDB Rental Price Prediction

## Start here

| Need | Entry |
|---|---|
| Understand the complete experiment story and current conclusion | [E1–E6 team experiment summary](experiments/e6/TEAM_EXPERIMENT_SUMMARY_zh.md) |
| Find an experiment or reproduction command | [Experiment index](experiments/README.md) |
| Inspect E2/E3 feature engineering | [Feature-engineering workspace](experiments/feature_engineering/README.md) and [results analysis](analysis/feature_engineering_results_zh.md) |
| Inspect E5 validation and error analysis | [E5 workspace](experiments/error_analysis/README.md) and [findings](analysis/e5_results_zh.md) |
| Inspect E6 integration, follow-ups, trend models, and submissions | [E6 workspace](experiments/e6/README.md) |

The current best local single model is the damped market trend plus CatBoost
residual model documented in the team summary. The latest Kaggle-ready file is
`experiments/e6/submissions/submission_damped_trend_catboost_no_long_time_fulltrain.csv`.

This repository has a small shared experiment framework for comparable model
results. It uses one cleaning policy, one shared feature pipeline, one fixed
temporal split, and one set of evaluation metrics. Raw CSV files are not modified.

## Setup

Python 3.12 is recommended. From the repository root on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

The pinned core versions were verified with Python 3.12.9. Open
`experiments/example_baseline.ipynb` after starting Jupyter from the repository
root.

## Shared workflow

The call chain is

```text
load_data -> make_listing -> [load_auxiliary -> make_auxiliary -> attach_auxiliary]
          -> [scale_numerical] -> [onehot_encoding] -> encode_high_card
```

```python
from sklearn.linear_model import LinearRegression

from src.evaluation import evaluate
from src.features import encode_high_card, make_listing, onehot_encoding, scale_numerical
from src.validation import load_data

train_df, valid_df = load_data()
X_train = make_listing(train_df)
X_valid = make_listing(valid_df)
y_train = train_df["MONTHLY_RENT"]
y_valid = valid_df["MONTHLY_RENT"]

X_fit, X_eval = scale_numerical(X_train, X_valid)
X_fit, X_eval = onehot_encoding(X_fit, X_eval)
X_fit, X_eval = encode_high_card(X_fit, X_eval, method="ordinal")

model = LinearRegression().fit(X_fit, y_train)
predictions = model.predict(X_eval)
metrics = evaluate(y_valid, predictions)
```

`make_listing` is deterministic. It turns `RENT_APPROVAL_DATE` and
`LEASE_COMMENCE_DATE` into `month_index` and `flat_age`, and keeps
`FLOOR_AREA_SQM`, `TOWN`, `FLAT_TYPE`, `FLAT_MODEL`, `BLOCK`, and `STREET`.

`scale_numerical`, `onehot_encoding`, `ordinal_encoding`, and `encode_high_card`
learn statistics, so each takes the training frame first and any number of
frames to transform with it (`f(X_train, X_valid, X_test)`). They are fitted on
the first frame only.

- All models: `onehot_encoding` for low-card columns and `encode_high_card` for
  high-card columns.
- Linear / Ridge additionally call `scale_numerical` first; tree and boosting
  models do not need scaling.

`encode_high_card` currently supports `method="ordinal"`; new high-cardinality
encodings are registered in `_HIGH_CARD_ENCODERS` in `src/features.py`.
Auxiliary columns are described in `analysis/feature_engineering_zh.md`.

## Fixed experiment policy

- Training: 2021-01 through 2024-04.
- Validation: 2024-05 through 2025-03.
- The split uses `RENT_APPROVAL_DATE`, never random sampling.
- Listing labels are normalized before exact duplicates are removed, matching
  `scripts/cleaning.ipynb`. This reduces the 150,000 raw rows to 149,359 cleaned
  rows and resets the cleaned index while preserving the order of retained rows.
- `test.csv` is not loaded by `load_data` and is not used for training or
  validation.
- Metrics are RMSE, MAE, R², and Mean Error (`prediction - actual`).

## Difference from the original baseline

The original baseline reads raw files from Colab's `/content`, uppercases its two
categorical features, and derives its cutoff from the last 20% of observed
months. The shared framework instead applies the listing rules from
`cleaning.ipynb` (lowercase/trim/collapse whitespace and normalized flat type),
drops the constant `FURNISHED` and `FEE` columns, removes exact duplicates after
label normalization, uses explicit date boundaries, treats negative `flat_age`
as missing, and reports Mean Error in addition to the original three metrics.

The old `data_transformation.ipynb` tries to execute a missing
`preprocessing.ipynb`. The shared framework does not execute or depend on any
notebook.

## Team experiment steps

### 1. Start from the shared example

Start Jupyter from the repository root, then open
`experiments/example_baseline.ipynb`. Each member should make a separate copy
with a descriptive name, for example:

```text
experiments/
├── example_baseline.ipynb
├── model_comparison_member_a.ipynb
├── model_comparison_member_b.ipynb
├── feature_engineering.ipynb
├── auxiliary_data.ipynb
└── error_analysis.ipynb
```

Do not edit the shared example or the old notebooks for an individual
experiment.

### 2. Use the same loading and feature code

Every experiment should begin with:

```python
from src.evaluation import evaluate
from src.features import make_listing
from src.validation import load_data

train_df, valid_df = load_data()

X_train = make_listing(train_df)
X_valid = make_listing(valid_df)
y_train = train_df["MONTHLY_RENT"]
y_valid = valid_df["MONTHLY_RENT"]
```

This gives every member the same 117,777 training rows and 31,582 validation
rows. Do not load `test.csv`, shuffle the rows, or create another split for these
experiments.

### 3. Model comparison experiments

Keep the shared features and encoders unchanged. Replace only the model
(`experiments/model_comparison.ipynb` is the listing-only reference):

```python
X_fit, X_eval = onehot_encoding(X_train, X_valid)
X_fit, X_eval = encode_high_card(X_fit, X_eval, method="ordinal")

model = YourModel().fit(X_fit, y_train)
predictions = model.predict(X_eval)
metrics = evaluate(y_valid, predictions)
```

Set a fixed `random_state` for models that contain randomness. Record the model
name and all non-default parameters.

### 4. Feature engineering experiments

Do not change the shared `make_listing` function. Add experimental columns in
the member's own notebook and pass them to the encoders explicitly with
`columns=` (columns outside `NUMERIC_FEATURES` / `LOW_CARD_FEATURES` /
`HIGH_CARD_FEATURES` are not picked up by default):

```python
X_train = make_listing(train_df)
X_valid = make_listing(valid_df)

X_train["new_numeric_feature"] = ...
X_valid["new_numeric_feature"] = ...

numeric, low_card, high_card = feature_groups(X_train)
X_fit, X_eval = scale_numerical(
    X_train, X_valid, columns=numeric + ["new_numeric_feature"]
)
```

Deterministic transformations can be applied separately to training and
validation data. Any operation that learns statistics, categories, thresholds,
or mappings must be fitted on training data only.

### 5. Auxiliary-data experiments

`load_auxiliary` reads and cleans the six auxiliary CSVs. `make_auxiliary` turns
them into join-ready tables, and `attach_auxiliary` joins whichever tables are in
the dict it receives, so a table-level ablation is just a smaller dict. The
controlled implementations are under `experiments/feature_engineering/`:

```python
from src.auxiliary import attach_auxiliary, load_auxiliary
from src.features import make_auxiliary, order_features

raw = load_auxiliary()
auxiliary = make_auxiliary(
    df_coe=raw["coe"],
    df_mrt=raw["mrt"],
    df_schools=raw["schools"],
    df_hdb=raw["hdb"],
    df_malls=raw["malls"],
    df_stock=raw["stock"],
)

X_train = order_features(attach_auxiliary(make_listing(train_df), auxiliary))
X_valid = order_features(attach_auxiliary(make_listing(valid_df), auxiliary))
```

MRT, schools, and malls need the coordinates from the HDB table, so they can
only be attached together with `"hdb"`.

Preserve the listing index and row order during joins. Check that each enriched
table has the same number of rows and the same index as its input before fitting
a model. Training-derived aggregate features must be learned on `train_df` and
then applied to `valid_df`.

### 6. Error-analysis experiments

Keep predictions aligned to the validation index:

```python
error_df = valid_df.copy()
error_df["actual"] = y_valid
error_df["predicted"] = predictions
error_df["error"] = predictions - y_valid
error_df["absolute_error"] = error_df["error"].abs()
```

The sign convention is the same as the shared Mean Error metric: positive error
means over-prediction and negative error means under-prediction.

### 7. Minimum experiment record

Each experiment notebook should report:

```python
print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))
print("Features:", list(X_train.columns))
print("Model:", model)
print("Metrics:", evaluate(y_valid, predictions))
```

Before sharing a result, confirm that:

- training and validation indices still match their targets;
- encoders and scalers were fitted on the training frame only (it is the
  first argument);
- all predictions and all four metric values are finite;
- the notebook records the model settings and any additional features;
- the shared files under `src/` were not changed solely for one experiment.
