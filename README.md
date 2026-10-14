# CS5228 HDB Rental Price Prediction

This repository has a small shared experiment framework for comparable model
results. It uses one cleaning policy, five baseline features, one fixed temporal
split, and one set of evaluation metrics. The raw CSV files and the original
notebooks are not modified.

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

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.evaluation import evaluate
from src.features import make_features, make_preprocessor
from src.validation import load_data

train_df, valid_df = load_data()
X_train = make_features(train_df)
X_valid = make_features(valid_df)
y_train = train_df["MONTHLY_RENT"]
y_valid = valid_df["MONTHLY_RENT"]

experiment = Pipeline([
    ("preprocess", make_preprocessor()),
    ("model", LinearRegression()),
])
experiment.fit(X_train, y_train)
predictions = experiment.predict(X_valid)
metrics = evaluate(y_valid, predictions)
```

Putting preprocessing and the model in one Pipeline is important: `fit` learns
the numeric medians, scaling statistics, and one-hot categories from training
rows only. Validation rows are only passed to `predict`/`transform`.

`make_features` is deterministic and always returns exactly:

1. `floor_area_sqm`
2. `flat_age`
3. `month_index`
4. `TOWN`
5. `FLAT_TYPE`

For a model comparison, replace only the `model` step. For feature experiments,
start with `make_features`, add columns in the experiment notebook, and pass the
extended numeric/categorical column lists to `make_preprocessor`; do not change
the shared baseline definition.

## Fixed experiment policy

- Training: 2021-01 through 2024-04.
- Validation: 2024-05 through 2025-03.
- The split uses `RENT_APPROVAL_DATE`, never random sampling.
- All 150,000 training CSV rows are retained in original order and with original
  index labels. Exact duplicates are intentionally kept because flat unit numbers
  are unavailable.
- `test.csv` is not loaded by `load_data` and is not used for training or
  validation.
- Metrics are RMSE, MAE, R², and Mean Error (`prediction - actual`).

## Difference from the original baseline

The original baseline reads raw files from Colab's `/content`, uppercases its two
categorical features, and derives its cutoff from the last 20% of observed
months. The shared framework instead applies the listing rules from
`cleaning.ipynb` (lowercase/trim/collapse whitespace and normalized flat type),
drops the constant `FURNISHED` and `FEE` columns, keeps duplicates, uses explicit
date boundaries, treats negative `flat_age` as missing, and reports Mean Error in
addition to the original three metrics.

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
from src.features import make_features, make_preprocessor
from src.validation import load_data

train_df, valid_df = load_data()

X_train = make_features(train_df)
X_valid = make_features(valid_df)
y_train = train_df["MONTHLY_RENT"]
y_valid = valid_df["MONTHLY_RENT"]
```

This gives every member the same 117,988 training rows and 32,012 validation
rows. Do not load `test.csv`, shuffle the rows, or create another split for these
experiments.

### 3. Model comparison experiments

Keep the shared features and preprocessor unchanged. Replace only the final
model in the Pipeline:

```python
experiment = Pipeline([
    ("preprocess", make_preprocessor()),
    ("model", YourModel()),
])

experiment.fit(X_train, y_train)
predictions = experiment.predict(X_valid)
metrics = evaluate(y_valid, predictions)
```

Set a fixed `random_state` for models that contain randomness. Record the model
name and all non-default parameters.

### 4. Feature engineering experiments

Do not change the shared `make_features` function. Add experimental columns in
the member's own notebook and pass the extended column lists to
`make_preprocessor`:

```python
X_train = make_features(train_df)
X_valid = make_features(valid_df)

X_train["new_numeric_feature"] = ...
X_valid["new_numeric_feature"] = ...

numeric_features = [
    "floor_area_sqm",
    "flat_age",
    "month_index",
    "new_numeric_feature",
]
categorical_features = ["TOWN", "FLAT_TYPE"]

experiment = Pipeline([
    ("preprocess", make_preprocessor(numeric_features, categorical_features)),
    ("model", YourModel()),
])
```

Deterministic transformations can be applied separately to training and
validation data. Any operation that learns statistics, categories, thresholds,
or mappings must be fitted on training data only.

### 5. Auxiliary-data experiments

Call `load_data` first, then enrich the two partitions separately:

```python
train_df, valid_df = load_data()

train_enriched = add_auxiliary_data(train_df)
valid_enriched = add_auxiliary_data(valid_df)
```

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
print("Model:", experiment.named_steps["model"])
print("Metrics:", evaluate(y_valid, predictions))
```

Before sharing a result, confirm that:

- training and validation indices still match their targets;
- preprocessing was fitted through `experiment.fit(X_train, y_train)` only;
- all predictions and all four metric values are finite;
- the notebook records the model settings and any additional features;
- the shared files under `src/` were not changed solely for one experiment.
