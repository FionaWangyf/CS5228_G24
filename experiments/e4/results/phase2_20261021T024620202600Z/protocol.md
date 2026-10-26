# E4 Phase 2 protocol

Five predeclared temporal configurations; fixed non-temporal inputs: floor_area_sqm,
flat_age, TOWN, FLAT_TYPE. Calendar month 1–12 and quarter 1–4 are one-hot encoded;
year is numeric. Shared cleaning and split: 117777 rows, 2021-01–2024-04 training;
31582 rows, 2024-05–2025-03 validation. All new fits use identical LightGBM parameters,
train-fitted median imputation/scaling and category encoding. No test data, validation
tuning, early stopping, auxiliary inputs or hybrid models. Pipeline fit time includes
preprocessing. Mean Error is prediction minus actual. Positive RMSE/MAE improvement
means lower error than approved E4.0.

E4.0 comparison values and predictions come only from original approved run
 e40_20261021T024110526575Z. Re-running the original notebook section creates a new
baseline run but does not substitute it into this comparison. New artifacts use a
unique UTC run directory. All exact feature lists, parameters, source/baseline hashes,
record fingerprints, versions, metrics, warnings and training checks are in run.json.

Monthly diagnostics are descriptive. Lower overall RMSE does not establish better
extrapolation, causality, statistical significance or performance on an independent
future holdout. Calendar repetition and numeric time saturation require separate
interpretation; E4.4 contains redundant temporal information. No further model or
sensitivity experiment is implemented in this phase.
