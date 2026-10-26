# E6 protocol — fixed before new fits

Purpose: integrate E2–E5 findings with three targeted changes, not perform a new
feature/model search. Existing E2/E3/E4/E5 artifacts must remain unchanged.

| ID | Listing and auxiliary inputs | Explicit time representation | Contrast |
|---|---|---|---|
| A | Core + HDB + MRT + Schools + Malls | month_index | Existing D1/E5 Malls reference |
| B | Core + HDB + MRT + Malls | month_index | B vs A removes Schools |
| C | Same non-time inputs as B | numeric year + one-hot calendar_month | C vs B integrates E4.1 |
| D | Core + HDB + Malls | month_index | D vs B removes MRT |

Core means the eight E2.5 listing columns; C replaces its month_index, not flat_age.
All runs use E2/E3 preprocessing: one-hot low-card categories, numeric ordinal
high-card categories, no numeric scaling/imputation (LightGBM handles NaN).
Calendar month is explicitly included in the low-card encoder. All learned
mappings use only each fold's training rows. C tests the E4.1 representation in
this common pipeline, not reproduction of the original E4 preprocessing.

LightGBM: n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42,
n_jobs=-1, verbosity=-1. No tuning, early stopping, blending, target calibration,
trend model or test.csv. Macro features excluded; static geographic snapshots
are retained with the same historical-availability limitation as E3.

Use the existing E5 folds: 2022-06 and 2022-12 origins, each predicting 16 months;
2024-04 origin predicting the original 11-month validation period. Fixed-origin
models never update using forecast-period rental labels. Historical windows
overlap and are retrospective diagnostics; do not pool them as independent CV.

Reuse A predictions with strict index, label, source-hash and metric checks.
Additionally refit A on the formal split to verify the common E6 pipeline against
the saved reference. Fit B/C/D at all three origins. Store predictions and metrics.

Decision principles: examine within-window paired RMSE/MAE and signed error,
monthly consistency and late horizons. Favor simpler models when changes are
small and inconsistent; do not select by one lowest formal RMSE alone. Retain
month_index if C has inconsistent or worse long-horizon behavior. Any additional
combined configuration is permitted only to confirm jointly supported decisions,
not to expand to a full factorial search. Explain judgment where metrics conflict;
no significance threshold or automatic weighted score is invented after results.

Error analysis: monthly/horizon bands, town, flat type, area, training-address
support and true-rent bands; matched-sample changes and squared-error shares.
True-rent groups and oracle monthly offsets are diagnostic only. No row removal
or price correction based on validation labels. Flag groups under 100 rows.
