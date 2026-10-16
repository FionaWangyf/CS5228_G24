# Model Comparison Results

| RMSE Rank | Model | Family | RMSE | MAE | R2 | Mean Error | Training Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | TabM | Tabular neural network (extended features) | 474.00 | 344.00 | 0.4784 | N/A | ~155 |
| 2 | CatBoost | Gradient-boosted trees | 482.68 | 358.5 | 0.4602 | -43.92 | 4.13 |
| 3 | LightGBM | Gradient-boosted trees | 483.09 | 359.34 | 0.4593 | -44.02 | 0.85 |
| 4 | HistGradientBoosting | Histogram gradient boosting | 484.21 | 360.63 | 0.4568 | -45.59 | 1.14 |
| 5 | Extra Trees | Randomized tree ensemble | 492.94 | 368.43 | 0.4371 | -56.17 | 16.5 |
| 6 | Random Forest | Tree ensemble | 493.0 | 368.47 | 0.4369 | -53.13 | 7.53 |
| 7 | Linear Regression | Linear | 616.28 | 460.27 | 0.1201 | 351.26 | 0.2 |
| 8 | Ridge Regression | Regularized linear | 616.29 | 460.28 | 0.1201 | 351.24 | 0.21 |
| 9 | Dummy Regressor | Constant baseline | 839.61 | 683.05 | -0.6332 | -522.78 | 0.15 |

TabM uses the same fixed train/validation dates but a larger, different feature set than the five-feature baseline models. Its runtime is approximate (about 155 seconds, inferred from the progress log); Mean Error was not included in the supplied output.
