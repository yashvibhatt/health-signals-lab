# Backtest results (finalized data)

Origins: 156 weekly origins, 2023-10-01 to 2026-09-20 (expanding window, refit at every origin). Target: national wILI %.

Correlated states (chosen on data before 2023-10-01): il, tn, pa, ca, ga.


## MAE

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| persistence | 0.282 | 0.514 | 0.699 | 0.862 |
| seasonal_naive | 0.707 | 0.708 | 0.707 | 0.701 |
| arima | 0.207 | 0.454 | 0.707 | 0.884 |
| lightgbm | 0.156 | 0.317 | 0.441 | 0.539 |
| xgboost | 0.161 | 0.273 | 0.370 | 0.435 |
| lightgbm+states | 0.168 | 0.316 | 0.424 | 0.484 |
| xgboost+states | 0.175 | 0.292 | 0.370 | 0.436 |

## RMSE

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| persistence | 0.481 | 0.845 | 1.091 | 1.264 |
| seasonal_naive | 1.091 | 1.094 | 1.095 | 1.091 |
| arima | 0.428 | 0.950 | 1.364 | 1.593 |
| lightgbm | 0.256 | 0.541 | 0.725 | 0.828 |
| xgboost | 0.269 | 0.445 | 0.596 | 0.662 |
| lightgbm+states | 0.302 | 0.597 | 0.723 | 0.801 |
| xgboost+states | 0.320 | 0.528 | 0.660 | 0.763 |

## MAPE (%)

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| persistence | 8.1 | 15.3 | 21.7 | 27.9 |
| seasonal_naive | 22.1 | 22.0 | 22.0 | 21.7 |
| arima | 5.7 | 12.1 | 19.5 | 25.4 |
| lightgbm | 4.8 | 9.3 | 13.7 | 16.9 |
| xgboost | 4.8 | 8.5 | 11.5 | 14.1 |
| lightgbm+states | 4.9 | 8.8 | 12.3 | 14.6 |
| xgboost+states | 5.0 | 8.3 | 10.8 | 13.3 |

## MAE skill vs persistence (no change from origin week) (% reduction; positive = better)

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| persistence | 0.0 | 0.0 | 0.0 | 0.0 |
| seasonal_naive | -150.7 | -37.7 | -1.1 | 18.7 |
| arima | 26.6 | 11.7 | -1.1 | -2.6 |
| lightgbm | 44.7 | 38.3 | 36.9 | 37.5 |
| xgboost | 42.9 | 46.9 | 47.1 | 49.5 |
| lightgbm+states | 40.4 | 38.5 | 39.3 | 43.9 |
| xgboost+states | 37.9 | 43.2 | 47.1 | 49.4 |

## MAE skill vs seasonal naive (% reduction; positive = better)

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| persistence | 60.1 | 27.4 | 1.1 | -23.0 |
| seasonal_naive | 0.0 | 0.0 | 0.0 | 0.0 |
| arima | 70.7 | 35.9 | 0.0 | -26.1 |
| lightgbm | 77.9 | 55.2 | 37.6 | 23.1 |
| xgboost | 77.2 | 61.4 | 47.7 | 37.9 |
| lightgbm+states | 76.2 | 55.4 | 40.0 | 31.0 |
| xgboost+states | 75.2 | 58.8 | 47.7 | 37.8 |

## MAE skill vs ARIMA (% reduction; positive = better)

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| persistence | -36.2 | -13.2 | 1.1 | 2.5 |
| seasonal_naive | -241.5 | -55.9 | 0.0 | 20.7 |
| arima | 0.0 | 0.0 | 0.0 | 0.0 |
| lightgbm | 24.6 | 30.2 | 37.6 | 39.0 |
| xgboost | 22.2 | 39.9 | 47.7 | 50.8 |
| lightgbm+states | 18.8 | 30.4 | 40.0 | 45.2 |
| xgboost+states | 15.5 | 35.7 | 47.7 | 50.7 |

## MAE in high-activity weeks (actual wILI ≥ 4.0%, n=37 per horizon)

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| persistence | 0.717 | 1.235 | 1.545 | 1.777 |
| seasonal_naive | 1.652 | 1.652 | 1.652 | 1.652 |
| arima | 0.588 | 1.346 | 1.996 | 2.368 |
| lightgbm | 0.368 | 0.769 | 0.970 | 1.125 |
| xgboost | 0.404 | 0.610 | 0.832 | 0.910 |
| lightgbm+states | 0.452 | 0.824 | 1.048 | 1.134 |
| xgboost+states | 0.485 | 0.792 | 0.940 | 1.047 |

## MAE in other weeks (actual wILI < 4.0%)

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| persistence | 0.146 | 0.288 | 0.431 | 0.570 |
| seasonal_naive | 0.414 | 0.411 | 0.408 | 0.397 |
| arima | 0.088 | 0.174 | 0.299 | 0.410 |
| lightgbm | 0.090 | 0.175 | 0.274 | 0.352 |
| xgboost | 0.086 | 0.167 | 0.223 | 0.284 |
| lightgbm+states | 0.080 | 0.157 | 0.227 | 0.277 |
| xgboost+states | 0.079 | 0.136 | 0.190 | 0.241 |

## Model selection (target weeks before 2025-09-28) and untouched holdout (origins from 2025-09-28)

Selection period: 103 origins. Holdout: 52 origins.

Selection-period MAE:

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| persistence | 0.269 | 0.503 | 0.682 | 0.845 |
| seasonal_naive | 0.781 | 0.782 | 0.782 | 0.773 |
| arima | 0.195 | 0.425 | 0.678 | 0.848 |
| lightgbm | 0.157 | 0.299 | 0.362 | 0.483 |
| xgboost | 0.152 | 0.252 | 0.327 | 0.390 |
| lightgbm+states | 0.158 | 0.288 | 0.356 | 0.423 |
| xgboost+states | 0.162 | 0.273 | 0.326 | 0.394 |

Holdout MAE:

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| persistence | 0.311 | 0.553 | 0.770 | 0.960 |
| seasonal_naive | 0.570 | 0.574 | 0.578 | 0.580 |
| arima | 0.232 | 0.527 | 0.804 | 1.023 |
| lightgbm | 0.156 | 0.358 | 0.609 | 0.666 |
| xgboost | 0.181 | 0.320 | 0.462 | 0.534 |
| lightgbm+states | 0.191 | 0.380 | 0.573 | 0.615 |
| xgboost+states | 0.204 | 0.336 | 0.469 | 0.531 |

Rank by mean MAE over horizons (1 = best): persistence: selection 6, holdout 7; seasonal_naive: selection 7, holdout 5; arima: selection 5, holdout 6; lightgbm: selection 4, holdout 4; xgboost: selection 1, holdout 1; lightgbm+states: selection 3, holdout 3; xgboost+states: selection 2, holdout 2.

Selected on the selection period: **xgboost** (mean MAE 0.280). On the holdout it ranks 1 of 7 (mean MAE 0.374; best on holdout: xgboost, 0.374).

## Real-time (as-of) data vs finalized data

First-reported national wILI differed from the final value by a mean absolute 0.024 points (max 0.142; final minus first-reported: mean -0.009) over the 150 origin weeks with a published issue. The other 6 origins had no issue in Delphi and used finalized data.

MAE when models see data as published at each origin vs finalized data (national-only features; scored against finalized values):

- **persistence**: h=1: 0.282 → 0.284 (+0.7%); h=2: 0.514 → 0.516 (+0.4%); h=3: 0.699 → 0.700 (+0.1%); h=4: 0.862 → 0.864 (+0.2%)
- **seasonal_naive**: h=1: 0.707 → 0.707 (+0.0%); h=2: 0.708 → 0.708 (+0.0%); h=3: 0.707 → 0.707 (+0.0%); h=4: 0.701 → 0.701 (+0.0%)
- **arima**: h=1: 0.207 → 0.207 (+0.0%); h=2: 0.454 → 0.453 (-0.2%); h=3: 0.707 → 0.705 (-0.3%); h=4: 0.884 → 0.879 (-0.6%)
- **lightgbm**: h=1: 0.156 → 0.160 (+2.6%); h=2: 0.317 → 0.311 (-1.9%); h=3: 0.441 → 0.436 (-1.1%); h=4: 0.539 → 0.532 (-1.3%)
- **xgboost**: h=1: 0.161 → 0.165 (+2.5%); h=2: 0.273 → 0.270 (-1.1%); h=3: 0.370 → 0.359 (-3.0%); h=4: 0.435 → 0.432 (-0.7%)
