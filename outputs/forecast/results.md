# Backtest results (finalized data)

Origins: 156 weekly origins, 2023-10-01 to 2026-09-20 (expanding window, refit at every origin). Target: national wILI %.

Correlated states (chosen on data before 2023-10-01): il, tn, pa, ca, ga.


## MAE

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| seasonal_naive | 0.707 | 0.708 | 0.707 | 0.701 |
| arima | 0.207 | 0.454 | 0.707 | 0.884 |
| lightgbm | 0.156 | 0.317 | 0.441 | 0.539 |
| xgboost | 0.161 | 0.273 | 0.370 | 0.435 |
| lightgbm+states | 0.168 | 0.316 | 0.424 | 0.484 |
| xgboost+states | 0.175 | 0.292 | 0.370 | 0.436 |

## RMSE

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| seasonal_naive | 1.091 | 1.094 | 1.095 | 1.091 |
| arima | 0.428 | 0.950 | 1.364 | 1.593 |
| lightgbm | 0.256 | 0.541 | 0.725 | 0.828 |
| xgboost | 0.269 | 0.445 | 0.596 | 0.662 |
| lightgbm+states | 0.302 | 0.597 | 0.723 | 0.801 |
| xgboost+states | 0.320 | 0.528 | 0.660 | 0.763 |

## MAPE (%)

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| seasonal_naive | 22.1 | 22.0 | 22.0 | 21.7 |
| arima | 5.7 | 12.1 | 19.5 | 25.4 |
| lightgbm | 4.8 | 9.3 | 13.7 | 16.9 |
| xgboost | 4.8 | 8.5 | 11.5 | 14.1 |
| lightgbm+states | 4.9 | 8.8 | 12.3 | 14.6 |
| xgboost+states | 5.0 | 8.3 | 10.8 | 13.3 |

## MAE skill vs seasonal naive (% reduction)

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| seasonal_naive | 0.0 | 0.0 | 0.0 | 0.0 |
| arima | 70.7 | 35.9 | 0.0 | -26.1 |
| lightgbm | 77.9 | 55.2 | 37.6 | 23.1 |
| xgboost | 77.2 | 61.4 | 47.7 | 37.9 |
| lightgbm+states | 76.2 | 55.4 | 40.0 | 31.0 |
| xgboost+states | 75.2 | 58.8 | 47.7 | 37.8 |

## MAE in high-activity weeks (actual wILI ≥ 4.0%, n=37 per horizon)

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| seasonal_naive | 1.652 | 1.652 | 1.652 | 1.652 |
| arima | 0.588 | 1.346 | 1.996 | 2.368 |
| lightgbm | 0.368 | 0.769 | 0.970 | 1.125 |
| xgboost | 0.404 | 0.610 | 0.832 | 0.910 |
| lightgbm+states | 0.452 | 0.824 | 1.048 | 1.134 |
| xgboost+states | 0.485 | 0.792 | 0.940 | 1.047 |

## MAE in other weeks (actual wILI < 4.0%)

| model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| seasonal_naive | 0.414 | 0.411 | 0.408 | 0.397 |
| arima | 0.088 | 0.174 | 0.299 | 0.410 |
| lightgbm | 0.090 | 0.175 | 0.274 | 0.352 |
| xgboost | 0.086 | 0.167 | 0.223 | 0.284 |
| lightgbm+states | 0.080 | 0.157 | 0.227 | 0.277 |
| xgboost+states | 0.079 | 0.136 | 0.190 | 0.241 |

## Real-time (as-of) data vs finalized data

First-reported national wILI differed from the final value by a mean absolute 0.024 points (max 0.142; final minus first-reported: mean -0.009) over 156 origin weeks. 6 origins had no published issue in Delphi and fell back to finalized data.

MAE when models see data as published at each origin vs finalized data (national-only features; scored against finalized values):

- **seasonal_naive**: h=1: 0.707 → 0.707 (+0.0%); h=2: 0.708 → 0.708 (+0.0%); h=3: 0.707 → 0.707 (+0.0%); h=4: 0.701 → 0.701 (+0.0%)
- **arima**: h=1: 0.207 → 0.207 (+0.0%); h=2: 0.454 → 0.453 (-0.2%); h=3: 0.707 → 0.705 (-0.3%); h=4: 0.884 → 0.879 (-0.6%)
- **lightgbm**: h=1: 0.156 → 0.160 (+2.6%); h=2: 0.317 → 0.311 (-1.9%); h=3: 0.441 → 0.436 (-1.1%); h=4: 0.539 → 0.532 (-1.3%)
- **xgboost**: h=1: 0.161 → 0.165 (+2.5%); h=2: 0.273 → 0.270 (-1.1%); h=3: 0.370 → 0.359 (-3.0%); h=4: 0.435 → 0.432 (-0.7%)
