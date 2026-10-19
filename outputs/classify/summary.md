# Literature extraction summary

Model: `claude-haiku-4-5-20251001` · prompt `v1` · 300 abstracts · 300 extracted (0 needed a retry) · 0 failed after 3 attempts.

Tokens for the full extraction: 413,798 input + 33,284 output ≈ $0.58 at $1.0/$5.0 per M tokens. This run made 0 new API calls (the rest came from the disk cache).

## Study type by publication year

| study_type | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Total |
|---|---|---|---|---|---|---|---|
| clinical | 0 | 2 | 1 | 4 | 2 | 0 | 9 |
| forecasting | 2 | 15 | 11 | 19 | 26 | 16 | 89 |
| other | 1 | 3 | 1 | 1 | 7 | 2 | 15 |
| review | 0 | 4 | 7 | 5 | 7 | 3 | 26 |
| surveillance | 2 | 29 | 28 | 37 | 54 | 11 | 161 |
| Total | 5 | 53 | 48 | 66 | 96 | 32 | 300 |

## Abstracts using each method family, by publication year

(An abstract counts once per family; families come from regex mapping of the extracted method names in `lab/classify/methods.py`; columns can sum to more than the number of abstracts. 204 of 300 abstracts name at least one method; 96 name none.)

| family | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Total |
|---|---|---|---|---|---|---|---|
| Regression (GLM/GAM/linear) | 0 | 7 | 6 | 9 | 22 | 5 | 49 |
| Other time-series methods | 0 | 6 | 4 | 8 | 8 | 2 | 28 |
| LSTM / RNN | 0 | 3 | 6 | 3 | 4 | 7 | 23 |
| ARIMA family | 0 | 3 | 2 | 6 | 8 | 2 | 21 |
| Lab / genomic methods | 0 | 1 | 3 | 3 | 11 | 2 | 20 |
| Ensemble | 1 | 3 | 2 | 4 | 7 | 1 | 18 |
| Other neural network | 0 | 0 | 3 | 6 | 0 | 6 | 15 |
| Transformer | 0 | 1 | 1 | 2 | 3 | 6 | 13 |
| Statistical tests / descriptive | 0 | 1 | 2 | 2 | 4 | 2 | 11 |
| Bayesian model | 0 | 2 | 1 | 0 | 2 | 6 | 11 |
| Other machine learning | 0 | 1 | 0 | 3 | 3 | 3 | 10 |
| Compartmental (SIR/SEIR) | 1 | 3 | 0 | 3 | 2 | 1 | 10 |
| Spatial analysis | 0 | 1 | 1 | 3 | 2 | 1 | 8 |
| Random forest | 0 | 0 | 2 | 1 | 2 | 1 | 6 |
| Exponential smoothing | 0 | 0 | 1 | 1 | 2 | 0 | 4 |
| XGBoost | 0 | 0 | 0 | 3 | 1 | 0 | 4 |
| Other gradient boosting | 0 | 0 | 1 | 0 | 2 | 0 | 3 |
| Prophet | 0 | 0 | 0 | 1 | 1 | 0 | 2 |
| Other | 3 | 14 | 11 | 15 | 30 | 14 | 87 |
