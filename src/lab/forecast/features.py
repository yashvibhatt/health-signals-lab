"""Feature matrix for direct h-step-ahead forecasting of weekly ILI.

Row t uses ONLY values observed at or before week t. The target for row t is
log(y[t+h] / y[t]) - the log growth over the next h weeks - so tree models are not
capped at the highest ILI level seen in training. Prediction: y_hat = y[t] * exp(target_hat).

Index must be a regular weekly DatetimeIndex (one row per week, no gaps).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.data.delphi import date_to_epiweek

TARGET = "target"


def make_features(y: pd.Series, h: int, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features for origin rows t (plus the target column, NaN where t+h is not yet observed)."""
    f = pd.DataFrame(index=y.index)
    for k in range(4):
        f[f"lag_{k}"] = y.shift(k)                       # y[t-k]
    f["diff_1"] = y - y.shift(1)
    for w in (4, 8):
        f[f"roll_mean_{w}"] = y.rolling(w).mean()        # window ends at t (inclusive)
        f[f"roll_std_{w}"] = y.rolling(w).std()
    f["ratio_to_roll_mean_8"] = y / f["roll_mean_8"]
    f["last_year_target_week"] = y.shift(52 - h)         # y[t+h-52], observed by t since h <= 52

    # Seasonality of the *target* week (a calendar fact, known in advance).
    target_week = [date_to_epiweek((d + pd.Timedelta(weeks=h)).date()) % 100 for d in y.index]
    angle = 2 * np.pi * np.array(target_week) / 52.18
    f["target_woy_sin"], f["target_woy_cos"] = np.sin(angle), np.cos(angle)

    if extra is not None:
        for col in extra.columns:
            s = extra[col].reindex(y.index)
            f[f"{col}_lag_0"] = s
            f[f"{col}_lag_1"] = s.shift(1)
            f[f"{col}_diff_1"] = s - s.shift(1)

    f[TARGET] = np.log(y.shift(-h) / y)
    return f


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != TARGET]
