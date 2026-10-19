"""Forecasting models behind one interface: forecast(y_hist, h) -> point forecast of y[T+h].

`y_hist` ends at the forecast origin T. Tree models are trained on rows built from y_hist only.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from lab.forecast.features import TARGET, feature_columns, make_features


def seasonal_naive(y_hist: pd.Series, h: int) -> float:
    """Same week last year: y[T+h-52]."""
    return float(y_hist.iloc[-53 + h])


class ArimaModel:
    """ARIMA on log(ILI) over a trailing window; one fit per origin serves all horizons."""

    def __init__(self, order: tuple[int, int, int], window_weeks: int):
        self.order, self.window = tuple(order), window_weeks
        self._cache: tuple[pd.Timestamp, np.ndarray] | None = None

    def forecast(self, y_hist: pd.Series, h: int, max_h: int = 4) -> float:
        origin = y_hist.index[-1]
        if self._cache is None or self._cache[0] != origin:
            endog = np.log(y_hist.iloc[-self.window:].to_numpy())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(endog, order=self.order).fit()
            self._cache = (origin, np.exp(fit.forecast(max_h)))
        return float(self._cache[1][h - 1])


def make_regressor(kind: str, params: dict):
    if kind == "lightgbm":
        return LGBMRegressor(**params)
    if kind == "xgboost":
        return XGBRegressor(**params)
    raise ValueError(kind)


def training_frame(
    y_hist: pd.Series, h: int, extra: pd.DataFrame | None, drop_mask: pd.Series | None
) -> tuple[pd.DataFrame, pd.Series]:
    """(all feature rows, rows usable for training). Training rows need an observed target."""
    frame = make_features(y_hist, h, extra)
    train = frame.dropna(subset=[TARGET])
    if drop_mask is not None:
        # Drop rows whose origin or target week falls in an excluded (e.g. COVID) season.
        bad = drop_mask.reindex(frame.index).fillna(False)
        bad_target = bad.shift(-h).fillna(False).astype(bool)
        train = train.loc[~(bad | bad_target).reindex(train.index)]
    origin = y_hist.index[-1]
    # Leakage guard: every training label must be observed by the origin.
    assert train.index.max() + pd.Timedelta(weeks=h) <= origin, "training label after origin"
    return frame, train


def fit_tree(kind: str, params: dict, train: pd.DataFrame):
    model = make_regressor(kind, params)
    model.fit(train[feature_columns(train)], train[TARGET])
    return model


def predict_tree(model, frame: pd.DataFrame, y_hist: pd.Series) -> float:
    origin = y_hist.index[-1]
    x = frame.loc[[origin], feature_columns(frame)]
    return float(y_hist.iloc[-1] * np.exp(model.predict(x)[0]))
