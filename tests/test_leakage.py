"""No look-ahead: anything after the forecast origin must not affect features, training, or forecasts."""
import numpy as np
import pandas as pd
import pytest

from lab.forecast.backtest import default_specs, run_backtest
from lab.forecast.features import TARGET, feature_columns, make_features
from lab.forecast.models import training_frame
from lab.reports.correlate import select_correlated

CFG = {
    "arima": {"order": [1, 1, 0], "window_weeks": 104},
    "lightgbm": {"n_estimators": 30, "verbose": -1, "random_state": 0},
    "xgboost": {"n_estimators": 30, "random_state": 0},
}


def seasonal_series(n=260, seed=0, name="nat"):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 2 + 1.5 * np.maximum(0, np.sin(2 * np.pi * t / 52.18)) ** 3 + rng.normal(0, 0.05, n)
    return pd.Series(y, index=pd.date_range("2018-09-30", periods=n, freq="7D"), name=name)


def tamper_after(s, origin):
    s = s.copy()
    s.loc[s.index > origin] = 99.0
    return s


@pytest.mark.parametrize("h", [1, 2, 3, 4])
def test_features_at_t_use_only_past(h):
    y = seasonal_series()
    origin = y.index[200]
    a = make_features(y, h).loc[:origin, feature_columns(make_features(y, h))]
    b = make_features(tamper_after(y, origin), h).loc[:origin, feature_columns(make_features(y, h))]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("h", [1, 4])
def test_training_labels_observed_by_origin(h):
    y = seasonal_series()
    origin = y.index[200]
    _, train = training_frame(y.loc[:origin], h, None, None)
    assert train.index.max() + pd.Timedelta(weeks=h) <= origin
    assert train[TARGET].notna().all()


def test_backtest_forecasts_unchanged_by_future_data():
    y = seasonal_series()
    extra = pd.DataFrame({"s1": seasonal_series(seed=1), "s2": seasonal_series(seed=2)})
    origins = list(y.index[180:184])
    specs = default_specs(with_extra=True)
    clean = run_backtest(y, origins, [1, 4], specs, CFG, extra_final=extra)
    # Corrupt everything after the LAST origin's history; truth changes, forecasts must not.
    last = origins[-1]
    y2, extra2 = tamper_after(y, last), extra.apply(lambda s: tamper_after(s, last))
    dirty = run_backtest(y2, origins, [1, 4], specs, CFG, extra_final=extra2)
    pd.testing.assert_series_equal(clean["y_pred"], dirty["y_pred"])


def test_correlated_state_selection_ignores_backtest_period():
    wide = pd.DataFrame({"nat": seasonal_series(), "a": seasonal_series(seed=3), "b": seasonal_series(seed=4)})
    cutoff = wide.index[150]
    tampered = wide.copy()
    tampered.loc[tampered.index >= cutoff, "a"] = np.random.default_rng(9).normal(size=(wide.index >= cutoff).sum())
    assert select_correlated(wide, "nat", 2, before=cutoff) == select_correlated(tampered, "nat", 2, before=cutoff)
