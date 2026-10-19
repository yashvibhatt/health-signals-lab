"""Rolling-origin (expanding-window) backtest.

For every origin T the models only see `series_at(T)`, which ends at T. With the default
`series_at` that is the finalized series truncated at T; for the real-time comparison it is
the snapshot CDC published right after week T (Delphi `issues=`). Forecasts are always scored
against finalized values.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from lab.forecast.models import ArimaModel, fit_tree, predict_tree, seasonal_naive, training_frame

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str                  # "tree" | "persistence" | "naive" | "arima"
    tree_kind: str | None = None
    use_extra: bool = False    # include correlated-state features


def default_specs(with_extra: bool) -> list[ModelSpec]:
    specs = [
        ModelSpec("persistence", "persistence"),
        ModelSpec("seasonal_naive", "naive"),
        ModelSpec("arima", "arima"),
        ModelSpec("lightgbm", "tree", "lightgbm"),
        ModelSpec("xgboost", "tree", "xgboost"),
    ]
    if with_extra:
        specs += [
            ModelSpec("lightgbm+states", "tree", "lightgbm", use_extra=True),
            ModelSpec("xgboost+states", "tree", "xgboost", use_extra=True),
        ]
    return specs


def run_backtest(
    y_final: pd.Series,
    origins: list[pd.Timestamp],
    horizons: list[int],
    specs: list[ModelSpec],
    cfg: dict,
    extra_final: pd.DataFrame | None = None,
    drop_mask: pd.Series | None = None,
    series_at: Callable[[pd.Timestamp], pd.Series] | None = None,
) -> pd.DataFrame:
    arima = ArimaModel(cfg["arima"]["order"], cfg["arima"]["window_weeks"])
    rows = []
    for i, origin in enumerate(origins):
        y_hist = series_at(origin) if series_at else y_final.loc[:origin]
        assert y_hist.index[-1] == origin, "history must end exactly at the origin"
        extra_hist = None if extra_final is None else extra_final.loc[:origin]
        for h in horizons:
            target_date = origin + pd.Timedelta(weeks=h)
            if target_date not in y_final.index:
                continue
            frames = {}
            for spec in specs:
                if spec.kind == "persistence":
                    pred = float(y_hist.iloc[-1])          # no change from the origin week
                elif spec.kind == "naive":
                    pred = seasonal_naive(y_hist, h)
                elif spec.kind == "arima":
                    pred = arima.forecast(y_hist, h, max_h=max(horizons))
                else:
                    if spec.use_extra not in frames:
                        frames[spec.use_extra] = training_frame(
                            y_hist, h, extra_hist if spec.use_extra else None, drop_mask)
                    frame, train = frames[spec.use_extra]
                    model = fit_tree(spec.tree_kind, cfg[spec.tree_kind], train)
                    pred = predict_tree(model, frame, y_hist)
                rows.append({
                    "origin": origin, "horizon": h, "target_date": target_date, "model": spec.name,
                    "y_true": float(y_final[target_date]), "y_pred": pred,
                })
        if (i + 1) % 26 == 0:
            log.info("  %d/%d origins done", i + 1, len(origins))
    return pd.DataFrame(rows)


def score(preds: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE, MAPE (%) per model x horizon."""
    err = preds["y_pred"] - preds["y_true"]
    d = preds.assign(abs_err=err.abs(), sq_err=err**2, ape=100 * err.abs() / preds["y_true"])
    g = d.groupby(["model", "horizon"])
    out = pd.DataFrame({
        "MAE": g["abs_err"].mean(),
        "RMSE": np.sqrt(g["sq_err"].mean()),
        "MAPE": g["ape"].mean(),
        "n": g.size(),
    }).reset_index()
    return out.round({"MAE": 3, "RMSE": 3, "MAPE": 1})
