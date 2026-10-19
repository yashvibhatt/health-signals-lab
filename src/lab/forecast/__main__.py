"""Forecast national weekly ILI 1-4 weeks ahead.

    python -m lab.forecast backtest [--no-asof]   rolling-origin backtest -> outputs/forecast/
    python -m lab.forecast report                 rebuild results.md from saved predictions
    python -m lab.forecast train                  fit production model on all data -> outputs/models/
    python -m lab.forecast predict [--refresh]    load saved model, forecast next 4 weeks
"""
from __future__ import annotations

import argparse
import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lab.config import load_config, path  # noqa: E402
from lab.data.delphi import asof_series, fetch_latest_national, issue_snapshot, load_ili, season_series  # noqa: E402
from lab.forecast.backtest import default_specs, run_backtest, score  # noqa: E402
from lab.forecast.features import feature_columns, make_features  # noqa: E402
from lab.forecast.models import fit_tree, training_frame  # noqa: E402
from lab.logging_setup import setup_logging  # noqa: E402
from lab.reports.correlate import select_correlated  # noqa: E402

log = logging.getLogger("lab.forecast")
HIGH_ACTIVITY = 4.0  # wILI % threshold for "high-activity" weeks in the results breakdown


def _setup():
    cfg = load_config()
    fcfg = cfg["forecast"]
    wide = load_ili()
    y = wide[fcfg["target"]]
    seasons = season_series(wide.index)
    drop = seasons.isin(cfg["delphi"]["covid_seasons"]) if fcfg["exclude_covid_from_training"] else None
    out = path("outputs") / "forecast"
    out.mkdir(exist_ok=True)
    return cfg, fcfg, wide, y, drop, out


def _pivot(metrics: pd.DataFrame, value: str) -> pd.DataFrame:
    return metrics.pivot(index="model", columns="horizon", values=value).add_prefix("h=")


def _md_table(df: pd.DataFrame, fmt: str) -> str:
    cols = [df.index.name or "", *map(str, df.columns)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx), *(format(v, fmt) for v in row)]) + " |")
    return "\n".join(lines)


def cmd_backtest(asof: bool) -> None:
    cfg, fcfg, wide, y, drop, out = _setup()
    start = pd.Timestamp(fcfg["backtest_start"])
    origins = [d for d in y.index if start <= d < y.index[-1]]
    states = select_correlated(wide, fcfg["target"], fcfg["n_correlated_states"], before=start)
    log.info("Correlated states (selected on data before %s): %s", start.date(), states)
    log.info("Backtest: %d origins %s -> %s, horizons %s",
             len(origins), origins[0].date(), origins[-1].date(), fcfg["horizons"])

    preds = run_backtest(y, origins, fcfg["horizons"], default_specs(with_extra=True), fcfg,
                         extra_final=wide[states], drop_mask=drop)
    preds.to_csv(out / "backtest_predictions.csv", index=False)
    if asof:
        log.info("Real-time (as-of) backtest: national-only models on data as published at each origin")
        preds_asof = run_backtest(y, origins, fcfg["horizons"], default_specs(with_extra=False), fcfg,
                                  drop_mask=drop, series_at=lambda t: asof_series(y, t))
        preds_asof.to_csv(out / "backtest_predictions_asof.csv", index=False)
    (out / "correlated_states.txt").write_text(",".join(states) + "\n")
    cmd_report()


def cmd_report() -> None:
    """Write results.md, metric CSVs and the chart from saved backtest predictions (no refitting)."""
    cfg, fcfg, wide, y, drop, out = _setup()
    preds = pd.read_csv(out / "backtest_predictions.csv", parse_dates=["origin", "target_date"])
    states = (out / "correlated_states.txt").read_text().strip().split(",")
    origins = sorted(preds["origin"].unique())
    metrics = score(preds)
    metrics.to_csv(out / "backtest_metrics.csv", index=False)

    order = ["persistence", "seasonal_naive", "arima", "lightgbm", "xgboost", "lightgbm+states", "xgboost+states"]
    md = ["# Backtest results (finalized data)\n",
          f"Origins: {len(origins)} weekly origins, {origins[0].date()} to {origins[-1].date()} "
          f"(expanding window, refit at every origin). Target: national wILI %.\n",
          f"Correlated states (chosen on data before {fcfg['backtest_start']}): {', '.join(states)}.\n"]
    for metric, fmt in [("MAE", ".3f"), ("RMSE", ".3f"), ("MAPE", ".1f")]:
        md += [f"\n## {metric}{' (%)' if metric == 'MAPE' else ''}\n",
               _md_table(_pivot(metrics, metric).loc[order], fmt)]
    mae = _pivot(metrics, "MAE").loc[order]
    for base, title in [("persistence", "persistence (no change from origin week)"),
                        ("seasonal_naive", "seasonal naive"), ("arima", "ARIMA")]:
        md += [f"\n## MAE skill vs {title} (% reduction; positive = better)\n",
               _md_table((1 - mae / mae.loc[base]) * 100, ".1f")]

    high = preds.assign(ae=(preds["y_pred"] - preds["y_true"]).abs(),
                        week=lambda d: d["y_true"].ge(HIGH_ACTIVITY).map({True: "high", False: "low"}))
    tab = high.pivot_table(index="model", columns=["week", "horizon"], values="ae", aggfunc="mean")
    n_high = int((high["week"].eq("high") & high["model"].eq(order[0]) & high["horizon"].eq(1)).sum())
    md += [f"\n## MAE in high-activity weeks (actual wILI ≥ {HIGH_ACTIVITY}%, n={n_high} per horizon)\n",
           _md_table(tab["high"].loc[order].add_prefix("h="), ".3f"),
           f"\n## MAE in other weeks (actual wILI < {HIGH_ACTIVITY}%)\n",
           _md_table(tab["low"].loc[order].add_prefix("h="), ".3f")]

    asof_file = out / "backtest_predictions_asof.csv"
    if asof_file.exists():
        md += _asof_section(y, origins, preds, pd.read_csv(asof_file), out)

    (out / "results.md").write_text("\n".join(md) + "\n")
    _plot(preds, fcfg["production_model"], out / "backtest_h1_h4.png")
    print("\n".join(md))


def _asof_section(y, origins, preds_revised, preds_asof, out) -> list[str]:
    """Compare national-only models run on as-published (real-time) vs finalized data."""
    names = preds_asof["model"].unique()
    m_asof = score(preds_asof)
    m_rev = score(preds_revised[preds_revised["model"].isin(names)])
    cmp = m_rev.merge(m_asof, on=["model", "horizon"], suffixes=("_revised", "_asof"))
    cmp["MAE_change_pct"] = (100 * (cmp["MAE_asof"] / cmp["MAE_revised"] - 1)).round(1)
    cmp.to_csv(out / "asof_vs_revised.csv", index=False)

    snaps = {t: issue_snapshot(t) for t in origins}
    valid = [t for t, s in snaps.items() if s is not None and t in s.index]
    first = pd.Series({t: snaps[t].loc[t] for t in valid})
    rev = y.loc[valid] - first
    n_fallback = len(origins) - len(valid)
    md = ["\n## Real-time (as-of) data vs finalized data\n",
          f"First-reported national wILI differed from the final value by a mean absolute "
          f"{rev.abs().mean():.3f} points (max {rev.abs().max():.3f}; final minus first-reported: "
          f"mean {rev.mean():+.3f}) over the {len(valid)} origin weeks with a published issue. "
          f"The other {n_fallback} origins had no issue in Delphi and used finalized data.\n",
          "MAE when models see data as published at each origin vs finalized data "
          "(national-only features; scored against finalized values):\n"]
    for model in ["persistence", "seasonal_naive", "arima", "lightgbm", "xgboost"]:
        c = cmp[cmp["model"] == model].set_index("horizon")
        md.append(f"- **{model}**: " + "; ".join(
            f"h={h}: {c.loc[h, 'MAE_revised']:.3f} → {c.loc[h, 'MAE_asof']:.3f} "
            f"({c.loc[h, 'MAE_change_pct']:+.1f}%)" for h in c.index))
    return md


def _plot(preds: pd.DataFrame, model: str, out) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, h in zip(axes, (1, 4)):
        d = preds[(preds["model"] == model) & (preds["horizon"] == h)]
        naive = preds[(preds["model"] == "seasonal_naive") & (preds["horizon"] == h)]
        ax.plot(d["target_date"], d["y_true"], color="#0b0b0b", lw=2, label="Actual")
        ax.plot(d["target_date"], d["y_pred"], color="#2a78d6", lw=1.4, label=model)
        ax.plot(naive["target_date"], naive["y_pred"], color="#eb6834", lw=1, ls="--", label="Seasonal naive")
        ax.set_title(f"{h}-week-ahead", loc="left", fontsize=10)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("National wILI %")
    axes[1].legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def cmd_train() -> None:
    cfg, fcfg, wide, y, drop, _ = _setup()
    kind = fcfg["production_model"]
    states = select_correlated(wide, fcfg["target"], fcfg["n_correlated_states"]) \
        if fcfg.get("production_use_states") else []
    extra = wide[states] if states else None
    bundle = {"kind": kind, "states": states, "trained_through": y.index[-1], "models": {}, "columns": {}}
    for h in fcfg["horizons"]:
        _, train = training_frame(y, h, extra, drop)
        bundle["models"][h] = fit_tree(kind, fcfg[kind], train)
        bundle["columns"][h] = feature_columns(train)
        log.info("h=%d: trained %s on %d rows", h, kind, len(train))
    out = path("outputs") / "models" / "forecast.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, out)
    log.info("Saved %s (states=%s, data through %s) -> %s", kind, states or "none",
             y.index[-1].date(), out)


def predict_next(bundle: dict, wide: pd.DataFrame, target: str) -> pd.DataFrame:
    y = wide[target]
    origin = y.index[-1]
    extra = wide[bundle["states"]] if bundle["states"] else None
    rows = []
    for h, model in bundle["models"].items():
        frame = make_features(y, h, extra)
        x = frame.loc[[origin], bundle["columns"][h]]
        rows.append({
            "origin_week": origin.date(), "horizon": h,
            "target_week": (origin + pd.Timedelta(weeks=h)).date(),
            "forecast_wili": round(float(y.iloc[-1] * np.exp(model.predict(x)[0])), 3),
        })
    return pd.DataFrame(rows)


def cmd_predict(refresh: bool) -> None:
    cfg = load_config()
    target = cfg["forecast"]["target"]
    bundle = joblib.load(path("outputs") / "models" / "forecast.joblib")
    if bundle["states"]:
        log.warning("Model uses state features; forecasting from the pinned analysis window")
        wide = load_ili(refresh=refresh)
    else:
        # Newest published week. The latest weeks are preliminary, but the as-of backtest showed
        # national first reports differ from final values by only a few hundredths of a point.
        wide = fetch_latest_national(refresh=refresh).to_frame(target)
    if wide.index[-1] > bundle["trained_through"]:
        log.info("Data now runs to %s (model trained through %s)", wide.index[-1].date(),
                 bundle["trained_through"].date())
    fc = predict_next(bundle, wide, target)
    out = path("outputs") / "forecast" / "forecast_next4.csv"
    out.parent.mkdir(exist_ok=True)
    fc.to_csv(out, index=False)
    log.info("Latest observed national wILI: %.3f (week of %s)",
             wide[target].iloc[-1], wide.index[-1].date())
    print(fc.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--no-asof", action="store_true", help="skip the real-time data comparison")
    sub.add_parser("report", help="rebuild results.md from saved backtest predictions")
    sub.add_parser("train")
    p = sub.add_parser("predict")
    p.add_argument("--refresh", action="store_true", help="re-download latest ILI before predicting")
    args = parser.parse_args()
    setup_logging()
    if args.cmd == "backtest":
        cmd_backtest(asof=not args.no_asof)
    elif args.cmd == "report":
        cmd_report()
    elif args.cmd == "train":
        cmd_train()
    else:
        cmd_predict(args.refresh)


if __name__ == "__main__":
    main()
