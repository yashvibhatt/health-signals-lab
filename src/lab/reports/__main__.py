"""`python -m lab.reports`: fetch ILI (cached), compute lead/lag, write one HTML report per HHS region.

Outputs (outputs/reports/):
    index.html, hhs1.html ... hhs10.html, *.png charts
    xcorr_state_pairs.csv     every state pair, all seasons
    xcorr_vs_national.csv     every state vs national, all seasons and excluding COVID seasons
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from lab.config import load_config, path, load_env
from lab.data.delphi import (
    HHS_REGIONS, NATIONAL, date_to_epiweek, fetch_fluview, season_series, tidy_ili,
)
from lab.logging_setup import setup_logging
from lab.reports import charts
from lab.reports.correlate import lag_matrix, mask_rows, pairwise_leads, vs_reference, weekly_change
from lab.reports.findings import peak_timing, region_findings

log = logging.getLogger("lab.reports")


def label(loc: str) -> str:
    return {"nat": "National", "ny": "NY (excl. NYC)"}.get(loc, loc.upper())


def excluded_reasons(coverage: pd.DataFrame, cfg: dict) -> dict[str, str]:
    reasons = {
        "pr": "territory; outside the 50 states + DC comparison",
        "vi": "territory; outside the 50 states + DC comparison",
        "jfk": "NYC reported separately and series ends 2025-W39; NY uses NY-excl.-NYC instead",
    }
    reasons = {k: v for k, v in reasons.items() if k in cfg["excluded_locations"]}
    for loc, row in coverage[~coverage["included"]].iterrows():
        reasons[loc] = (f"only {row['coverage']:.0%} of weeks reported "
                        f"(first week {int(row['first_epiweek'])}; threshold {cfg['min_coverage']:.0%})")
    return reasons


def partial_notes(wide: pd.DataFrame, seasons: pd.Series) -> dict[str, str]:
    """Included locations that are missing one or more whole seasons."""
    notes = {}
    for loc in wide.columns:
        missing = [s for s in sorted(seasons.unique()) if wide.loc[(seasons == s).values, loc].isna().all()]
        if missing:
            notes[loc] = f"included, but no data for {', '.join(missing)}"
    return notes


def pair_text(p: pd.Series, min_peak_r: float) -> str:
    if pd.notna(p["leader"]):
        w = int(p["lead_weeks"])
        return f"{label(p['leader'])} leads {label(p['follower'])} by {w} week{'s' * (w != 1)}"
    if p["peak_lag"] == 0:
        return "Move in the same week"
    return f"Peak at lag {p['peak_lag']:+d} but r < {min_peak_r}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download ILI data")
    args = parser.parse_args()
    setup_logging()
    load_env()

    cfg = load_config()
    dcfg, ccfg = cfg["delphi"], cfg["correlation"]
    max_lag, min_r, covid = ccfg["max_lag"], ccfg["min_peak_r"], dcfg["covid_seasons"]
    min_gain = ccfg["min_gain_over_lag0"]
    out_dir = path("outputs") / "reports"
    out_dir.mkdir(exist_ok=True)

    wide, coverage = tidy_ili(fetch_fluview(refresh=args.refresh))
    seasons = season_series(wide.index)
    changes = weekly_change(wide)
    # Mask (not drop) COVID weeks so lagged shifts still pair weeks that are k weeks apart.
    changes_ex_covid = mask_rows(changes, seasons.isin(covid))
    mmwr_week = pd.Series([date_to_epiweek(d.date()) % 100 for d in wide.index], index=wide.index)
    changes_ex_holiday = mask_rows(changes, mmwr_week.isin(ccfg["holiday_mmwr_weeks"]))
    states = [c for c in wide.columns if c != NATIONAL]
    excluded = excluded_reasons(coverage, dcfg)
    partial = partial_notes(wide, seasons)
    timing = peak_timing(wide, seasons, exclude=covid, smooth=ccfg["peak_smoothing_weeks"])
    log.info("Analysing %d states, %d weeks, %d seasons", len(states), len(wide), seasons.nunique())

    # Nationwide tables (also consumed by Module 3 for context; Module 3 re-selects leaders itself).
    vs_all = vs_reference(changes, states, NATIONAL, max_lag, min_r, min_gain)
    vs_all_ex = vs_reference(changes_ex_covid, states, NATIONAL, max_lag, min_r, min_gain)
    vs_all_hol = vs_reference(changes_ex_holiday, states, NATIONAL, max_lag, min_r, min_gain)
    keep = ["location", "peak_lag", "peak_r", "r_lag0"]
    (vs_all.merge(vs_all_ex[keep], on="location", suffixes=("", "_ex_covid"))
           .merge(vs_all_hol[keep], on="location", suffixes=("", "_ex_holiday"))
           .to_csv(out_dir / "xcorr_vs_national.csv", index=False))
    all_pairs = pairwise_leads(changes, states, max_lag, min_r, min_gain)
    all_pairs.to_csv(out_dir / "xcorr_state_pairs.csv", index=False)
    log.info("Computed %d state pairs nationwide", len(all_pairs))

    summary = {
        "n_weeks": len(wide), "start": wide.index.min().date(), "end": wide.index.max().date(),
        "n_seasons": seasons.nunique(), "n_states": len(states) + 1,
    }
    env = Environment(loader=PackageLoader("lab.reports", "templates"), autoescape=select_autoescape())
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    index_rows = []

    for slug, members in HHS_REGIONS.items():
        locs = [m for m in members if m in states]
        title = f"HHS Region {slug[3:]}"
        labels = {loc: label(loc) for loc in [*locs, NATIONAL]}

        pairs = pairwise_leads(changes, locs, max_lag, min_r, min_gain)
        vs = vs_all[vs_all["location"].isin(locs)]
        vs_ex = vs_all_ex[vs_all_ex["location"].isin(locs)]
        vs_hol = vs_all_hol[vs_all_hol["location"].isin(locs)]
        findings = region_findings(title, pairs, vs, vs_ex, vs_hol, timing[locs], labels,
                                   min_r, min_gain, covid)

        charts.plot_series(wide, locs, seasons, covid, labels, out_dir / f"{slug}_series.png")
        charts.plot_lag_heatmap(lag_matrix(pairs, locs, "peak_lag"), lag_matrix(pairs, locs, "peak_r"),
                                labels, out_dir / f"{slug}_heatmap.png", max_lag)
        charts.plot_vs_national(vs, labels, out_dir / f"{slug}_vs_national.png", max_lag)

        state_table = [{
            "name": labels[loc], "mean": wide[loc].mean(), "max": wide[loc].max(),
            "max_week": wide[loc].idxmax().date(), "missing": int(wide[loc].isna().sum()),
        } for loc in locs]
        ranked = pairs.assign(has_lead=pairs["leader"].notna()).sort_values(
            ["has_lead", "peak_r"], ascending=False).head(ccfg["top_n_pairs"])
        top_pairs = [{
            "a": labels[p["loc_a"]], "b": labels[p["loc_b"]], "peak_lag": p["peak_lag"],
            "peak_r": p["peak_r"], "r_lag0": p["r_lag0"], "gain_over_lag0": p["gain_over_lag0"],
            "text": pair_text(p, min_r),
        } for _, p in ranked.iterrows()]
        vs_m = vs.merge(vs_ex, on="location", suffixes=("", "_ex")).merge(
            vs_hol[["location", "peak_lag"]].rename(columns={"peak_lag": "peak_lag_hol"}), on="location")
        vs_table = [{
            "name": labels[r["location"]], "peak_lag": r["peak_lag"], "peak_r": r["peak_r"],
            "peak_lag_ex": r["peak_lag_ex"], "peak_lag_hol": r["peak_lag_hol"],
            "timing": timing[r["location"]].median(),
        } for _, r in vs_m.iterrows()]

        html = env.get_template("region.html.j2").render(
            region_title=title, slug=slug, states_included=[labels[x] for x in locs],
            findings=findings, state_table=state_table, top_pairs=top_pairs, vs_table=vs_table,
            excluded=[(label(m), f"excluded: {excluded[m]}") for m in members if m in excluded]
            + [(label(m), partial[m]) for m in locs if m in partial],
            summary={**summary, "n_states": len(locs), "n_pairs": len(pairs)},
            max_lag=max_lag, min_peak_r=min_r, min_gain=min_gain, covid=covid, generated=generated,
            holiday_weeks=ccfg["holiday_mmwr_weeks"], smooth=ccfg["peak_smoothing_weeks"],
        )
        (out_dir / f"{slug}.html").write_text(html)

        leads = pairs[pairs["leader"].notna()]
        top = leads.sort_values("peak_r", ascending=False).head(1)
        index_rows.append({
            "slug": slug, "title": title, "n_states": len(locs), "n_pairs": len(pairs),
            "n_leads": len(leads),
            "top": "none" if top.empty else (
                f"{label(top.iloc[0]['leader'])} → {label(top.iloc[0]['follower'])} "
                f"({int(top.iloc[0]['lead_weeks'])} wk, r={top.iloc[0]['peak_r']:.2f})"),
        })
        log.info("%s: %d states, %d pairs, %d lead/lag pairs", title, len(locs), len(pairs), len(leads))

    nat_leaders = vs_all[vs_all["leader"] == vs_all["location"]].sort_values("peak_r", ascending=False)
    html = env.get_template("index.html.j2").render(
        regions=index_rows, summary=summary, min_peak_r=min_r, generated=generated,
        national_leaders=[{"name": label(r["location"]), **r} for _, r in nat_leaders.iterrows()],
        lag_counts=sorted(vs_all["peak_lag"].value_counts().items()),
        lag_counts_hol=sorted(vs_all_hol["peak_lag"].value_counts().items()),
        n_pair_leads=int(all_pairs["leader"].notna().sum()), n_pairs_all=len(all_pairs),
        n_pair_lag0=int((all_pairs["peak_lag"] == 0).sum()), min_gain=min_gain,
        excluded=sorted((label(k), f"excluded: {v}") for k, v in excluded.items())
        + sorted((label(k), v) for k, v in partial.items()),
    )
    (out_dir / "index.html").write_text(html)
    log.info("Wrote %d region reports + index -> %s", len(index_rows), out_dir / "index.html")


if __name__ == "__main__":
    main()
