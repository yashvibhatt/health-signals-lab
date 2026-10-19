"""Fetch weekly ILINet data (national + states) from the Delphi Epidata fluview endpoint.

Docs: https://cmu-delphi.github.io/delphi-epidata/api/fluview.html  (free, no key)
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

from lab.config import load_config, path

log = logging.getLogger(__name__)

API_URL = "https://api.delphi.cmu.edu/epidata/fluview/"
REQUEST_PAUSE_S = 1.0  # be polite to a free public API
ROW_LIMIT = 3650       # Delphi's max rows per response
BATCH_SIZE = 8         # 8 locations x ~418 weeks stays under ROW_LIMIT
MAX_WAIT_S = 300       # fail with a clear message rather than sleep for an hour

# HHS regions -> member jurisdictions (lower-case postal codes, as used by Delphi).
HHS_REGIONS: dict[str, list[str]] = {
    "hhs1": ["ct", "me", "ma", "nh", "ri", "vt"],
    "hhs2": ["nj", "ny"],
    "hhs3": ["de", "dc", "md", "pa", "va", "wv"],
    "hhs4": ["al", "fl", "ga", "ky", "ms", "nc", "sc", "tn"],
    "hhs5": ["il", "in", "mi", "mn", "oh", "wi"],
    "hhs6": ["ar", "la", "nm", "ok", "tx"],
    "hhs7": ["ia", "ks", "mo", "ne"],
    "hhs8": ["co", "mt", "nd", "sd", "ut", "wy"],
    "hhs9": ["az", "ca", "hi", "nv"],
    "hhs10": ["ak", "id", "or", "wa"],
}
STATES = sorted(s for members in HHS_REGIONS.values() for s in members)
NATIONAL = "nat"


def epiweek_start(epiweek: int) -> date:
    """Sunday that starts an MMWR epiweek (week 1 = first Sun-Sat week with >=4 days in Jan)."""
    year, week = divmod(epiweek, 100)
    jan4 = date(year, 1, 4)
    week1_start = jan4 - timedelta(days=(jan4.weekday() + 1) % 7)
    return week1_start + timedelta(weeks=week - 1)


def season_of(epiweek: int) -> str:
    """Flu season label, e.g. 201845 -> '2018-19', 201910 -> '2018-19'."""
    year, week = divmod(epiweek, 100)
    start = year if week >= 40 else year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def fetch_locations(
    locations: list[str], start: int, end: int, max_retries: int = 5, issue: int | None = None
) -> pd.DataFrame:
    """One API call for several locations. Backs off on HTTP 429 (anonymous rate limit).

    With `issue`, returns the values exactly as published in that weekly issue (as-of data).
    """
    params = {"regions": ",".join(locations), "epiweeks": f"{start}-{end}"}
    if issue is not None:
        params["issues"] = issue
    if key := os.environ.get("DELPHI_API_KEY"):
        params["api_key"] = key
    for attempt in range(max_retries):
        resp = requests.get(API_URL, params=params, timeout=60)
        if resp.status_code != 429:
            break
        wait = int(resp.headers.get("Retry-After", 0)) or 5 * 2**attempt
        if wait > MAX_WAIT_S:
            raise RuntimeError(
                f"Delphi rate limit hit; server asks to wait {wait}s. Retry later, or register a "
                "free key at https://api.delphi.cmu.edu/epidata/admin/registration_form and set "
                "DELPHI_API_KEY. Batches already downloaded are cached in data/raw/."
            )
        log.info("Rate limited; retrying in %ds", wait)
        time.sleep(wait)
    resp.raise_for_status()
    time.sleep(REQUEST_PAUSE_S)
    payload = resp.json()
    if payload.get("result") == -2 and issue is not None:  # "no results": issue not published
        return pd.DataFrame()
    if payload.get("result") != 1:
        raise RuntimeError(f"Delphi error for {locations}: {payload.get('message')}")
    if len(payload["epidata"]) >= ROW_LIMIT:
        raise RuntimeError("Response hit the API row limit; reduce BATCH_SIZE.")
    return pd.DataFrame(payload["epidata"])


def fetch_fluview(refresh: bool = False) -> pd.DataFrame:
    """Download (or load cached) raw fluview rows for national + every state/territory."""
    cfg = load_config()["delphi"]
    out = path("raw") / "fluview.csv"
    if out.exists() and not refresh:
        log.info("Using cached %s", out)
        return pd.read_csv(out)

    locations = [NATIONAL, *STATES, cfg["ny_location"], *cfg["excluded_locations"]]
    batch_dir = path("raw") / "fluview_batches"
    batch_dir.mkdir(exist_ok=True)
    frames = []
    for i in range(0, len(locations), BATCH_SIZE):
        batch = locations[i : i + BATCH_SIZE]
        batch_file = batch_dir / f"{'_'.join(batch)}.csv"
        if batch_file.exists() and not refresh:
            df = pd.read_csv(batch_file)
        else:
            df = fetch_locations(batch, cfg["start_epiweek"], cfg["end_epiweek"])
            df.to_csv(batch_file, index=False)
        log.info("%-40s %5d rows", ",".join(batch), len(df))
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(out, index=False)
    log.info("Saved %d rows -> %s", len(raw), out)
    return raw


def tidy_ili(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (wide weekly ILI table, coverage table).

    Wide table: index = week_start date, columns = location codes ('nat', 'ca', ...),
    values = weighted ILI %. Locations below `min_coverage` are dropped (see coverage table).
    """
    cfg = load_config()["delphi"]
    df = raw[["region", "epiweek", "wili"]].copy()
    # "ny" (combined) stops in 2025; use NY-excluding-NYC under the "ny" label.
    df = df[df["region"] != "ny"]
    df.loc[df["region"] == cfg["ny_location"], "region"] = "ny"
    df = df[~df["region"].isin(cfg["excluded_locations"])]

    all_weeks = sorted(raw.loc[raw["region"] == NATIONAL, "epiweek"].unique())
    wide = df.pivot_table(index="epiweek", columns="region", values="wili").reindex(all_weeks)

    coverage = wide.notna().mean().rename("coverage").to_frame()
    coverage["first_epiweek"] = wide.apply(lambda s: s.first_valid_index())
    coverage["included"] = coverage["coverage"] >= cfg["min_coverage"]
    for loc in coverage.index[~coverage["included"]]:
        log.warning("Dropping %s: only %.0f%% weekly coverage", loc, 100 * coverage.loc[loc, "coverage"])

    wide = wide.loc[:, coverage["included"]]
    # Fill isolated single-week gaps only; longer gaps stay NaN.
    wide = wide.interpolate(limit=1, limit_area="inside")
    wide.index = pd.to_datetime([epiweek_start(w) for w in wide.index])
    wide.index.name = "week_start"
    return wide, coverage.sort_values("coverage")


def load_ili(refresh: bool = False) -> pd.DataFrame:
    """Convenience: fetched + tidied wide ILI table."""
    wide, _ = tidy_ili(fetch_fluview(refresh=refresh))
    return wide


def date_to_epiweek(d: date) -> int:
    """Inverse of epiweek_start for any date inside the week."""
    for year in (d.year + 1, d.year, d.year - 1):
        start = epiweek_start(year * 100 + 1)
        if start <= d:
            return year * 100 + (d - start).days // 7 + 1
    raise ValueError(d)


def season_series(index: pd.DatetimeIndex) -> pd.Series:
    """Season label ('2018-19', ...) for each week_start in a DatetimeIndex."""
    labels = [season_of(date_to_epiweek(d.date())) for d in index]
    return pd.Series(labels, index=index, name="season")


def issue_snapshot(origin: pd.Timestamp, location: str = NATIONAL) -> pd.Series | None:
    """Values exactly as published in the weekly issue right after week `origin` (cached).

    CDC's issue for week N+1 carries the trailing ~52 weeks through week N. Returns None if
    Delphi has no such issue.
    """
    issue = date_to_epiweek((origin + pd.Timedelta(weeks=1)).date())
    cache_dir = path("raw") / "fluview_issues"
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"{location}_{issue}.csv"
    if cache.exists():
        try:
            snap = pd.read_csv(cache)
        except pd.errors.EmptyDataError:
            snap = pd.DataFrame()
    else:
        snap = fetch_locations([location], 199740, issue, issue=issue)
        snap.to_csv(cache, index=False)
    if snap.empty or "wili" not in snap:
        return None
    return pd.Series(snap["wili"].to_numpy(),
                     index=pd.to_datetime([epiweek_start(e) for e in snap["epiweek"]]))


def asof_series(final: pd.Series, origin: pd.Timestamp, location: str = NATIONAL) -> pd.Series:
    """Series as it was known right after week `origin` was first reported.

    Weeks in the issue snapshot replace finalized values; older weeks (rarely revised after a
    year) keep final values. Falls back to finalized data if the issue is missing (logged).
    """
    s = final.loc[:origin].copy()
    snap = issue_snapshot(origin, location)
    if snap is None:
        log.warning("No issue after %s for %s; using finalized data", origin.date(), location)
        return s
    if origin not in snap.index:
        log.warning("Issue after %s lacks that week; using finalized value for it", origin.date())
    s.update(snap[snap.index <= origin])
    return s


def fetch_latest_national(refresh: bool = False) -> pd.Series:
    """National wILI from the configured start through the newest published week (for `predict`).

    Unlike the pinned analysis window, this always extends to today. Cached until --refresh.
    """
    cfg = load_config()["delphi"]
    cache = path("raw") / "fluview_nat_latest.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache)
    else:
        df = fetch_locations([NATIONAL], cfg["start_epiweek"], date_to_epiweek(date.today()))
        df.to_csv(cache, index=False)
    y = pd.Series(df["wili"].to_numpy(), index=pd.to_datetime([epiweek_start(e) for e in df["epiweek"]]),
                  name=NATIONAL).sort_index()
    y.index.name = "week_start"
    return y
