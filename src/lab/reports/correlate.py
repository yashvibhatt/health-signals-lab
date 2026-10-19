"""Lagged cross-correlations between weekly ILI series.

Convention used everywhere in this project:
    r(k) = corr( dA_t , dB_{t+k} )      for k in [-max_lag, +max_lag]
where dX is the week-over-week change in ILI %. If r peaks at k > 0, A's changes
show up in B k weeks later, i.e. **A leads B by k weeks**.

We correlate *changes* rather than levels because every state's ILI peaks in winter;
raw levels correlate ~0.9 at lag 0 almost everywhere, which hides who moves first.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def weekly_change(wide: pd.DataFrame) -> pd.DataFrame:
    """First difference of each column (ILI percentage points per week)."""
    return wide.diff()


def mask_rows(changes: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Set masked weeks to NaN. Masking (not dropping) keeps shift(k) == k weeks apart."""
    return changes.mask(np.broadcast_to(np.asarray(mask, dtype=bool)[:, None], changes.shape))


def lagged_xcorr(a: pd.Series, b: pd.Series, max_lag: int) -> pd.Series:
    """r(k) = corr(a_t, b_{t+k}); NaNs are dropped pairwise for each lag."""
    return pd.Series(
        {k: a.corr(b.shift(-k)) for k in range(-max_lag, max_lag + 1)}, name="r"
    )


def summarize_pair(
    a_name: str, b_name: str, xc: pd.Series, min_peak_r: float, min_gain: float = 0.0
) -> dict:
    peak_lag = int(xc.idxmax())
    peak_r = float(xc.max())
    gain = peak_r - float(xc.loc[0])
    is_lead = peak_lag != 0 and peak_r >= min_peak_r and gain >= min_gain
    leader, follower = (a_name, b_name) if peak_lag > 0 else (b_name, a_name)
    return {
        "loc_a": a_name,
        "loc_b": b_name,
        "peak_lag": peak_lag,
        "peak_r": round(peak_r, 3),
        "r_lag0": round(float(xc.loc[0]), 3),
        "gain_over_lag0": round(gain, 3),
        "leader": leader if is_lead else None,
        "follower": follower if is_lead else None,
        "lead_weeks": abs(peak_lag) if is_lead else 0,
    }


def pairwise_leads(
    changes: pd.DataFrame, locations: list[str], max_lag: int, min_peak_r: float, min_gain: float = 0.0
) -> pd.DataFrame:
    """Peak lag / peak r for every unordered pair of locations."""
    rows = [
        summarize_pair(a, b, lagged_xcorr(changes[a], changes[b], max_lag), min_peak_r, min_gain)
        for a, b in combinations(locations, 2)
    ]
    return pd.DataFrame(rows)


def vs_reference(
    changes: pd.DataFrame,
    locations: list[str],
    reference: str,
    max_lag: int,
    min_peak_r: float,
    min_gain: float = 0.0,
) -> pd.DataFrame:
    """Each location vs a reference series (e.g. national). peak_lag > 0 => location leads."""
    rows = [
        summarize_pair(
            loc, reference, lagged_xcorr(changes[loc], changes[reference], max_lag), min_peak_r, min_gain
        )
        for loc in locations
        if loc != reference
    ]
    return pd.DataFrame(rows).rename(columns={"loc_a": "location"}).drop(columns="loc_b")


def lag_matrix(pairs: pd.DataFrame, locations: list[str], value: str = "peak_lag") -> pd.DataFrame:
    """Square matrix from pair rows. For peak_lag, cell [A, B] > 0 means row A leads column B."""
    m = pd.DataFrame(np.nan, index=locations, columns=locations)
    sign = -1 if value == "peak_lag" else 1
    for row in pairs.itertuples():
        v = getattr(row, value)
        m.loc[row.loc_a, row.loc_b] = v
        m.loc[row.loc_b, row.loc_a] = sign * v
    for loc in locations:
        m.loc[loc, loc] = 0 if value == "peak_lag" else 1.0
    return m


def select_leaders(
    wide: pd.DataFrame,
    reference: str,
    n: int,
    max_lag: int,
    min_peak_r: float,
    before: pd.Timestamp | None = None,
    min_gain: float = 0.0,
) -> list[str]:
    """Locations that most reliably lead `reference`, using ONLY data strictly before `before`.

    Used by Module 3 so the leader choice never sees backtest-period data.
    Ranked by peak_r among locations whose peak lag is positive (they move first).
    """
    hist = wide if before is None else wide.loc[wide.index < before]
    changes = weekly_change(hist)
    locs = [c for c in hist.columns if c != reference]
    vs = vs_reference(changes, locs, reference, max_lag, min_peak_r, min_gain)
    leaders = vs[vs["leader"] == vs["location"]]
    return leaders.sort_values("peak_r", ascending=False)["location"].head(n).tolist()
