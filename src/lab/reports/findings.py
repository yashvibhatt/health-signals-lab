"""Plain-English findings built ONLY from computed numbers (templated text, no LLM)."""
from __future__ import annotations

import pandas as pd


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _join(names: list[str]) -> str:
    if not names:
        return "none"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def peak_timing(
    wide: pd.DataFrame, seasons: pd.Series, exclude: list[str], smooth: int = 1
) -> pd.DataFrame:
    """Weeks between each location's seasonal ILI peak and the national peak, per season.

    Peaks are located on a centred `smooth`-week rolling mean so a single holiday-week spike
    does not count as the season peak.

    Negative = the location peaked earlier than the nation. Columns = locations, index = season.
    """
    wide = wide.rolling(smooth, center=True, min_periods=1).mean()
    rows = {}
    for season in sorted(set(seasons) - set(exclude)):
        block = wide.loc[seasons == season]
        # Locations with no data this season get NaN instead of a bogus peak.
        peaks = block.dropna(axis=1, how="all").idxmax().reindex(block.columns)
        rows[season] = (peaks - peaks["nat"]).dt.days / 7
    return pd.DataFrame(rows).T.drop(columns="nat")


def region_findings(
    region_name: str,
    pairs: pd.DataFrame,
    vs_nat: pd.DataFrame,
    vs_nat_ex_covid: pd.DataFrame,
    vs_nat_ex_holiday: pd.DataFrame,
    timing: pd.DataFrame,
    labels: dict[str, str],
    min_peak_r: float,
    min_gain: float,
    covid: list[str],
) -> list[str]:
    out = []
    lab = lambda x: labels[x]  # noqa: E731

    n_pairs = len(pairs)
    sync = int((pairs["peak_lag"] == 0).sum())
    leads = pairs[pairs["leader"].notna()]
    out.append(
        f"Across {_plural(n_pairs, 'state pair')} in {region_name}, {sync} "
        f"({sync / n_pairs:.0%}) have their strongest week-over-week correlation at lag 0, "
        f"i.e. they move in the same week. {_plural(len(leads), 'pair')} show a lead of 1 or more weeks "
        f"(peak r ≥ {min_peak_r} and at least {min_gain} higher than at lag 0)."
    )
    if len(leads):
        top = leads.sort_values("peak_r", ascending=False).iloc[0]
        out.append(
            f"The strongest lead/lag pair is {lab(top['leader'])} leading {lab(top['follower'])} by "
            f"{_plural(int(top['lead_weeks']), 'week')} (peak r = {top['peak_r']:.2f}, "
            f"versus r = {top['r_lag0']:.2f} at lag 0)."
        )
    else:
        best = pairs.sort_values("peak_r", ascending=False).iloc[0]
        out.append(
            f"The most tightly linked pair is {lab(best['loc_a'])} and {lab(best['loc_b'])} "
            f"(r = {best['peak_r']:.2f} at lag {best['peak_lag']:+d})."
        )

    before = vs_nat.loc[vs_nat["leader"] == vs_nat["location"], "location"].map(lab).tolist()
    after = vs_nat.loc[vs_nat["follower"] == vs_nat["location"], "location"].map(lab).tolist()
    if before or after:
        out.append(
            f"Compared with national ILI, states whose changes come first: {_join(before)}; "
            f"states whose changes come later: {_join(after)}. The rest move in the same week "
            f"as the national curve."
        )
    else:
        out.append(
            f"Compared with national ILI, all {len(vs_nat)} states move in the same week as the "
            f"national curve (peak correlation at lag 0, r = {vs_nat['peak_r'].min():.2f} to "
            f"{vs_nat['peak_r'].max():.2f})."
        )

    med = timing.median().sort_values()
    if len(med):
        first, last = med.index[0], med.index[-1]
        out.append(
            f"By seasonal peak timing (median over {len(timing)} non-COVID seasons), "
            f"{lab(first)} peaks earliest ({med[first]:+.1f} weeks vs the national peak) and "
            f"{lab(last)} latest ({med[last]:+.1f} weeks)."
        )

    for alt, what in [(vs_nat_ex_covid, f"the {_join(covid)} seasons"),
                      (vs_nat_ex_holiday, "the late-December/early-January holiday weeks")]:
        merged = vs_nat.merge(alt, on="location", suffixes=("", "_alt"))
        changed = merged.loc[merged["peak_lag"] != merged["peak_lag_alt"], "location"].map(lab).tolist()
        out.append(
            f"Robustness: excluding {what} changes the peak lag vs national for "
            f"{len(changed)} of {len(merged)} states" + (f" ({_join(changed)})." if changed else ".")
        )
    return out
