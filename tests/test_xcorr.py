from datetime import date

import numpy as np
import pandas as pd

from lab.data.delphi import date_to_epiweek, epiweek_start, season_of
from lab.reports.correlate import lag_matrix, lagged_xcorr, pairwise_leads, select_leaders


def _synthetic(lead: int = 2, n: int = 300, seed: int = 0) -> pd.DataFrame:
    """B is A delayed by `lead` weeks plus noise, so A should lead B by `lead`."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n).cumsum()
    b = np.roll(a, lead) + rng.normal(scale=0.1, size=n)
    idx = pd.date_range("2018-09-30", periods=n, freq="7D")
    return pd.DataFrame({"a": a, "b": b}, index=idx).iloc[lead:]


def test_lagged_xcorr_finds_known_lead():
    d = _synthetic(lead=2).diff()
    xc = lagged_xcorr(d["a"], d["b"], max_lag=4)
    assert xc.idxmax() == 2
    # Reversing the arguments flips the sign of the lag.
    assert lagged_xcorr(d["b"], d["a"], max_lag=4).idxmax() == -2


def test_pairwise_leads_labels_leader():
    d = _synthetic(lead=1).diff()
    row = pairwise_leads(d, ["a", "b"], max_lag=4, min_peak_r=0.3).iloc[0]
    assert (row["leader"], row["follower"], row["lead_weeks"]) == ("a", "b", 1)


def test_lag_matrix_is_antisymmetric():
    d = _synthetic(lead=3).diff()
    m = lag_matrix(pairwise_leads(d, ["a", "b"], 4, 0.3), ["a", "b"])
    assert m.loc["a", "b"] == 3 and m.loc["b", "a"] == -3


def test_select_leaders_ignores_data_after_cutoff():
    """Changing data after `before` must not change which leaders are selected."""
    d = _synthetic(lead=2).rename(columns={"a": "x", "b": "nat"})
    cutoff = d.index[150]
    tampered = d.copy()
    tampered.loc[tampered.index >= cutoff] = 0.0
    assert select_leaders(d, "nat", 1, 4, 0.3, before=cutoff) == \
           select_leaders(tampered, "nat", 1, 4, 0.3, before=cutoff) == ["x"]


def test_epiweek_roundtrip_and_season():
    assert epiweek_start(202401) == date(2023, 12, 31)   # MMWR 2024 week 1 starts Sun Dec 31
    assert epiweek_start(202053) == date(2020, 12, 27)   # 2020 has 53 MMWR weeks
    for ew in (201840, 201901, 202053, 202139, 202640):
        assert date_to_epiweek(epiweek_start(ew)) == ew
    assert season_of(201840) == "2018-19" and season_of(201939) == "2018-19"
    assert season_of(202601) == "2025-26"
