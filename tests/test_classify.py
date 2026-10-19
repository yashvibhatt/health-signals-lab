"""Module 1 tests: no network, no API key - Claude is replaced by a fake client."""
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from lab.classify import extract
from lab.classify.evaluate import evaluate, load_gold, write_gold_template
from lab.classify.methods import families, method_family
from lab.classify.schema import Extraction

GOOD = {"study_type": "forecasting", "disease": ["influenza", "Influenza"], "data_sources": ["ILINet"],
        "methods": ["ARIMA", "LSTM"], "geography": ["United States"],
        "key_finding": "LSTM beat ARIMA for 1-4 week ahead ILI forecasts."}
CFG = {"model": "test-model", "prompt_version": "t", "max_tokens": 100, "max_attempts": 3, "workers": 1}
RECORD = {"pmid": "123", "title": "T", "abstract": "A"}


class FakeClient:
    """Returns the queued responses in order; counts calls."""

    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls += 1
        stop, payload = self.responses.pop(0)
        return SimpleNamespace(stop_reason=stop, usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                               parsed_output=SimpleNamespace(model_dump=lambda: dict(payload)))


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "CACHE_DIR", tmp_path / "cache")


def test_schema_dedupes_lists_and_rejects_bad_values():
    e = Extraction.model_validate(GOOD)
    assert e.disease == ["influenza"]
    with pytest.raises(ValidationError):
        Extraction.model_validate({**GOOD, "study_type": "modelling"})
    with pytest.raises(ValidationError):
        Extraction.model_validate({**GOOD, "key_finding": "x" * 500})


def test_retry_then_cache_hit_makes_no_second_call():
    client = FakeClient([("max_tokens", GOOD), ("end_turn", {**GOOD, "key_finding": ""}), ("end_turn", GOOD)])
    first = extract.extract_one(client, CFG, RECORD)
    assert first["ok"] and first["attempts"] == 3 and not first["cached"]
    again = extract.extract_one(client, CFG, RECORD)
    assert again["cached"] and again["extraction"] == first["extraction"]
    assert client.calls == 3


def test_failure_after_max_attempts_is_not_cached():
    client = FakeClient([("max_tokens", GOOD)] * 3 + [("end_turn", GOOD)])
    assert not extract.extract_one(client, CFG, RECORD)["ok"]
    assert extract.extract_one(client, CFG, RECORD)["ok"]   # retried on the next run
    assert client.calls == 4


def test_method_families():
    assert method_family("SARIMA") == "ARIMA family"
    assert method_family("Long short-term memory network") == "LSTM / RNN"
    assert method_family("negative binomial regression") == "Regression (GLM/GAM/linear)"
    assert method_family("SEIR model") == "Compartmental (SIR/SEIR)"
    assert method_family("gradient-boosting machines") == "Other gradient boosting"
    assert method_family("interrupted time series analysis") == "Other time-series methods"
    assert method_family("RT-qPCR") == "Lab / genomic methods"
    assert method_family("Wavelet coherence") == "Other time-series methods"
    assert method_family("Delphi survey") == "Other"
    assert families(["ARIMA", "SARIMA", "XGBoost"]) == ["ARIMA family", "XGBoost"]


def test_gold_template_is_blank_and_never_overwritten(tmp_path):
    abstracts = pd.DataFrame({"pmid": [str(i) for i in range(20)], "title": "t", "abstract": "a"})
    gold = tmp_path / "gold.csv"
    sheet = tmp_path / "sheet.csv"
    assert write_gold_template(abstracts, gold, n=5, seed=0, sheet=sheet)
    df = pd.read_csv(gold, keep_default_na=False)
    assert len(df) == 5 and (df["study_type"] == "").all()
    assert "abstract" not in df.columns                                 # no abstract text committed
    assert "abstract" in pd.read_csv(sheet).columns                     # labeling sheet has it
    df.loc[0, "study_type"] = "review"
    df.to_csv(gold, index=False)
    assert not write_gold_template(abstracts, gold, n=5, seed=0)       # labels preserved
    assert pd.read_csv(gold, keep_default_na=False).loc[0, "study_type"] == "review"
    with pytest.raises(SystemExit):                                     # blanks remain -> refuse
        load_gold(gold)


def test_evaluate_metrics():
    gold = pd.DataFrame({"pmid": list("abcd"), "title": "t",
                         "study_type": ["forecasting", "forecasting", "review", "surveillance"]})
    pred = pd.DataFrame({"pmid": list("abcd"),
                         "study_type": ["forecasting", "surveillance", "review", "surveillance"]})
    res = evaluate(gold, pred)
    assert res["accuracy"] == 0.75
    assert res["per_class"].loc["forecasting", "recall"] == 0.5
    assert res["per_class"].loc["surveillance", "precision"] == 0.5
    assert res["confusion"].loc["forecasting", "surveillance"] == 1
    assert len(res["errors"]) == 1
