"""Structured fields extracted from each abstract (validated with pydantic)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

StudyType = Literal["forecasting", "surveillance", "clinical", "review", "other"]
STUDY_TYPES: list[str] = list(StudyType.__args__)

# Shared by the LLM prompt and the human labeling guide, so both use the same definitions.
STUDY_TYPE_DEFINITIONS = {
    "forecasting": "Primary aim is predicting future (or nowcasting current, not-yet-reported) "
                   "disease activity, e.g. building or evaluating a forecasting/nowcasting model.",
    "surveillance": "Primary aim is describing, monitoring or detecting disease activity from "
                    "surveillance data (trends, burden, seasonality, early warning, system evaluation) "
                    "without forecasting as the main contribution.",
    "clinical": "Patient-level clinical research: treatment, vaccine efficacy/effectiveness trials, "
                "diagnosis, outcomes or risk factors in patients.",
    "review": "Narrative, systematic or scoping review, or meta-analysis of existing literature.",
    "other": "Anything else (e.g. lab/virology, genomics, transmission modelling without forecasting, "
             "policy, methods unrelated to the categories above).",
}


class Extraction(BaseModel):
    study_type: StudyType = Field(description="Single best category for the study's primary aim.")
    disease: list[str] = Field(description="Diseases studied, e.g. 'influenza', 'RSV', 'COVID-19'.")
    data_sources: list[str] = Field(
        description="Data sources used, e.g. 'ILINet', 'FluSurv-NET', 'Google Trends', 'EHR'. "
                    "Empty list if none stated.")
    methods: list[str] = Field(
        description="Analytic/model methods by short standard name, e.g. 'ARIMA', 'LSTM', 'LightGBM', "
                    "'random forest', 'SEIR'. Empty list if none stated.")
    geography: list[str] = Field(
        description="Countries/regions studied, e.g. 'United States', 'China', 'global'. "
                    "Empty list if not stated.")
    key_finding: str = Field(description="The main finding in one sentence, using only the abstract.")

    @field_validator("key_finding")
    @classmethod
    def one_sentence(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 400:
            raise ValueError("key_finding must be a single sentence (1-400 characters)")
        return v

    @field_validator("disease", "data_sources", "methods", "geography")
    @classmethod
    def clean_list(cls, v: list[str]) -> list[str]:
        seen, out = set(), []
        for item in (s.strip() for s in v):
            if item and item.lower() not in seen:
                seen.add(item.lower())
                out.append(item)
        return out
