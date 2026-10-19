"""Map free-text method names from the LLM onto a small set of method families for counting.

First matching pattern wins, so more specific families come first. Unmatched names -> "Other".
"""
from __future__ import annotations

import re

METHOD_FAMILIES: list[tuple[str, str]] = [
    ("LSTM / RNN", r"lstm|long short|\bgru\b|recurrent|\brnn\b"),
    ("Transformer", r"transformer|attention|informer|\btft\b"),
    ("XGBoost", r"xgboost"),
    ("LightGBM", r"lightgbm|light gbm"),
    ("Other gradient boosting", r"gradient[- ]boost|\bgbm\b|catboost|adaboost|boosted"),
    ("Random forest", r"random forest"),
    ("Other neural network", r"neural|deep learning|\bcnn\b|\bmlp\b|convolutional"),
    ("ARIMA family", r"arima|autoregressive integrated|box-jenkins"),
    ("Exponential smoothing", r"exponential smoothing|holt|\bets\b"),
    ("Prophet", r"prophet"),
    ("Compartmental (SIR/SEIR)", r"\b(sir|seir|sirs|seirs|sis|sei)\b|compartmental|mechanistic|metapopulation"),
    ("Bayesian model", r"bayes|markov chain monte carlo|\bmcmc\b"),
    ("Ensemble", r"ensemble"),
    ("Regression (GLM/GAM/linear)",
     r"regression|\bglm|generalized linear|generalised linear|poisson|negative binomial|\bgam\b|additive model"),
    ("Other machine learning", r"machine learning|\bsvm\b|support vector|nearest neighbo|\bknn\b|transfer learning|language model"),
    ("Other time-series methods",
     r"interrupted time series|\bvar\b|vector autoreg|kalman|wavelet|decomposition|\bstl\b|moving average"
     r"|distributed lag|moving epidemic method|time series"),
    ("Spatial analysis", r"spatial|hotspot|hot spot|geographic|cluster|standard deviation(al)? ellipse"),
    ("Lab / genomic methods", r"pcr|sequenc|phylogen|genom|evolutionary analysis|antigenic"),
    ("Statistical tests / descriptive",
     r"chi-square|chi square|fisher|spearman|pearson|correlation|t-test|descriptive|principal component|\bpca\b"),
]
_COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in METHOD_FAMILIES]


def method_family(name: str) -> str:
    for family, pattern in _COMPILED:
        if pattern.search(name):
            return family
    return "Other"


def families(methods: list[str]) -> list[str]:
    """Distinct families for one abstract's method list, in first-seen order."""
    return list(dict.fromkeys(method_family(m) for m in methods))
