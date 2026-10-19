"""Gold-label template + evaluation of LLM study_type against human labels.

The template never contains model predictions (labels must be made blind), and evaluation
refuses to run until every row has a human label. No gold labels are ever generated here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support

from lab.classify.schema import STUDY_TYPE_DEFINITIONS, STUDY_TYPES

# The committed gold file carries no abstract text (publisher copyright); abstracts for labeling
# go to a git-ignored sheet in data/raw/ and can always be re-fetched with `make data`.
GOLD_COLUMNS = ["pmid", "title", "study_type", "labeler_notes"]


def write_gold_template(abstracts: pd.DataFrame, out: Path, n: int, seed: int,
                        sheet: Path | None = None) -> bool:
    """Create the blank gold file (pmid, title, labels) and, if `sheet` is given, a labeling sheet
    that also shows each abstract. Never overwrites an existing gold file (it may hold labels)."""
    if out.exists():
        return False
    sample = abstracts.sample(n=n, random_state=seed).sort_values("pmid")
    template = sample[["pmid", "title", "abstract"]].assign(study_type="", labeler_notes="")
    template[GOLD_COLUMNS].to_csv(out, index=False)
    if sheet is not None:
        template.to_csv(sheet, index=False)
    guide = ["# Labeling guide for data/gold_labels.csv\n",
             f"Fill `study_type` for all {n} rows with exactly one of: "
             + ", ".join(f"`{t}`" for t in STUDY_TYPES) + ".",
             "Read the abstracts in `data/raw/gold_labeling_sheet.csv` (created by `make gold-template`; "
             "not committed). Label from the title and abstract only, choosing the study's PRIMARY aim. "
             "Use `labeler_notes` for borderline cases. Do not look at model outputs while labeling.\n"]
    guide += [f"- **{k}**: {v}" for k, v in STUDY_TYPE_DEFINITIONS.items()]
    (out.parent / "LABELING_GUIDE.md").write_text("\n".join(guide) + "\n")
    return True


def load_gold(path: Path) -> pd.DataFrame:
    gold = pd.read_csv(path, dtype={"pmid": str}, keep_default_na=False)
    gold["study_type"] = gold["study_type"].str.strip().str.lower()
    blank = gold["study_type"] == ""
    if blank.any():
        raise SystemExit(f"{blank.sum()} of {len(gold)} rows in {path} have no study_type label yet. "
                         "Label every row before running evaluate.")
    bad = sorted(set(gold["study_type"]) - set(STUDY_TYPES))
    if bad:
        raise SystemExit(f"Unknown labels {bad}; allowed: {STUDY_TYPES}")
    return gold


def evaluate(gold: pd.DataFrame, extractions: pd.DataFrame) -> dict:
    """Accuracy, per-class precision/recall/F1, confusion matrix, Cohen's kappa."""
    merged = gold[["pmid", "title", "study_type"]].merge(
        extractions[["pmid", "study_type"]].rename(columns={"study_type": "predicted"}),
        on="pmid", how="left")
    missing = merged["predicted"].isna()
    scored = merged[~missing]
    y_true, y_pred = scored["study_type"], scored["predicted"]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=STUDY_TYPES, zero_division=0)
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=STUDY_TYPES)
    cm = pd.crosstab(pd.Categorical(y_true, STUDY_TYPES), pd.Categorical(y_pred, STUDY_TYPES),
                     rownames=["gold"], colnames=["predicted"], dropna=False)
    return {
        "n_gold": len(gold), "n_scored": len(scored), "n_missing_prediction": int(missing.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": float(per_class.loc[per_class["support"] > 0, "f1"].mean()),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "per_class": per_class, "confusion": cm,
        "errors": scored[y_true != y_pred].rename(columns={"study_type": "gold"}),
    }
