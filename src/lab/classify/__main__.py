"""Module 1: LLM extraction from PubMed abstracts, plus evaluation against human gold labels.

    python -m lab.classify template   create data/gold_labels.csv (blank labels) - no API calls
    python -m lab.classify run        extract fields with Claude (cached) -> outputs/classify/
    python -m lab.classify evaluate   score study_type against your filled-in gold labels
"""
from __future__ import annotations

import argparse
import logging
import sys

import anthropic
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from lab.classify.evaluate import evaluate, load_gold, write_gold_template  # noqa: E402
from lab.classify.extract import extract_all  # noqa: E402
from lab.classify.methods import families  # noqa: E402
from lab.classify.schema import STUDY_TYPES  # noqa: E402
from lab.config import ROOT, load_config, path, load_env  # noqa: E402
from lab.data.pubmed import fetch_abstracts  # noqa: E402
from lab.logging_setup import setup_logging  # noqa: E402

log = logging.getLogger("lab.classify")
GOLD = ROOT / "data" / "gold_labels.csv"


def _out():
    out = path("outputs") / "classify"
    out.mkdir(exist_ok=True)
    return out


def _md(df: pd.DataFrame) -> str:
    cols = [df.index.name or "", *map(str, df.columns)]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join([str(i), *map(str, r)]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join(rows)


def cmd_template() -> None:
    cfg = load_config()["classify"]
    created = write_gold_template(fetch_abstracts(), GOLD, cfg["gold_sample_size"], cfg["gold_seed"])
    log.info("%s %s", "Created" if created else "Kept existing (not overwritten)", GOLD)


def cmd_run() -> None:
    cfg = load_config()["classify"]
    abstracts = fetch_abstracts()
    try:
        results = extract_all(abstracts.to_dict("records"), cfg)
    except anthropic.AuthenticationError:
        sys.exit("ANTHROPIC_API_KEY is missing or invalid (set it in .env).")
    except anthropic.BadRequestError as e:
        sys.exit(f"Anthropic API rejected the request: {e.message}")

    ok = [r for r in results if r["ok"]]
    rows = []
    for r in ok:
        e = r["extraction"]
        rows.append({"pmid": r["pmid"], **{k: e[k] for k in ("study_type", "key_finding")},
                     "disease": "; ".join(e["disease"]), "methods": "; ".join(e["methods"]),
                     "method_families": "; ".join(families(e["methods"])),
                     "data_sources": "; ".join(e["data_sources"]), "geography": "; ".join(e["geography"])})
    ext = abstracts[["pmid", "year", "journal", "title", "publication_types"]].merge(
        pd.DataFrame(rows), on="pmid", how="inner")
    out = _out()
    ext.to_csv(out / "extractions.csv", index=False)
    pd.DataFrame([r for r in results if not r["ok"]], columns=["pmid", "attempts", "error"]).to_csv(
        out / "failures.csv", index=False)

    # Count of abstracts using each method family, by publication year.
    fam = ext.assign(family=ext["method_families"].str.split("; ")).explode("family")
    fam = fam[fam["family"].fillna("") != ""]
    by_year = pd.crosstab(fam["family"], fam["year"].astype(int), margins=True, margins_name="Total")
    by_year = by_year.sort_values("Total", ascending=False)
    by_year.to_csv(out / "methods_by_year.csv")
    types = pd.crosstab(ext["study_type"], ext["year"].astype(int), margins=True, margins_name="Total")
    types.to_csv(out / "study_type_by_year.csv")

    usage_in = sum(r["usage"]["input_tokens"] for r in ok)
    usage_out = sum(r["usage"]["output_tokens"] for r in ok)
    price = cfg["price_per_mtok"]
    cost = usage_in / 1e6 * price["input"] + usage_out / 1e6 * price["output"]
    n_new = sum(r["ok"] and not r["cached"] for r in results)
    retried = sum(r["ok"] and r["attempts"] > 1 for r in results)
    md = ["# Literature extraction summary\n",
          f"Model: `{cfg['model']}` · prompt `{cfg['prompt_version']}` · {len(abstracts)} abstracts · "
          f"{len(ok)} extracted ({retried} needed a retry) · {len(results) - len(ok)} failed after "
          f"{cfg['max_attempts']} attempts.\n",
          f"Tokens for the full extraction: {usage_in:,} input + {usage_out:,} output ≈ ${cost:.2f} at "
          f"${price['input']}/${price['output']} per M tokens. This run made {n_new} new API calls "
          f"(the rest came from the disk cache).\n",
          "## Study type by publication year\n", _md(types),
          "\n## Abstracts using each method family, by publication year\n",
          "(An abstract counts once per family; families come from regex mapping of the "
          "extracted method names in `lab/classify/methods.py`.)\n", _md(by_year)]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    cmd_template()


def cmd_evaluate() -> None:
    out = _out()
    gold = load_gold(GOLD)
    res = evaluate(gold, pd.read_csv(out / "extractions.csv", dtype={"pmid": str}))
    res["confusion"].to_csv(out / "confusion_matrix.csv")
    res["errors"].to_csv(out / "evaluation_errors.csv", index=False)
    _plot_confusion(res["confusion"], out / "confusion_matrix.png")
    pc = res["per_class"].round(3)
    pc["support"] = pc["support"].astype(int)
    md = ["# study_type evaluation vs human gold labels\n",
          f"Gold rows: {res['n_gold']} · scored: {res['n_scored']} · missing prediction: "
          f"{res['n_missing_prediction']}\n",
          f"- **Accuracy:** {res['accuracy']:.3f}",
          f"- **Macro F1** (classes present in gold): {res['macro_f1']:.3f}",
          f"- **Cohen's kappa:** {res['kappa']:.3f}\n",
          "## Per-class precision / recall\n", _md(pc),
          "\n## Confusion matrix (rows = gold, columns = predicted)\n", _md(res["confusion"]),
          f"\nDisagreements: {len(res['errors'])} (see `evaluation_errors.csv`)."]
    (out / "evaluation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def _plot_confusion(cm: pd.DataFrame, out) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(STUDY_TYPES)), STUDY_TYPES, rotation=35, ha="right")
    ax.set_yticks(range(len(STUDY_TYPES)), STUDY_TYPES)
    vmax = cm.values.max() or 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.iat[i, j]
            ax.text(j, i, v, ha="center", va="center", color="white" if v > vmax / 2 else "#0b0b0b")
    ax.set_xlabel("Predicted (Claude)")
    ax.set_ylabel("Gold (human)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["template", "run", "evaluate"])
    args = parser.parse_args()
    setup_logging()
    load_env()
    {"template": cmd_template, "run": cmd_run, "evaluate": cmd_evaluate}[args.cmd]()


if __name__ == "__main__":
    main()
