"""Fetch PubMed abstracts via NCBI E-utilities (esearch + efetch). Free; key optional.

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
"""
from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from lab.config import load_config, path

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _params(**extra) -> dict:
    p = {"db": "pubmed", "tool": "health-signals-lab", **extra}
    if email := os.environ.get("NCBI_EMAIL"):
        p["email"] = email
    if key := os.environ.get("NCBI_API_KEY"):
        p["api_key"] = key
    return p


def _pause() -> None:
    # NCBI allows 3 requests/s without a key, 10/s with one.
    time.sleep(0.11 if os.environ.get("NCBI_API_KEY") else 0.35)


def esearch(query: str, reldate: int, retmax: int, sort: str) -> list[str]:
    resp = requests.get(f"{EUTILS}/esearch.fcgi", params=_params(
        term=query, datetype="pdat", reldate=reldate, retmax=retmax, sort=sort, retmode="json",
    ), timeout=60)
    resp.raise_for_status()
    result = resp.json()["esearchresult"]
    log.info("esearch: %s matches; taking %d PMIDs", result["count"], len(result["idlist"]))
    _pause()
    return result["idlist"]


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_article(art: ET.Element) -> dict:
    cit = art.find("MedlineCitation")
    a = cit.find("Article")
    parts = []
    for ab in a.findall("Abstract/AbstractText"):
        label = ab.get("Label")
        parts.append(f"{label}: {_text(ab)}" if label else _text(ab))
    pub = a.find("Journal/JournalIssue/PubDate")
    year = _text(pub.find("Year")) if pub is not None else ""
    if not year and pub is not None:
        year = _text(pub.find("MedlineDate"))[:4]
    return {
        "pmid": _text(cit.find("PMID")),
        "year": int(year) if year.isdigit() else None,
        "journal": _text(a.find("Journal/Title")),
        "title": _text(a.find("ArticleTitle")),
        "abstract": "\n".join(p for p in parts if p),
        "publication_types": "; ".join(_text(t) for t in a.findall("PublicationTypeList/PublicationType")),
    }


def efetch(pmids: list[str], batch: int = 100) -> list[dict]:
    records = []
    for i in range(0, len(pmids), batch):
        resp = requests.post(f"{EUTILS}/efetch.fcgi", data=_params(
            id=",".join(pmids[i : i + batch]), retmode="xml"), timeout=120)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        records += [parse_article(a) for a in root.findall("PubmedArticle")]
        _pause()
    return records


def fetch_abstracts(refresh: bool = False) -> pd.DataFrame:
    """Top-N relevant abstracts for the configured query, cached to data/raw/pubmed_abstracts.csv."""
    out = path("raw") / "pubmed_abstracts.csv"
    if out.exists() and not refresh:
        log.info("Using cached %s", out)
        return pd.read_csv(out, dtype={"pmid": str})
    cfg = load_config()["pubmed"]
    # Over-fetch: some records have no abstract and are skipped.
    pmids = esearch(cfg["query"], cfg["reldate_days"], int(cfg["n_abstracts"] * 1.3), cfg["sort"])
    df = pd.DataFrame(efetch(pmids))
    df["rank"] = df["pmid"].map({p: i for i, p in enumerate(pmids)})
    df = df[df["abstract"].str.len() > 0].sort_values("rank").head(cfg["n_abstracts"])
    df = df.drop(columns="rank").reset_index(drop=True)
    df.to_csv(out, index=False)
    log.info("Saved %d abstracts (%d-%d) -> %s", len(df), df["year"].min(), df["year"].max(), out)
    return df
