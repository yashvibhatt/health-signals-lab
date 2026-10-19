"""LLM extraction: one Claude call per abstract, structured JSON output, validation, retry, disk cache.

Cache: data/cache/llm/<sha256(model, prompt_version, pmid, abstract)>.json holds the validated
extraction plus token usage, so reruns make no API calls and cost nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from pydantic import ValidationError

from lab.classify.schema import STUDY_TYPE_DEFINITIONS, Extraction
from lab.config import ROOT

log = logging.getLogger(__name__)

CACHE_DIR = ROOT / "data" / "cache" / "llm"

SYSTEM_PROMPT = (
    "You extract structured information from biomedical abstracts about respiratory virus "
    "surveillance and forecasting. Use only what the title and abstract state; never infer "
    "details that are not there, and use an empty list when a field is not stated.\n\n"
    "study_type: choose the single category that best describes the study's PRIMARY aim:\n"
    + "\n".join(f"- {k}: {v}" for k, v in STUDY_TYPE_DEFINITIONS.items())
    + "\n\nmethods: short standard names (e.g. 'ARIMA', 'SARIMA', 'LSTM', 'XGBoost', 'LightGBM', "
    "'random forest', 'SEIR', 'Bayesian hierarchical model', 'ensemble'), not descriptions.\n"
    "key_finding: one sentence."
)


def cache_key(model: str, prompt_version: str, pmid: str, abstract: str) -> str:
    raw = json.dumps([model, prompt_version, SYSTEM_PROMPT, pmid, abstract])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def user_message(title: str, abstract: str) -> str:
    return f"Title: {title}\n\nAbstract:\n{abstract}"


def call_claude(client: anthropic.Anthropic, model: str, max_tokens: int, title: str,
                abstract: str) -> tuple[Extraction, dict]:
    """One structured-output request. Raises ValueError on refusal / truncation."""
    resp = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message(title, abstract)}],
        output_format=Extraction,
    )
    if resp.stop_reason != "end_turn":
        raise ValueError(f"stop_reason={resp.stop_reason}")
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    # Re-validate explicitly so our validators (one-sentence finding, list cleaning) always run.
    return Extraction.model_validate(resp.parsed_output.model_dump()), usage


def extract_one(client, cfg: dict, record: dict) -> dict:
    """Cached extraction for one abstract, retrying on invalid output."""
    key = cache_key(cfg["model"], cfg["prompt_version"], record["pmid"], record["abstract"])
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return {**json.loads(cache_file.read_text()), "cached": True}

    last_error = None
    for attempt in range(1, cfg["max_attempts"] + 1):
        try:
            ext, usage = call_claude(client, cfg["model"], cfg["max_tokens"],
                                     record["title"], record["abstract"])
            result = {"pmid": record["pmid"], "ok": True, "attempts": attempt,
                      "extraction": ext.model_dump(), "usage": usage, "model": cfg["model"]}
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result, indent=1))
            return {**result, "cached": False}
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
            log.warning("PMID %s attempt %d failed: %s", record["pmid"], attempt, last_error)
    # Failures are not cached, so the next run retries them.
    return {"pmid": record["pmid"], "ok": False, "attempts": cfg["max_attempts"],
            "error": last_error, "cached": False}


def extract_all(records: list[dict], cfg: dict, client=None) -> list[dict]:
    client = client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        results = list(pool.map(lambda r: extract_one(client, cfg, r), records))
    n_cached = sum(r["cached"] for r in results)
    n_fail = sum(not r["ok"] for r in results)
    log.info("Extracted %d abstracts (%d from cache, %d new API results, %d failed)",
             len(results), n_cached, len(results) - n_cached - n_fail, n_fail)
    return results
