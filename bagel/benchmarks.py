"""
Loads benchmark files (EL_benchmarks truth.jsonl: medmentions_assessed, craft) into run items:

    {"id", "document": {"id", "text"}, "mention": {"text"}, "abbreviations": {...},
     "judgments": [{"curie", "verdict", ...}], "closed_world": bool, "proposed_by": {...}}
"""
import json
from pathlib import Path

from bagel.abbreviations import load_abbreviation_maps


def _from_truth(record: dict) -> dict:
    document_id = record.get("pmid") or record.get("article")
    return {
        "id": record["id"],
        "document": {"id": str(document_id), "text": record["text"]},
        "mention": {"text": record["mention"]},
        "judgments": record["judgments"],
        "closed_world": record["closed_world"],
        "mention_type": record.get("mention_type"),
    }


def _from_assessed_dump(record: dict) -> dict:
    """Items written by `bagel dataset` from the old bagel-eval dump (gold as {curie: verdict})."""
    return {
        "id": record["id"],
        "document": record["document"],
        "mention": record["mention"],
        "abbreviations": record.get("abbreviations", {}),
        "judgments": [{"curie": c, "verdict": v} for c, v in record["gold"].items() if v in ("correct", "incorrect")],
        "closed_world": False,
    }


def load_benchmark(path: str | Path, abbreviations: str | Path | None = None, limit: int | None = None) -> list[dict]:
    """Benchmark items; `abbreviations` is an optional JSONL of {"pmid", "abbreviation_map"} per document."""
    with open(path) as stream:
        records = [json.loads(line) for line in stream]
    items = [_from_truth(r) if "judgments" in r else _from_assessed_dump(r) for r in records]
    ids = [i["id"] for i in items]
    if len(set(ids)) != len(ids):
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"{path} has repeated item ids, e.g. {repeated[:3]}")
    if abbreviations:
        maps = load_abbreviation_maps(abbreviations)
        for item in items:
            item["abbreviations"] = maps.get(item["document"]["id"], {})
    for item in items:
        item.setdefault("abbreviations", {})
    return items[:limit] if limit else items
