"""
The human-assessed MedMentions mentions from the BagelWithCheese evaluation app (bagel-eval), as a dataset.

Reads a pg_dump of that app's database (tables abstracts, recognized_entities, assessment, results) and an
abbreviation file ({"pmid", "abbreviation_map"} per line). Writes one JSON line per assessed mention:

    {"id", "document": {"id", "text"}, "mention": {"text"}, "abbreviations": {...},
     "gold": {curie: "correct" | "incorrect" | "unsure"}, "medmentions": curie | null,
     "baselines": {model: curie | null}, "bwc_expanded_text": str}

Assessor names are not copied: each (mention, CURIE) gets one verdict from its agree/disagree counts. CURIEs are
renormalized with the current NodeNorm, since preferred identifiers change between Babel releases.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from bagel.abbreviations import load_abbreviation_maps
from bagel.retrieval import RetrievalConfig, nodenorm

TABLES = ("abstracts", "recognized_entities", "assessment", "results")


def _unescape(field: str):
    if field == r"\N":
        return None
    out, i = [], 0
    escapes = {"t": "\t", "n": "\n", "r": "\r", "\\": "\\"}
    while i < len(field):
        if field[i] == "\\" and i + 1 < len(field) and field[i + 1] in escapes:
            out.append(escapes[field[i + 1]])
            i += 2
        else:
            out.append(field[i])
            i += 1
    return "".join(out)


def read_copy_tables(path: str | Path, tables=TABLES) -> dict[str, list[dict]]:
    """Reads the COPY ... FROM stdin blocks of the given tables from a plain-text pg_dump."""
    rows = {t: [] for t in tables}
    current, columns = None, None
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if current is None:
                if line.startswith("COPY public."):
                    name = line[len("COPY public."):].split(" ", 1)[0]
                    if name in rows:
                        current = name
                        columns = [c.strip() for c in line[line.index("(") + 1:line.index(")")].split(",")]
                continue
            if line.rstrip("\n") == r"\.":
                current = None
                continue
            fields = [_unescape(f) for f in line.rstrip("\n").split("\t")]
            rows[current].append(dict(zip(columns, fields)))
    return rows


def verdict(counts: Counter) -> str:
    if counts["agree"] > counts["disagree"]:
        return "correct"
    if counts["disagree"] > counts["agree"]:
        return "incorrect"
    return "unsure"


def build(dump_path: str | Path, abbreviations_path: str | Path, out_path: str | Path,
          nodenorm_config: RetrievalConfig | None = None) -> int:
    tables = read_copy_tables(dump_path)
    abstracts = {r["pmid"]: r["abstract"] for r in tables["abstracts"]}
    mentions = {r["id"]: r for r in tables["recognized_entities"]}
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for row in tables["assessment"]:
        counts[row["idx"]][row["identifier"]][row["assessment"]] += 1
    results: dict[str, dict[str, str | None]] = defaultdict(dict)
    for row in tables["results"]:
        if row["idx"] in counts:
            results[row["idx"]][row["model"]] = row["identifier"]
    abbreviations = load_abbreviation_maps(abbreviations_path)

    curies = {c for idx in counts for c in counts[idx]} | {c for r in results.values() for c in r.values() if c}
    config = nodenorm_config or RetrievalConfig()
    preferred = {}
    with httpx.Client(timeout=config.timeout) as client:
        ordered = sorted(curies)
        for start in range(0, len(ordered), 500):
            batch = ordered[start:start + 500]
            normalized = nodenorm(client, config, batch)
            preferred.update({c: (normalized.get(c) or {}).get("id", {}).get("identifier", c) for c in batch})

    written = 0
    with open(out_path, "w") as out:
        for idx in sorted(counts, key=int):
            mention = mentions[idx]
            merged: dict[str, Counter] = defaultdict(Counter)
            for curie, c in counts[idx].items():
                merged[preferred[curie]].update(c)
            gold = {curie: verdict(c) for curie, c in merged.items()}
            model_results = {m: (preferred[c] if c else None) for m, c in results[idx].items()}
            record = {
                "id": f"mm-{idx}",
                "document": {"id": mention["pmid"], "text": abstracts[mention["pmid"]]},
                "mention": {"text": mention["original_text"]},
                "abbreviations": abbreviations.get(mention["pmid"], {}),
                "gold": gold,
                "medmentions": model_results.pop("medmentions", None),
                "baselines": model_results,
                "bwc_expanded_text": mention["expanded_text"],
            }
            out.write(json.dumps(record) + "\n")
            written += 1
    return written
