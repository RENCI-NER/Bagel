"""Stage 2 (retrieve): candidate CURIEs from NameRes and SapBERT, normalized with NodeNorm."""
import math

import httpx
from pydantic import BaseModel

from bagel.cache import Cache, cache_key
from bagel.types import Candidate


class RetrievalConfig(BaseModel):
    name_res_url: str = "https://name-resolution-sri.renci.org/lookup"
    sapbert_url: str = "https://sap-qdrant.apps.renci.org/annotate/"
    nodenorm_url: str = "https://nodenormalization-sri.renci.org/get_normalized_nodes"
    count: int = 10  # results requested from each retriever
    timeout: float = 60.0


def nameres_lookup(client: httpx.Client, config: RetrievalConfig, text: str, biolink_type: str | None) -> list[dict]:
    params = {"string": text, "autocomplete": "false", "offset": 0, "limit": config.count}
    if biolink_type:
        params["biolink_type"] = biolink_type
    response = client.get(config.name_res_url, params=params)
    response.raise_for_status()
    return response.json()


def sapbert_lookup(client: httpx.Client, config: RetrievalConfig, text: str, biolink_type: str | None) -> list[dict]:
    payload = {"text": text, "model_name": "sapbert", "count": config.count}
    if biolink_type:
        payload["args"] = {"bl_type": biolink_type}
    response = client.post(config.sapbert_url, json=payload)
    response.raise_for_status()
    return response.json()


def nodenorm(client: httpx.Client, config: RetrievalConfig, curies: list[str]) -> dict:
    payload = {"curies": curies, "conflate": True, "drug_chemical_conflate": True, "description": True,
               "include_taxa": True}
    response = client.post(config.nodenorm_url, json=payload)
    response.raise_for_status()
    return response.json()


def merge_hits(nameres_hits: list[dict], sapbert_hits: list[dict]) -> dict[str, dict]:
    """Merges retriever hits by CURIE, keeping each source's rank (1-based) and score."""
    merged = {}
    for rank, hit in enumerate(nameres_hits, 1):
        entry = merged.setdefault(hit["curie"], {"label": hit["label"], "category": hit["types"][0]})
        entry.setdefault("name_res_rank", rank)
        entry.setdefault("name_res_score", hit["score"])
        entry["taxa_ids"] = hit.get("taxa", [])
    for rank, hit in enumerate(sapbert_hits, 1):
        entry = merged.setdefault(hit["curie"], {"label": hit["name"], "category": hit["category"]})
        entry.setdefault("sapbert_rank", rank)
        entry.setdefault("sapbert_score", hit["score"])
    return merged


def normalize_candidates(merged: dict[str, dict], normalized: dict, taxon_labels: dict[str, str]) -> list[Candidate]:
    """
    Turns merged hits into candidates keyed by NodeNorm-preferred CURIE. Retrieved CURIEs that NodeNorm doesn't know
    are kept under their own CURIE. CURIEs that normalize to the same clique are combined, keeping each source's best
    rank. Candidates are ordered by best rank from either source, then name_res rank, sapbert rank and CURIE, so the
    order is deterministic.
    """
    candidates: dict[str, Candidate] = {}
    for curie, hit in merged.items():
        node = normalized.get(curie)
        if node:
            identity = node["id"]
            taxa_ids = hit.get("taxa_ids") or node.get("taxa") or []
            candidate = Candidate(
                curie=identity["identifier"],
                label=identity.get("label") or hit["label"],
                category=node["type"][0],
                description=identity.get("description", ""),
                taxa_ids=taxa_ids,
            )
        else:
            taxa_ids = hit.get("taxa_ids", [])
            candidate = Candidate(curie=curie, label=hit["label"], category=hit["category"], taxa_ids=taxa_ids)
        candidate.taxa = [taxon_labels[t] for t in taxa_ids if taxon_labels.get(t)]
        candidate.source_curies = [curie]
        for source in ("name_res", "sapbert"):
            setattr(candidate, f"{source}_rank", hit.get(f"{source}_rank"))
            setattr(candidate, f"{source}_score", hit.get(f"{source}_score"))
        existing = candidates.get(candidate.curie)
        if existing is None:
            candidates[candidate.curie] = candidate
        else:
            _combine(existing, candidate)
    return sorted(candidates.values(), key=_order_key)


def _combine(target: Candidate, other: Candidate) -> None:
    for source in ("name_res", "sapbert"):
        other_rank = getattr(other, f"{source}_rank")
        target_rank = getattr(target, f"{source}_rank")
        if other_rank is not None and (target_rank is None or other_rank < target_rank):
            setattr(target, f"{source}_rank", other_rank)
            setattr(target, f"{source}_score", getattr(other, f"{source}_score"))
    target.source_curies += [c for c in other.source_curies if c not in target.source_curies]
    if not target.taxa_ids:
        target.taxa_ids, target.taxa = other.taxa_ids, other.taxa


def _order_key(candidate: Candidate):
    missing = math.inf
    name_res = candidate.name_res_rank if candidate.name_res_rank is not None else missing
    sapbert = candidate.sapbert_rank if candidate.sapbert_rank is not None else missing
    return min(name_res, sapbert), name_res, sapbert, candidate.curie


def taxon_labels(client: httpx.Client, config: RetrievalConfig, taxa_ids: set[str]) -> dict[str, str]:
    if not taxa_ids:
        return {}
    normalized = nodenorm(client, config, sorted(taxa_ids))
    return {t: node["id"].get("label", "") for t, node in normalized.items() if node}


def retrieve(client: httpx.Client, config: RetrievalConfig, query: str, biolink_type: str | None = None,
             cache: Cache | None = None) -> list[Candidate]:
    key = cache_key("retrieve", config.model_dump(exclude={"timeout"}), query, biolink_type)
    if cache is not None and (cached := cache.get(key)) is not None:
        return [Candidate(**c) for c in cached]
    merged = merge_hits(nameres_lookup(client, config, query, biolink_type),
                        sapbert_lookup(client, config, query, biolink_type))
    normalized = nodenorm(client, config, list(merged)) if merged else {}
    taxa = {t for hit in merged.values() for t in hit.get("taxa_ids", [])}
    taxa |= {t for node in normalized.values() if node for t in node.get("taxa", [])}
    candidates = normalize_candidates(merged, normalized, taxon_labels(client, config, taxa))
    if cache is not None:
        cache.set(key, [c.model_dump() for c in candidates])
    return candidates
