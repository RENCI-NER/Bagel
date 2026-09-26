from httpx import AsyncClient
import json
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from logutil import LoggingUtil
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt

logger = LoggingUtil.init_logging(name=__name__)


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
async def get_sapbert_ids(entity: str, session: AsyncClient, count: int = 10, entity_type: str = None, url: str = None):
    payload = {
        "text": f"{entity}",
        "model_name": "sapbert",
        "count": count
    }
    if entity_type:
        payload["args"] = {
            "bl_type": entity_type
        }
    response = await session.post(url, json=payload)
    reformatted = {}
    if response.status_code == 200:
        sapbert_json = response.json()
        for index, value in enumerate(sapbert_json):
            reformatted[value["curie"]] = reformatted.get(value["curie"], {})
            # merge similar queries ...
            # more merging if normalization brings them together too, in the normalization section ahead.
            reformatted[value["curie"]].update({
                "name": value["name"],
                "sapbert_score": value["score"],
                "sapbert_rank": index + 1,
                "category": value["category"]
            })
    else:
        logger.error(f"Error: sapbert call for {entity} , payload: {payload} returned code: {response.status_code}")
        raise Exception(f"Error: sapbert call for {entity} , payload: {payload} returned code: {response.status_code}")
    return reformatted


def build_nameres_url(url: str, entity: str, count: int = 10, entity_type: str = None):
    """
    Builds a NameRes lookup URL. `url` is the configured base, conventionally ending in "string=" so the entity is
    appended; any string/limit/biolink_type already in it is replaced so the entity is URL-encoded (e.g. "Na+/K+",
    "T&E") and limit is not duplicated.
    """
    scheme, netloc, path, query, fragment = urlsplit(url)
    params = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True)
              if k not in ("string", "limit", "biolink_type")]
    params += [("string", entity), ("limit", str(count))]
    if entity_type:
        params.append(("biolink_type", entity_type))
    return urlunsplit((scheme, netloc, path, urlencode(params), fragment))


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
async def get_nameres_ids(entity: str, session: AsyncClient, count: int = 10, entity_type: str = None, url: str = None, nn_url: str = None):
    name_res_url = build_nameres_url(url, entity, count=count, entity_type=entity_type)
    response = await session.get(name_res_url)
    if response.status_code == 200:
        json_data = response.json()
        formatted = {
            f"{x['curie']}": {
                "name": x["label"],
                "name_res_rank": index + 1,
                "name_res_score": x["score"],
                "taxa": ", ".join(
                            [value for key, value in (await get_taxa_information(x["taxa"], nn_url)).items() if value]
                        ),
                "taxa_id": x["taxa"],
                "category": x["types"][0]
            } for index, x in enumerate(json_data)
        }
    else:
        logger.error(f"Error: nameres call for {entity} , url: {name_res_url} returned code: {response.status_code}")
        raise Exception(f"Error: nameres call for {entity}")
    return formatted

@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
async def get_entity_ids(entity: str, name_res_url: str, sapbert_url: str, node_norm_url: str, session: AsyncClient,
                         entity_type=None, count=10, nameres_count=None, sapbert_count=None):
    get_entities_tasks = [
        get_sapbert_ids(entity=entity,
                        entity_type=entity_type,
                        url=sapbert_url,
                        count=sapbert_count or count,
                        session=session
                        ),
        get_nameres_ids(entity=entity,
                        entity_type=entity_type,
                        url=name_res_url,
                        nn_url=node_norm_url,
                        count=nameres_count or count,
                        session=session
                        )
    ]
    responses = await asyncio.gather(*get_entities_tasks)
    logger.debug(f"Combined responses: {json.dumps(responses, indent=2)}")

    # merge them by identifier
    merged = {}
    for response in responses:
        for curie, value in response.items():
            merged.setdefault(curie, {
                "name_res_rank": -1,
                "sapbert_rank": -1,
                "name_res_score": -1,
                "sapbert_score": -1,
            }).update(value)
    # Fix the candidate order (and so the order/color codes the LLM sees) independent of set/dict ordering or
    # which service answered first: best rank from either source, then name_res rank, sapbert rank, curie.
    merged = dict(sorted(merged.items(), key=lambda kv: candidate_sort_key(kv[0], kv[1])))
    logger.debug(f"Merged: {json.dumps(merged, indent=2)}")

    nodenorm_payload = {
        "curies": list(merged.keys()),
        "conflate": True,
        "description": True,
        "drug_chemical_conflate": True,
        "include_taxa": True
    }
    normalized_curies = await session.post(node_norm_url, json=nodenorm_payload)
    final_results = {}
    if normalized_curies.status_code == 200:
        normalized_response_json = normalized_curies.json()
        for curie, entry in merged.items():
            normalized_identifier = normalized_response_json.get(curie)
            entry = dict(entry, source_curies=[curie])
            # make sure that we actually have normalized it
            if normalized_identifier:
                norm_id = normalized_identifier["id"]["identifier"]
                norm_label = normalized_identifier["id"].get("label", "")
                entry.update({
                    "description": normalized_identifier["id"].get("description", ""),
                    "category": normalized_identifier["type"][0],
                })
                if norm_label:
                    entry["name"] = norm_label
                # sapbert doesn't return taxa; fill them from node norm so e.g. human and mouse genes are
                # distinguishable to the LLM regardless of which service found them
                if not entry.get("taxa_id") and normalized_identifier.get("taxa"):
                    entry["taxa_id"] = normalized_identifier["taxa"]
                    entry["taxa"] = ", ".join(
                        label for label in (await get_taxa_information(entry["taxa_id"], node_norm_url)).values()
                        if label)
            else:
                # For some reason if these are not normalizing let's just add them into the final list
                logger.debug(f"Could not normalize {curie}, adding to the final results: {entry}")
                norm_id = curie
            if norm_id in final_results:
                # several source curies normalized to the same clique: combine rather than overwrite
                merge_candidates(final_results[norm_id], entry)
            else:
                final_results[norm_id] = entry
        # merging can improve a clique's best rank, so re-sort
        final_results = dict(sorted(final_results.items(), key=lambda kv: candidate_sort_key(kv[0], kv[1])))
    return final_results


RANK_SCORE_FIELDS = [("name_res_rank", "name_res_score"), ("sapbert_rank", "sapbert_score")]


def candidate_sort_key(curie: str, entry: dict):
    """ Sort key for candidates: best rank from any source first; ties broken by name_res rank, sapbert rank,
    then curie. Missing ranks (-1) sort last. """
    ranks = [entry.get(rank_field, -1) for rank_field, _ in RANK_SCORE_FIELDS]
    ranks = [r if r and r > 0 else float("inf") for r in ranks]
    return (min(ranks), *ranks, curie)


def merge_candidates(target: dict, other: dict):
    """
    Merges candidate `other` into `target` in place (both normalized to the same identifier). For each source the
    better (lower) rank is kept together with its score; the source curies are accumulated and fields missing on
    target (e.g. taxa from name_res when target only came from sapbert) are filled from other.
    """
    for rank_field, score_field in RANK_SCORE_FIELDS:
        other_rank = other.get(rank_field, -1)
        target_rank = target.get(rank_field, -1)
        if other_rank > 0 and (target_rank <= 0 or other_rank < target_rank):
            target[rank_field] = other_rank
            target[score_field] = other.get(score_field, -1)
    target["source_curies"] = target.get("source_curies", []) + [
        c for c in other.get("source_curies", []) if c not in target.get("source_curies", [])]
    for key, value in other.items():
        if value and not target.get(key):
            target[key] = value
    return target


taxa_cache = {}

async def get_taxa_information(taxa_ids: list, node_norm_url: str):
    # Determine which taxa_ids are missing from the cache
    missing_ids = [tid for tid in taxa_ids if tid not in taxa_cache]

    if missing_ids:
        async with AsyncClient() as client:
            payload = {
                "curies": missing_ids,
                "conflate": True,
                "description": True,
                "drug_chemical_conflate": True
            }
            response = await client.post(node_norm_url, json=payload)
            if response.status_code == 200:
                response_json = response.json()
                # Update cache with new results
                for key, value in response_json.items():
                    if value:
                        taxa_cache[key] = value.get('id', {}).get('label', '')
                    else:
                        logger.debug("Could not get taxa {key} from node norm".format(key=key))
            else:
                # Optionally, handle non-200 responses
                pass


    # Build the result from the cache
    return {tid: taxa_cache.get(tid, '') for tid in taxa_ids}


async def main(entity):
    async with AsyncClient() as client:
        return await get_entity_ids(
            entity=entity,
            sapbert_url="https://sap-qdrant.apps.renci.org/annotate/",
            name_res_url="https://name-resolution-sri.renci.org/lookup?autocomplete=false&offset=0&string=",
            node_norm_url="https://nodenormalization-sri.renci.org/get_normalized_nodes",
            count=10,
            session=client
        )


if __name__ == "__main__":
    result = asyncio.run(main("asthma"))
    import json

    print(json.dumps(result, indent=2))
    print(len(result))