from httpx import AsyncClient
import json
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
    if entity_type is not None:
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


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
async def get_nameres_ids(entity: str, session: AsyncClient, count: int = 10, entity_type: str = None, url: str = None):
    name_res_url = url + entity
    if entity_type is not None:
        name_res_url += f"&biolink_type={entity_type}"
    name_res_url += f"&limit={count}"
    response = await session.get(name_res_url)
    formatted = {}
    if response.status_code == 200:
        json_data = response.json()
        formatted = {
            f"{x['curie']}": {
                "name": x["label"],
                "name_res_rank": index + 1,
                "name_res_score": x["score"],
                "taxa": x["taxa"],
                "category": x["types"][0]
            } for index, x in enumerate(json_data)
        }
    else:
        logger.error(f"Error: nameres call for {entity} , url: {name_res_url} returned code: {response.status_code}")
        raise Exception(f"Error: nameres call for {entity}")
    return formatted


async def get_entity_ids(entity: str, name_res_url: str, sapbert_url: str, node_norm_url: str, session: AsyncClient,
                         entity_type=None, count=20):
    get_entities_tasks = [
        get_sapbert_ids(entity=entity,
                        entity_type=entity_type,
                        url=sapbert_url,
                        count=count,
                        session=session
                        ),
        get_nameres_ids(entity=entity,
                        entity_type=entity_type,
                        url=name_res_url,
                        count=count,
                        session=session
                        )
    ]
    responses = await asyncio.gather(*get_entities_tasks)
    logger.debug(f"Combined responses: {json.dumps(responses, indent=2)}")

    # merge them by identifier
    all_curies = set()
    for response in responses:
        all_curies.update(response.keys())
    logger.debug(f"All curies: {all_curies}")
    merged = {}
    for curie in all_curies:
        merged[curie] = {
            "name_res_rank": -1,
            "sapbert_rank": -1,
            "name_res_score": -1,
            "sapbert_score": -1,
        }
        for response in responses:
            merged[curie].update(response.get(curie, {}))

    logger.debug(f"Merged: {json.dumps(merged, indent=2)}")

    nodenorm_payload = {
        "curies": list(merged.keys()),
        "conflate": True,
        "description": True,
        "drug_chemical_conflate": True
    }
    normalized_curies = await session.post(node_norm_url, json=nodenorm_payload)
    final_results = {}
    added = []
    if normalized_curies.status_code == 200:
        normalized_response_json = normalized_curies.json()
        for curie in normalized_response_json:
            normalized_identifier = normalized_response_json[curie]
            # make sure that we actually have normalized it
            if normalized_identifier:
                norm_id = normalized_identifier["id"]["identifier"]
                norm_label = normalized_identifier["id"].get("label", "")
                norm_desc = normalized_identifier["id"].get("description", "")
                norm_category = normalized_identifier["type"][0]
                # go and get it from the above reformatted dict,
                reformatted_entry = merged[curie]
                final_results[norm_id] = reformatted_entry
                final_results[norm_id].update({
                    "description": norm_desc,
                    "category": norm_category,
                })
                if norm_label:
                    final_results[norm_id].update({
                        "name": norm_label
                    })
                added.append(curie)
        # For some reason if these are not normalizing let's just add them into the final list
        for x in merged:
            if x not in added:
                logger.warning(f"Could not normalize {x}, adding to the final results: {merged[x]}")
                final_results[x] = merged[x]
    return final_results


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
                        logger.warning("Could not get taxa {key} from node norm".format(key=key))
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