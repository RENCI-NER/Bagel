from langserve import CustomUserType, add_routes
from config import settings, OpenAIConfig, OLLAMAConfig, logger
from chain import ChainFactory, LLMHelper, get_ollama_llm, get_openai_llm
import prompt
from models import SynonymListContext, BaseModel, Field, Entity
from typing import List
import fastapi
import httpx


app = fastapi.FastAPI(title="Bagel API server", description="Runs bagel re-ranking prompts against pre-configured LLMs")


class Query(CustomUserType):
    abstract: str
    term: str
    synonym_list: str


class OllamaQuery(BaseModel):
    prompt_name: str
    context: SynonymListContext
    config: OLLAMAConfig = Field(default=OLLAMAConfig())


class BatchOllamaQuery(OllamaQuery):
    context: List[SynonymListContext]


class OpenAIQuery(BaseModel):
    prompt_name: str
    context: SynonymListContext
    config: OpenAIConfig = Field(default=OpenAIConfig())


class BatchOpenAIQuery(OpenAIQuery):
    context: List[SynonymListContext]


class OpenAICurieQuery(BaseModel):
    prompt_name: str
    text: str
    entity: str
    entity_type: str = ""
    config: OpenAIConfig = Field(default=OpenAIConfig())
    name_res_url: str = "https://name-resolution-sri.renci.org/lookup?autocomplete=false&offset=0&limit=10&string="
    sapbert_url: str = "https://sap-qdrant.apps.renci.org/annotate/"
    nodenorm_url: str = "https://nodenormalization-sri.renci.org/get_normalized_nodes"


@app.post('/find_curies_openai')
async def find_curies(query: OpenAICurieQuery):
    # find nameres results
    async with httpx.AsyncClient() as client:
        # create nameres url using entity
        name_res_url = query.name_res_url + query.entity
        # create sapbert request payload for entity
        sapbert_payload = {
            "text": f"{query.entity}",
            "model_name": "sapbert",
            "count": 10
        }
        # if entity type is sent as part of the query add it to the sapbert and nameres requests
        if query.entity_type:
            name_res_url += "&biolink_type=" + query.entity_type
            sapbert_payload["args"] = {
                "bl_type": query.entity_type
            }
        # Get response from sapbert and nameres
        name_res_response = await client.get(name_res_url)
        sapbert_response = await client.post(query.sapbert_url, json=sapbert_payload)

        # reformatting bucket to collect score , rank , and labels from nameres and sapbert
        reformatted = {}

        if name_res_response.status_code == 200:
            name_res_json = name_res_response.json()
            reformatted = {
                f"{x['curie']}": {
                    "name": x["label"],
                    "name_res_rank": index + 1,
                    "nameres_score": x["score"],
                    "taxa": x["taxa"],
                    "category": x["types"][0]
                } for index, x in enumerate(name_res_json)
            }

        if sapbert_response.status_code == 200:
            sapbert_json = sapbert_response.json()
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
        # Node norm payload to normalize and get descriptions of all curies gathered above,
        # all conflations are true
        nodenorm_payload = {
            "curies": list(reformatted.keys()),
            "conflate": True,
            "description": True,
            "drug_chemical_conflate": True
        }
        normalized_response = await client.post(query.nodenorm_url, json=nodenorm_payload)
        final_results = {}
        added = []
        if normalized_response.status_code == 200:
            normalized_response_json = normalized_response.json()
            for curie in normalized_response_json:
                normalized_identifier = normalized_response_json[curie]
                # make sure that we actually have normalized it
                if normalized_identifier:
                    norm_id = normalized_identifier["id"]["identifier"]
                    norm_label = normalized_identifier["id"].get("label", "")
                    norm_desc = normalized_identifier["id"].get("description", "")
                    norm_category = normalized_identifier["type"][0]
                    # go and get it from the above reformatted dict,
                    reformatted_entry = reformatted[curie]
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
            for x in reformatted:
                if x not in added:
                    final_results[x] = reformatted[x]
        llm = get_openai_llm(query.config)
        _prompt = prompt.load_prompt_from_hub(query.prompt_name)
        id_list = [
            Entity(**{
                "label": value["name"],
                "identifier": identifier,
                "description": value.get("description", ""),
                "entity_type": value.get("category", ""),
            }) for identifier, value in final_results.items()
        ]
        context: SynonymListContext = SynonymListContext(
            text=query.text,
            entity=query.entity,
            synonyms=id_list
        )

        result = await LLMHelper.ask(prompt=_prompt, llm=llm, synonym_context=context)
        remapped = {}
        for r in result:
            final = final_results[r['identifier']]
            final.update({
                'synonym_type': r['synonym_type']
            })
            remapped[r['identifier']] = final

        return remapped

@app.post('/group_synonyms_ollama')
async def group_synonyms_ollama(query: OllamaQuery):
    llm = get_ollama_llm(query.config)
    _prompt = prompt.load_prompt_from_hub(query.prompt_name)
    return await LLMHelper.ask(prompt=_prompt, llm=llm, synonym_context=query.context)


@app.post('/batch_group_synonyms_ollama')
async def batch_group_synonyms_ollama(query: BatchOllamaQuery):
    llm = get_ollama_llm(query.config)
    _prompt = prompt.load_prompt_from_hub(query.prompt_name)
    return await LLMHelper.ask_batch(prompt=_prompt, llm=llm, synonym_contexts=query.context)


@app.post('/group_synonyms_openai')
async def group_synonyms_openai(query: OpenAIQuery):
    llm = get_openai_llm(query.config)
    _prompt = prompt.load_prompt_from_hub(query.prompt_name)
    return await LLMHelper.ask(prompt=_prompt, llm=llm, synonym_context=query.context)


@app.post('/batch_group_synonyms_openai', description="Batch call multiple synonyms. "
                                                      "Note this is different from openai batch api.")
async def batch_group_synonyms_openai(query: BatchOpenAIQuery):
    llm = get_openai_llm(query.config)
    _prompt = prompt.load_prompt_from_hub(query.prompt_name)
    return await LLMHelper.ask_batch(prompt=_prompt, llm=llm, synonym_contexts=query.context)

# add langserve endpoints
if settings.langServe:
    for chain in ChainFactory.get_all_chains():
        add_routes(
            app,
            chain,
            path=f"/{chain.name}",
            input_type=Query
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=9001)
