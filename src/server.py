from langserve import CustomUserType, add_routes
from config import settings, OpenAIConfig, OLLAMAConfig, logger
from chain import ChainFactory, LLMHelper, get_ollama_llm, get_openai_llm
import prompt
from models import SynonymListContext, BaseModel, Field, Entity
from util.ner_util import get_entity_ids
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
    async with httpx.AsyncClient() as client:
        # collect results from name-res , and sap-bert
        final_results = await get_entity_ids(
            entity=query.entity,
            entity_type=query.entity_type,
            sapbert_url=query.sapbert_url,
            name_res_url=query.name_res_url,
            node_norm_url=query.nodenorm_url,
            session=client,
            count=20
        )
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
