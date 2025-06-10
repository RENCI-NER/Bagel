from langserve import CustomUserType
from config import OpenAIConfig, OLLAMAConfig
from chain import LLMHelper, get_ollama_llm, get_openai_llm
import prompt
from models import SynonymListContext, BaseModel, Field, Entity
from util.ner_util import get_entity_ids
import fastapi
import httpx


app = fastapi.FastAPI(title="Bagel API server", description="Runs bagel re-ranking prompts against pre-configured LLMs"
                                                            "**NOTE**: Ollama endpoints might need custom prompt as "
                                                            "JSON parsing is not fully functional.")
"""doc strings"""
PROMPT_NAME = ("Prompts are currently hosted on smith.langchain.com (https://smith.langchain.com/hub/bagel/)."
              "Prompt name is name for a public prompt hosted in smith.langchain.com.\n")
OPENAI_INSTRUCTIONS = ("To use OPENAI endpoints instead of default vllm please set config.url to empty string and pass in"
                       "appropriate openai credentials.")




class Query(CustomUserType):
    prompt_name: str
    text: str
    entity: str
    entity_type: str = ""
    name_res_url: str = "https://name-resolution-sri.renci.org/lookup?autocomplete=false&offset=0&limit=10&string="
    sapbert_url: str = "https://sap-qdrant.apps.renci.org/annotate/"
    nodenorm_url: str = "https://nodenormalization-sri.renci.org/get_normalized_nodes"


class OllamaQuery(Query):
    prompt_name: str
    context: SynonymListContext
    config: OLLAMAConfig = Field(default=OLLAMAConfig())


class OpenAIQuery(BaseModel):
    prompt_name: str
    context: SynonymListContext
    config: OpenAIConfig = Field(default=OpenAIConfig())


class OpenAICurieQuery(Query):
    config: OpenAIConfig = Field(default=OpenAIConfig())

class OllamaCurieQuery(Query):
    config: OLLAMAConfig = Field(default=OLLAMAConfig())



async def resolve_entities(query: Query, llm, count=20):
    """ Helper function to resolve ids and map back to LLM response."""
    async with httpx.AsyncClient() as client:
        final_results = await get_entity_ids(
            entity=query.entity,
            entity_type=query.entity_type,
            sapbert_url=query.sapbert_url,
            name_res_url=query.name_res_url,
            node_norm_url=query.nodenorm_url,
            session=client,
            count=count
        )
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
        _prompt = prompt.load_prompt_from_hub(query.prompt_name)
        result = await LLMHelper.ask(prompt=_prompt, llm=llm, synonym_context=context)
        remapped = {}
        for r in result:
            final = final_results[r['identifier']]
            final.update({
                'synonym_type': r['synonym_type']
            })
            remapped[r['identifier']] = final
        return remapped


@app.post('/find_curies_openai', description="This will make requests to Name resolver and Sapbert to find "
                                             "candidate identifiers for an entity in the text, and will perform LLM"
                                             "augmented reranking(classification)" + PROMPT_NAME + OPENAI_INSTRUCTIONS)
async def find_curies(query: OpenAICurieQuery):
    # collect results from name-res , and sap-bert
    llm = get_openai_llm(query.config)
    remapped = await resolve_entities(query, llm, count=20)
    return remapped


@app.post('/find_curies_ollama', description="This will make requests to Name resolver and Sapbert to find "
                                             "candidate identifiers for an entity in the text, and will perform LLM"
                                             "augmented reranking(classification)" + PROMPT_NAME )
async def find_curies(query: OllamaCurieQuery):
    # collect results from name-res , and sap-bert
    llm = get_ollama_llm(query.config)
    remapped = await resolve_entities(query, llm, count=20)
    return remapped

@app.post('/group_synonyms_openai', description="Expects a list of synonyms and will perform LLM augmented "
                                                "reranking(classification) for the provided list for the given entity."
                                                + PROMPT_NAME + OPENAI_INSTRUCTIONS)
async def group_synonyms_openai(query: OpenAIQuery):
    llm = get_openai_llm(query.config)
    _prompt = prompt.load_prompt_from_hub(query.prompt_name)
    return await LLMHelper.ask(prompt=_prompt, llm=llm, synonym_context=query.context)


@app.post('/group_synonyms_ollama', description="Expects a list of synonyms and will perform LLM augmented "
                                                "reranking(classification) for the provided list for the given entity."
                                                + PROMPT_NAME)
async def group_synonyms_ollama(query: OllamaQuery):
    llm = get_ollama_llm(query.config)
    _prompt = prompt.load_prompt_from_hub(query.prompt_name)
    return await LLMHelper.ask(prompt=_prompt, llm=llm, synonym_context=query.context)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=9001)
