import os.path

from langserve import CustomUserType
from config import OpenAIConfig, OLLAMAConfig, settings, logger
from chain import LLMHelper, get_ollama_llm, get_openai_llm
from fastapi.responses import FileResponse, JSONResponse
import prompt
from models import SynonymListContext, BaseModel, Field, Entity
from util.ner_util import get_entity_ids
import fastapi
import httpx
import traceback
from fnmatch import fnmatch
from urllib.parse import urlsplit


app = fastapi.FastAPI(title="Bagel API server", description="Runs bagel re-ranking prompts against pre-configured LLMs"
                                                            "**NOTE**: Ollama endpoints might need custom prompt as "
                                                            "JSON parsing is not fully functional.")
"""doc strings"""
PROMPT_NAME = ("Prompts are bundled with the server (see GET /prompts). prompt_name is a LangChain Hub style name, "
              "e.g. bagel/ask_classes (latest version) or bagel/ask_classes:4da4ad77 (pinned version).\n")
OPENAI_INSTRUCTIONS = ("To use OPENAI endpoints instead of default vllm please set config.url to empty string and pass in"
                       "appropriate openai credentials.")




class Query(CustomUserType):
    prompt_name: str
    text: str
    entity: str
    entity_count: int = 10
    entity_type: str = ""
    name_res_url: str = "https://name-resolution-sri.renci.org/lookup?autocomplete=false&offset=0&limit=10&string="
    sapbert_url: str = "https://sap-qdrant.apps.renci.org/annotate/"
    nodenorm_url: str = "https://nodenormalization-sri.renci.org/get_normalized_nodes"


class OllamaQuery(BaseModel):
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



class DisallowedURLError(ValueError):
    pass


def check_urls(*urls):
    """ Rejects request-supplied service/LLM URLs whose host isn't in settings.allowed_url_hosts, so the server
    can't be used to make requests to arbitrary hosts. """
    for url in urls:
        if not url:
            continue
        host = (urlsplit(url).hostname or "").lower()
        if not any(fnmatch(host, pattern.lower()) for pattern in settings.allowed_url_hosts):
            raise DisallowedURLError(f"Host {host!r} of {url!r} is not in allowed_url_hosts.")


def error_response(e: Exception):
    trb_str = traceback.format_exc()
    logger.error(trb_str)
    status_code = 400 if isinstance(e, (DisallowedURLError, prompt.PromptNotFoundError)) else 500
    body = {'error': str(e)}
    if settings.return_tracebacks:
        body['traceback'] = trb_str
    return JSONResponse(body, status_code=status_code)


async def resolve_entities(query: Query, llm):
    """ Helper function to resolve ids and map back to LLM response."""
    async with httpx.AsyncClient() as client:
        final_results = await get_entity_ids(
            entity=query.entity,
            entity_type=query.entity_type,
            sapbert_url=query.sapbert_url,
            name_res_url=query.name_res_url,
            node_norm_url=query.nodenorm_url,
            session=client,
            count=query.entity_count
        )
        id_list = [
            Entity(**{
                "label": value["name"],
                "identifier": identifier,
                "description": value.get("description", ""),
                "entity_type": value.get("category", ""),
                "taxa": value.get("taxa", "")
            }) for identifier, value in final_results.items()
        ]
        if not len(id_list):
            raise ValueError(f"No entities resolved found for term {query.entity} of type {query.entity_type}.")
        context: SynonymListContext = SynonymListContext(
            text=query.text,
            entity=query.entity,
            synonyms=id_list
        )
        _prompt = prompt.load_prompt(query.prompt_name)
        result = await LLMHelper.ask(prompt=_prompt, llm=llm, synonym_context=context)
        remapped = {}
        for r in result:
            final = final_results[r['identifier']]
            final.update({
                'synonym_type': r['synonym_type']
            })
            remapped[r['identifier']] = final
        return remapped


@app.get("/prompts", description="Lists the bundled prompts as {prompt_name: {latest, commits}}.")
async def list_prompts():
    return prompt._load_index()


@app.get("/", include_in_schema=False)
async def index():
    os.path.dirname(__file__)
    return FileResponse(f"{os.path.dirname(__file__)}/static/index.html")

@app.post('/find_curies_openai', description="This will make requests to Name resolver and Sapbert to find "
                                             "candidate identifiers for an entity in the text, and will perform LLM"
                                             "augmented reranking(classification)" + PROMPT_NAME + OPENAI_INSTRUCTIONS)
async def find_curies(query: OpenAICurieQuery):
    # collect results from name-res , and sap-bert
    try:
        check_urls(query.name_res_url, query.sapbert_url, query.nodenorm_url, query.config.url)
        llm = get_openai_llm(query.config)
        remapped = await resolve_entities(query, llm)
    except Exception as e:
        return error_response(e)
    return remapped


@app.post('/find_curies_ollama', description="This will make requests to Name resolver and Sapbert to find "
                                             "candidate identifiers for an entity in the text, and will perform LLM"
                                             "augmented reranking(classification)" + PROMPT_NAME )
async def find_curies(query: OllamaCurieQuery):
    # collect results from name-res , and sap-bert
    try:
        check_urls(query.name_res_url, query.sapbert_url, query.nodenorm_url, query.config.ollama_base_url)
        llm = get_ollama_llm(query.config)
        remapped = await resolve_entities(query, llm)
    except Exception as e:
        return error_response(e)
    return remapped

@app.post('/group_synonyms_openai', description="Expects a list of synonyms and will perform LLM augmented "
                                                "reranking(classification) for the provided list for the given entity."
                                                + PROMPT_NAME + OPENAI_INSTRUCTIONS)
async def group_synonyms_openai(query: OpenAIQuery):
    try:
        check_urls(query.config.url)
        llm = get_openai_llm(query.config)
        _prompt = prompt.load_prompt(query.prompt_name)
        response = await LLMHelper.ask(prompt=_prompt, llm=llm, synonym_context=query.context)
    except Exception as e:
        return error_response(e)
    return response


@app.post('/group_synonyms_ollama', description="Expects a list of synonyms and will perform LLM augmented "
                                                "reranking(classification) for the provided list for the given entity."
                                                + PROMPT_NAME)
async def group_synonyms_ollama(query: OllamaQuery):
    try:
        check_urls(query.config.ollama_base_url)
        llm = get_ollama_llm(query.config)
        _prompt = prompt.load_prompt(query.prompt_name)
        response = await LLMHelper.ask(prompt=_prompt, llm=llm, synonym_context=query.context)
    except Exception as e:
        return error_response(e)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=9001)
