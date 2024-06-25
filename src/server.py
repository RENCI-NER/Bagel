from langserve import CustomUserType, add_routes
from config import settings, OpenAIConfig, OLLAMAConfig
from chain import ChainFactory, LLMHelper, get_ollama_llm, get_openai_llm
import prompt
from models import SynonymListContext, BaseModel, Field
from typing import List
import fastapi

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


@app.post('/group_synonyms_ollama')
def group_synonyms_ollama(query: OllamaQuery):
    llm = get_ollama_llm(query.config)
    _prompt = prompt.load_prompt_from_hub(query.prompt_name)
    return LLMHelper.ask(prompt=_prompt, llm=llm, synonym_context=query.context)


@app.post('/batch_group_synonyms_ollama')
def batch_group_synonyms_ollama(query: BatchOllamaQuery):
    llm = get_ollama_llm(query.config)
    _prompt = prompt.load_prompt_from_hub(query.prompt_name)
    return LLMHelper.ask_batch(prompt=_prompt, llm=llm, synonym_contexts=query.context)


@app.post('/group_synonyms_openai')
def group_synonyms_openai(query: OpenAIQuery):
    llm = get_openai_llm(query.config)
    _prompt = prompt.load_prompt_from_hub(query.prompt_name)
    return LLMHelper.ask(prompt=_prompt, llm=llm, synonym_context=query.context)


@app.post('/batch_group_synonyms_openai', description="Batch call multiple synonyms. "
                                                      "Note this is different from openai batch api.")
def batch_group_synonyms_openai(query: BatchOpenAIQuery):
    llm = get_openai_llm(query.config)
    _prompt = prompt.load_prompt_from_hub(query.prompt_name)
    return LLMHelper.ask_batch(prompt=_prompt, llm=llm, synonym_contexts=query.context)

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
