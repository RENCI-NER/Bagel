import prompt
from prompt import load_prompts
from config import settings, Settings, OLLAMAConfig, OpenAIConfig, logger
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.llms import BaseLLM
from langchain.prompts import Prompt
from models import SynonymListContext, SynonymClassesResponse, Entity
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Dict



def get_ollama_llm(ollama_config: OLLAMAConfig):
    """
    Get an instance of ollama class
    :param ollama_config: configuration for ollama class
    :return:
    """
    return Ollama(
        base_url=ollama_config.ollama_base_url,
        model=ollama_config.llm_model_name,
        **ollama_config.llm_model_args
    )


def get_openai_llm(openai_config: OpenAIConfig):
    """
    Get an instance of ChatOpenAI class
    :param openai_config: Configuration for openai class
    :return:
    """
    return ChatOpenAI(
        api_key=openai_config.access_key or "EMPTY",
        organization=openai_config.organization,
        model=openai_config.llm_model_name,
        base_url=openai_config.url,
        **openai_config.llm_model_args,
    )


class LLMHelper:
    @classmethod
    async def ask(cls, llm: BaseLLM, prompt: Prompt, synonym_context: SynonymListContext):
        chain = cls.get_chain(prompt=prompt, llm=llm)
        logger.info("SENDING TO LLM ........")
        logger.info(prompt.format_prompt(**{
            'text': synonym_context.text,
            'term': synonym_context.entity,
            'synonym_list': synonym_context.pretty_print_synonyms()
        }).to_string())
        logger.info("END LLM MESSAGE. ")
        response = await chain.ainvoke({
            'text': synonym_context.text,
            'term': synonym_context.entity,
            'synonym_list': synonym_context.pretty_print_synonyms()
        }, verbose=True)

        return LLMHelper.re_map_responses(synonym_context.synonyms, response)


    @classmethod
    def re_map_responses(cls, synonym_list: List[Entity], llm_response: Dict[str, any]) -> any:
        # Map back ids to llm responses
        by_color = {
            entity.color_code: entity for entity in synonym_list
        }
        entities = []
        for item in llm_response:
            synonym_type = item['synonym_type']
            entity = by_color[item['color_code']]
            final = entity.dict()
            final.update({
                "synonym_type": synonym_type
            })
            del final['color_code']
            entities.append(final)
        return entities

    @classmethod
    async def ask_batch(cls, llm: BaseLLM, prompt: Prompt, synonym_contexts: List[SynonymListContext]):
        chain = cls.get_chain(prompt=prompt, llm=llm)
        responses = await chain.abatch([{
            'text': synonym_context.text,
            'term': synonym_context.entity,
            'synonym_list': synonym_context.pretty_print_synonyms()
        } for synonym_context in synonym_contexts])
        results = []
        for synonym_context, response in zip(synonym_contexts,responses):
            results.append(LLMHelper.re_map_responses(synonym_context.synonyms, response))
        return results


    @classmethod
    def get_chain(cls, prompt: Prompt, llm: BaseLLM, model_name: str = ""):
        chain = (prompt | llm | JsonOutputParser(pydantic_object=SynonymClassesResponse))
        chain.name = prompt.metadata['lc_hub_repo'] + '_' + model_name
        return chain


class ChainFactory:
    chains = {}

    @classmethod
    def get_llms(cls):
        # Add additional LLMS here
        return [
            (settings.openai_config.llm_model_name, get_openai_llm(settings.openai_config)),
            (settings.ollama_config.llm_model_name, get_ollama_llm(settings.ollama_config))
        ]

    @classmethod
    def init_chains(cls):
        prompts = load_prompts(settings.prompts)
        ollama = get_ollama_llm(settings.ollama_config)
        for key, value in prompts.items():
            ChainFactory.chains[key] = [LLMHelper.get_chain(value, llm=llms[1], model_name=llms[0])
                                        for llms in ChainFactory.get_llms()]

    @classmethod
    def get_chain(cls, prompt_name):
        if not cls.chains:
            cls.init_chains()
        if not prompt_name in ChainFactory.chains:
            raise ValueError(f"Prompt {prompt_name} not found locally or in hub.")
        else:
            return cls.chains[prompt_name]


    @classmethod
    def get_all_chains(cls):
        if not cls.chains:
            cls.init_chains()
        all_chains = []
        for prompt_name, chain in ChainFactory.chains.items():
            all_chains += chain
        return all_chains



