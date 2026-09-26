import prompt
from prompt import load_prompts
from config import settings, Settings, OLLAMAConfig, OpenAIConfig, logger
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.llms import BaseLLM
from langchain.prompts import Prompt
from models import SynonymListContext, Entity
from llm_output import parse_llm_json, map_color_codes
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from typing import Any, List, Dict



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
        inputs = synonym_context.prompt_inputs()
        logger.info(prompt.format_prompt(**inputs).to_string())
        logger.info("END LLM MESSAGE. ")
        response = await chain.ainvoke(inputs, verbose=True)

        return LLMHelper.re_map_responses(synonym_context.synonyms, response)


    @classmethod
    def re_map_responses(cls, synonym_list: List[Entity], llm_response: Any) -> List[Dict]:
        """ Maps LLM response items back to entities via color codes. Items that cannot be mapped are logged
        and skipped. """
        logger.info(llm_response)
        entities, problems = map_color_codes(synonym_list, parse_llm_json(llm_response))
        for problem in problems:
            logger.warning(f"Could not map LLM response item ({problem['reason']}): {problem['item']}")
        return entities

    @classmethod
    async def ask_batch(cls, llm: BaseLLM, prompt: Prompt, synonym_contexts: List[SynonymListContext]):
        chain = cls.get_chain(prompt=prompt, llm=llm)
        responses = await chain.abatch([synonym_context.prompt_inputs() for synonym_context in synonym_contexts])
        results = []
        for synonym_context, response in zip(synonym_contexts,responses):
            results.append(LLMHelper.re_map_responses(synonym_context.synonyms, response))
        return results


    @classmethod
    def get_chain(cls, prompt: Prompt, llm: BaseLLM, model_name: str = ""):
        chain = (prompt | llm | StrOutputParser() | RunnableLambda(parse_llm_json))
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



