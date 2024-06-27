from langchain.chains.base import Chain
import prompt
from prompt import load_prompts
from config import settings, Settings, OLLAMAConfig, OpenAIConfig
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.llms import BaseLLM
from langchain.prompts import Prompt
from models import SynonymListContext, SynonymClassesResponse
from langchain_core.output_parsers import JsonOutputParser
from typing import List


def get_ollama_llm(ollama_config: OLLAMAConfig):
    """
    Get an instance of ollama class
    :param ollama_config: configuration for ollama class
    :return:
    """
    return Ollama(
        base_url=ollama_config.ollama_base_url,
        model=ollama_config.llm_model_name
    )


def get_openai_llm(openai_config: OpenAIConfig):
    """
    Get an instance of ChatOpenAI class
    :param openai_config: Configuration for openai class
    :return:
    """
    return ChatOpenAI(
        api_key=openai_config.access_key,
        organization=openai_config.organization,
        model=openai_config.llm_model_name
    )


class LLMHelper:
    @classmethod
    async def ask(cls, llm: BaseLLM, prompt: Prompt, synonym_context: SynonymListContext):
        chain = cls.get_chain(prompt=prompt, llm=llm)
        return await chain.ainvoke({
            'text': synonym_context.text,
            'term': synonym_context.entity,
            'synonym_list': synonym_context.pretty_print_synonyms()
        }, verbose=True)

    @classmethod
    async def ask_batch(cls, llm: BaseLLM, prompt: Prompt, synonym_contexts: List[SynonymListContext]):
        chain = cls.get_chain(prompt=prompt, llm=llm)
        return await chain.abatch([{
            'text': synonym_context.text,
            'term': synonym_context.entity,
            'synonym_list': synonym_context.pretty_print_synonyms()
        } for synonym_context in synonym_contexts])

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




if __name__ == '__main__':



    # abstract = """
    # The effect of induced hypertension instituted after a 2-h delay following middle cerebral artery occlusion.
    # """
    # target = 'Hypertension'
    # synonym_list = [
    #     ('Hypertension, CTCAE', 'PhenotypicFeature',
    #      ['A disorder characterized by a pathological increase in blood pressure.']),
    #     ('intracranial hypertension', 'Disease',
    #      ['A finding characterized by increased cerebrospinal fluid pressure within the skull.']),
    #     ('Ocular hypertension', 'PhenotypicFeature',
    #      ['Intraocular pressure that is 2 standard deviations above the population mean.']),
    #     ('portal hypertension', 'Disease', [
    #         'Increased blood pressure in the portal venous system. It is most commonly caused by cirrhosis. Other causes include portal vein thrombosis, Budd-Chiari syndrome, and right heart failure. Complications include ascites, esophageal varices, encephalopathy, and splenomegaly.']),
    #     ('secondary hypertension', 'Disease', ['High blood pressure caused by an underlying medical condition.']),
    #     ('essential hypertension', 'Disease', ['Hypertension that presents without an identifiable cause.']), (
    #     'malignant hypertension', 'Disease',
    #     ['Severe hypertension that is characterized by rapid onset of extremely high blood pressure.']), (
    #     'renal hypertension', 'Disease',
    #     ["Hypertension caused by the kidney's hormonal response to narrowing or occlusion of the renal arteries."]),
    #     ('ocular hypertension', 'Disease', ['Abnormally high intraocular pressure.']),
    #     ('renovascular hypertension', 'Disease', ['High blood pressure secondary to renal artery stenosis.']),
    #     ('Hypertension (variable)', 'PhenotypicFeature', ['']), ('Hypertensive (finding)', 'PhenotypicFeature', ['']), (
    #     'hypertensive disorder', 'Disease', [
    #         'Persistently high systemic arterial blood pressure. Based on multiple readings (blood pressure determination), hypertension is currently defined as when systolic pressure is consistently greater than 140 mm Hg or when diastolic pressure is consistently 90 mm Hg or more.']),
    #     ('hypertension (systemic) susceptibility', 'Disease', ['']), ('Hypertelis', 'OrganismTaxon', ['']), (
    #     'Increased blood pressure', 'PhenotypicFeature', [
    #         'Abnormal increase in blood pressure. An individual measurement of increased blood pressure does not necessarily imply hypertension. In practical terms, multiple measurements are recommended to diagnose the presence of hypertension.']),
    #     ('Hypertet', 'NamedThing', ['']),
    #     ('hypertension complicated', 'Disease', [''])
    # ]
    #
    # entities = []
    # for s in synonym_list:
    #     entity = Entity(
    #         label=s[0],
    #         entity_type=s[1],
    #         description=s[2][0]
    #     )
    #     entities.append(entity)
    #
    # import json
    # # print(json.dumps([x.__dict__ for x in entities], indent=2))
    #
    # context = SynonymListContext(
    #     text=abstract,
    #     entity=target,
    #     synonyms=entities
    # )
    # print(context.pretty_print_synonyms())
    prompts = prompt.load_prompts(settings.prompts)

    llm = get_openai_llm(settings.openai_config)
    import json
    with open('/home/kebedey/projects/ner/scratch/terms_syns.json') as stream:
        data = json.load(stream)
    contexts: List[SynonymListContext] = []
    for terms in data['terms']:
        contexts.append(SynonymListContext(**{
            'entity': terms['term'],
            'text': data['text'],
            'synonyms':  terms['synonyms']
        }))





    import asyncio

    response = asyncio.run(LLMHelper.ask_batch(
        prompt=prompts['ask_classes'],
        llm=llm,
        synonym_contexts=contexts
    ))

    print(response)


