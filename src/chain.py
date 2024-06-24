from langchain.chains.base import Chain

import prompt

abstract = """
The effect of induced hypertension instituted after a 2-h delay following middle cerebral artery occlusion (MCAO) on brain edema formation and histochemical injury was studied. Under isoflurane anesthesia, the MCA of 14 spontaneously hypertensive rats was occluded. In the control group (n = 7), the mean arterial pressure (MAP) was not manipulated. In the hypertensive group (n = 7), the MAP was elevated by 25-30 mm Hg beginning 2 h after MCAO. Four hours after MCAO, the rats were killed and the brains harvested. The brains were sectioned along coronal planes spanning the distribution of ischemia produced by MCAO. Specific gravity (SG) was determined in the subcortex and in two sites in the cortex (core and periphery of the ischemic territory). The extent of neuronal injury was determined by 2,3,5-triphenyltetrazolium staining. In the ischemic core, there was no difference in SG in the subcortex and cortex in the two groups. In the periphery of the ischemic territory, SG in the cortex was greater (less edema accumulation) in the hypertensive group (1.041 +/- 0.001 vs 1.039 +/- 0.001, P less than 0.05). The area of histochemical injury (as a percent of the cross-sectional area of the hemisphere) was less in the hypertensive group (33 +/- 3% vs 21 +/- 2%, P less than 0.05). The data indicate that phenylephrine-induced hypertension instituted 2 h after MCAO does not aggravate edema in the ischemic core, that it improves edema in the periphery of the ischemic territory, and that it reduces the area of histochemical neuronal dysfunction.
"""
target = 'Hypertension'
synonym_list = [
    ('Hypertension, CTCAE', 'PhenotypicFeature', ['A disorder characterized by a pathological increase in blood pressure.']),
    ('intracranial hypertension', 'Disease', ['A finding characterized by increased cerebrospinal fluid pressure within the skull.']),
    ('Ocular hypertension', 'PhenotypicFeature', ['Intraocular pressure that is 2 standard deviations above the population mean.']),
    ('portal hypertension', 'Disease', ['Increased blood pressure in the portal venous system. It is most commonly caused by cirrhosis. Other causes include portal vein thrombosis, Budd-Chiari syndrome, and right heart failure. Complications include ascites, esophageal varices, encephalopathy, and splenomegaly.']), ('secondary hypertension', 'Disease', ['High blood pressure caused by an underlying medical condition.']), ('essential hypertension', 'Disease', ['Hypertension that presents without an identifiable cause.']), ('malignant hypertension', 'Disease', ['Severe hypertension that is characterized by rapid onset of extremely high blood pressure.']), ('renal hypertension', 'Disease', ["Hypertension caused by the kidney's hormonal response to narrowing or occlusion of the renal arteries."]), ('ocular hypertension', 'Disease', ['Abnormally high intraocular pressure.']), ('renovascular hypertension', 'Disease', ['High blood pressure secondary to renal artery stenosis.']), ('Hypertension (variable)', 'PhenotypicFeature', ['']), ('Hypertensive (finding)', 'PhenotypicFeature', ['']), ('hypertensive disorder', 'Disease', ['Persistently high systemic arterial blood pressure. Based on multiple readings (blood pressure determination), hypertension is currently defined as when systolic pressure is consistently greater than 140 mm Hg or when diastolic pressure is consistently 90 mm Hg or more.']), ('hypertension (systemic) susceptibility', 'Disease', ['']), ('Hypertelis', 'OrganismTaxon', ['']), ('Increased blood pressure', 'PhenotypicFeature', ['Abnormal increase in blood pressure. An individual measurement of increased blood pressure does not necessarily imply hypertension. In practical terms, multiple measurements are recommended to diagnose the presence of hypertension.']),
    ('Hypertet', 'NamedThing', ['']),
    ('hypertension complicated', 'Disease', [''])
]

from prompt import load_prompts
from config import settings, Settings, OLLAMAConfig, OpenAIConfig
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.llms import BaseLLM
from langchain.prompts import Prompt
from models import SynonymListContext, Entity


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
    async def ask(cls, llm: BaseLLM, prompt: Prompt, synonym_context: SynonymListContext ):
        chain = cls.get_chain(prompt, llm)
        return await chain.ainvoke({
            'text': synonym_context.text,
            'term': synonym_context.entity,
            'synonym_list': synonym_context.pretty_print_synonyms()
        }, verbose=True)

    @classmethod
    def get_chain(cls, prompt: Prompt, llm: BaseLLM, model_name: str):
        chain =  (prompt | llm)
        chain.name = prompt.metadata['lc_hub_repo'] + '_' + model_name
        return chain


class ChainFactory():
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
    all_chains = ChainFactory.get_all_chains()
    for c in all_chains:
        print(c.name)
    # from typing import Dict
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
    # context = SynonymListContext(
    #     text=abstract,
    #     entity=target,
    #     synonyms=entities
    # )
    #
    # prompts: Dict[str, Prompt] = load_prompts(settings.prompts)
    # import   asyncio
    # llm = get_ollama_llm(settings.ollama_config)
    # x = (prompts['ask_classes'].invoke(
    #     {"text": context.text, "term": context.entity, "synonym_list": context.pretty_print_synonyms()
    #      }))
    #
    # print(asyncio.run(LLMHelper.ask(llm, prompts['ask_classes'], context)))

    # chain = LLMHelper.get_chain(prompts['ask_classes'], llm)
    # res = asyncio.run(chain.ainvoke({
    #     'text': abstract,
    #     'term': entity,
    #     'synonym_list': synonym_list
    # }))

    #
    # response = asyncio.run(LLMHelper.ask(
    #     prompts['ask_classes'],
    #     llm,
    #     abstract,
    #     entity,
    #     synonym_list
    # ))
    # print(response)
