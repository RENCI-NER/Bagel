import asyncio

import pytest
from langchain_core.language_models import FakeListChatModel, FakeListLLM

import prompt
from chain import LLMHelper
from models import SynonymListContext, Entity

EXAMPLE_STYLE = '```json\n[{"synonym": "aspirin", "vocabulary class": "SmallMolecule", "synonymType": "exact", "color-code": "alizarin"}]\n```'
COT_STYLE = ('<reasoning>alizarin: aspirin, exact. amaranth: salicylic acid, related.</reasoning>\n'
             '[{"candidate": "aspirin", "relation_type": "exact", "color_code": "alizarin"},'
             ' {"candidate": "salicylic acid", "relation_type": "None", "color_code": "amaranth"}]')


@pytest.mark.parametrize("llm_cls", [FakeListChatModel, FakeListLLM])  # ChatOpenAI-like and Ollama-like
@pytest.mark.parametrize("prompt_name,answer", [("bagel/ask_classes_no_system", EXAMPLE_STYLE),
                                                ("bagel/ask_classes_cot", COT_STYLE)])
def test_ask_end_to_end(llm_cls, prompt_name, answer):
    context = SynonymListContext(text="Aspirin reduced events.", entity="Aspirin", synonyms=[
        Entity(label="aspirin", identifier="CHEBI:15365", entity_type="biolink:SmallMolecule"),
        Entity(label="salicylic acid", identifier="CHEBI:16914", entity_type="biolink:SmallMolecule"),
    ])
    result = asyncio.run(LLMHelper.ask(llm=llm_cls(responses=[answer]), prompt=prompt.load_prompt(prompt_name),
                                       synonym_context=context))
    assert [(r["identifier"], r["synonym_type"]) for r in result] == [("CHEBI:15365", "exact")]
