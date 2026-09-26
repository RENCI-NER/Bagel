import pytest

from llm_output import parse_llm_json, map_color_codes, LLMOutputError
from models import SynonymListContext, Entity


@pytest.fixture
def synonyms():
    return SynonymListContext(text="", entity="", synonyms=[
        Entity(label="aspirin", identifier="CHEBI:15365", entity_type="biolink:SmallMolecule"),
        Entity(label="salicylic acid", identifier="CHEBI:16914", entity_type="biolink:SmallMolecule"),
    ]).synonyms  # color codes: alizarin, amaranth


@pytest.mark.parametrize("item", [
    {"synonym": "aspirin", "synonym_type": "exact", "color_code": "alizarin"},          # prompt spec
    {"synonym": "aspirin", "synonymType": "exact", "color-code": "alizarin"},           # prompt's own example
    {"synonym": "aspirin", "synonymType": "exact", "colorCode": "alizarin"},
    {"candidate": "aspirin", "relation_type": "exact", "color_code": "alizarin"},       # ask_classes_cot
    {"synonym": "aspirin", "synonym_type": "Exact", "color_code": " Alizarin "},
])
def test_key_variants_map(synonyms, item):
    entities, problems = map_color_codes(synonyms, [item])
    assert problems == []
    assert [(e["identifier"], e["synonym_type"]) for e in entities] == [("CHEBI:15365", "exact")]
    assert "color_code" not in entities[0]


def test_unknown_color_is_reported_not_raised(synonyms):
    entities, problems = map_color_codes(synonyms, [
        {"synonym_type": "exact", "color_code": "alizarin"},
        {"synonym_type": "broad", "color_code": "chartreuse"},
        {"synonym_type": "broad"},
    ])
    assert [e["identifier"] for e in entities] == ["CHEBI:15365"]
    assert [p["reason"] for p in problems] == ["unknown_color", "missing_color"]


def test_no_match_types_are_dropped(synonyms):
    entities, problems = map_color_codes(synonyms, [{"relation_type": "None", "color_code": "amaranth"}])
    assert entities == [] and problems == []


def test_parse_fenced_and_reasoning():
    assert parse_llm_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    cot = ('<reasoning>alizarin is [maybe] exact; see {"note": 1}</reasoning>\n'
           '```json\n[{"candidate": "x", "relation_type": "exact", "color_code": "alizarin"}]\n```')
    assert parse_llm_json(cot)[0]["color_code"] == "alizarin"
    assert parse_llm_json('<think>maybe [alizarin] or {"x": 1}</think>\n[{"a": 2}]') == [{"a": 2}]
    assert parse_llm_json('Here you go: [{"a": 1}] thanks') == [{"a": 1}]
    assert parse_llm_json('{"synonyms": [{"a": 1}]}') == [{"a": 1}]
    assert parse_llm_json([{"a": 1}]) == [{"a": 1}]


def test_parse_garbage_raises():
    with pytest.raises(LLMOutputError):
        parse_llm_json("I could not find any synonyms.")


def test_v2_object_output(synonyms):
    response = parse_llm_json('{"interpretation": "the drug", "matches": [{"id": "alizarin", "relation": "exact"}]}')
    entities, problems = map_color_codes(synonyms, response)
    assert [(e["identifier"], e["synonym_type"]) for e in entities] == [("CHEBI:15365", "exact")]
