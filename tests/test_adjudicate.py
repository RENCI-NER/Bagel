import json

import pytest

from bagel.adjudicate import build_request, load_task, parse_decision
from bagel.types import Candidate, Document, Mention, Relation

CANDIDATES = [
    Candidate(curie="NCBIGene:7124", label="TNF", category="biolink:Gene", taxa=["Homo sapiens (human)"]),
    Candidate(curie="NCBIGene:24835", label="Tnf", category="biolink:Gene", taxa=["Rattus norvegicus (Norway rat)"]),
    Candidate(curie="UMLS:C0333516", label="Tumor necrosis", category="biolink:Disease"),
]
DOCUMENT = Document(id="1", text="Hepatic Tnf mRNA rose in Sprague-Dawley rats.")
MENTION = Mention(text="Tnf")


@pytest.fixture(params=["relations_v2", "relations_bwc"])
def task(request):
    return load_task(request.param)


def test_request_renders_every_candidate_with_its_id(task):
    messages, schema, ids = build_request(task, DOCUMENT, MENTION, CANDIDATES)
    rendered = "\n".join(m["content"] for m in messages)
    assert DOCUMENT.text in rendered
    for cid, candidate in zip(ids, CANDIDATES):
        assert cid in rendered and candidate.label in rendered
    assert "Norway rat" in rendered
    item = schema["properties"][task.candidates_key]["items"]
    assert item["properties"][task.id_key]["enum"] == ids


def _answer(task, relations: dict[str, str]):
    """An answer in the task's own format, judging the candidates at the given ids."""
    if task.name == "relations_v2":
        return {"interpretation": "the rat gene", "matches": [{"id": i, "relation": r} for i, r in relations.items()]}
    return {"reasoning": "the rat gene", "candidates": [
        {"candidate": "x", "color_code": i, "vocabulary_class": "Gene", "evaluation": "", "relation_type": r}
        for i, r in relations.items()]}


def test_labels_map_to_skos_and_omitted_candidates_are_nomatch():
    task = load_task("relations_v2")
    _, schema, ids = build_request(task, DOCUMENT, MENTION, CANDIDATES)
    answer = _answer(task, {ids[1]: "exact", ids[0]: "related"})
    decision = parse_decision(task, json.dumps(answer), schema, ids, CANDIDATES)
    assert decision.status == "ok"
    assert [(j.curie, j.relation) for j in decision.judgments] == [
        ("NCBIGene:7124", Relation.RELATED), ("NCBIGene:24835", Relation.EXACT), ("UMLS:C0333516", Relation.NONE)]
    assert decision.rationale == "the rat gene"


def test_direction_of_broad_and_narrow_labels():
    # v2 "narrow" = the mention is narrower = the candidate is broader = skos:broadMatch; bwc "subclass" likewise
    v2, bwc = load_task("relations_v2"), load_task("relations_bwc")
    assert v2.labels["narrow"] == bwc.labels["subclass"] == Relation.BROAD
    assert v2.labels["broad"] == bwc.labels["superclass"] == Relation.NARROW


def test_invalid_json_is_a_parse_error(task):
    _, schema, ids = build_request(task, DOCUMENT, MENTION, CANDIDATES)
    decision = parse_decision(task, "Sure! The rat gene is the match.", schema, ids, CANDIDATES)
    assert decision.status == "parse_error" and decision.raw.startswith("Sure!")


def test_unoffered_id_violates_the_schema(task):
    _, schema, ids = build_request(task, DOCUMENT, MENTION, CANDIDATES)
    decision = parse_decision(task, json.dumps(_answer(task, {"chartreuse": "exact"})), schema, ids, CANDIDATES)
    assert decision.status == "parse_error"


def test_same_candidate_judged_twice_is_an_error():
    task = load_task("relations_v2")
    _, schema, ids = build_request(task, DOCUMENT, MENTION, CANDIDATES)
    answer = {"interpretation": "", "matches": [{"id": ids[0], "relation": "exact"}, {"id": ids[0], "relation": "related"}]}
    assert parse_decision(task, json.dumps(answer), schema, ids, CANDIDATES).status == "parse_error"


def test_claude_cli_strips_only_a_fence_around_the_whole_answer():
    from bagel.backends.claude_cli import strip_fence
    assert strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fence('{"a": 1}') == '{"a": 1}'
    assert strip_fence('Here it is:\n```json\n{"a": 1}\n```') == 'Here it is:\n```json\n{"a": 1}\n```'
