import json

import pytest

from bagel.abbreviations import (build_abbreviation_request, clean_abbreviation_map, expand, extract_documents, find_parentheticals,
                                 load_abbreviation_maps,
                                 load_abbreviation_task, parse_extraction)
from bagel.backends import parse_backend_config
from bagel.types import Document

# abbreviation maps as the LLM extractor returned them for real MedMentions abstracts
CF_ABSTRACT = [{"abbreviation": "Pa", "definition": "Pseudomonas aeruginosa"},
               {"abbreviation": "CF", "definition": "cystic fibrosis"},
               {"abbreviation": "CPA", "definition": "chronic Pa infection"},
               {"abbreviation": "DCTN4", "definition": "dynactin 4"}]


def test_whole_words_only():
    amap = clean_abbreviation_map(CF_ABSTRACT)
    assert expand("CF patients", amap) == "cystic fibrosis patients"
    assert expand("CFTR", amap) == "CFTR"  # not "cystic fibrosisTR"
    assert expand("Pain", amap) == "Pain"  # not "Pseudomonas aeruginosain"


def test_nested_abbreviations_are_expanded():
    amap = clean_abbreviation_map(CF_ABSTRACT)
    assert expand("CPA", amap) == "chronic Pseudomonas aeruginosa infection"


def test_definition_followed_by_its_abbreviation_is_not_doubled():
    amap = clean_abbreviation_map([{"abbreviation": "OGTT", "definition": "oral glucose tolerance test"}])
    assert expand("oral glucose tolerance test (OGTT) 2-hour insulin", amap) == \
        "oral glucose tolerance test 2-hour insulin"


def test_clean_drops_unexpandable_entries():
    amap = clean_abbreviation_map([
        {"abbreviation": "3", "definition": "12-chloro-1-(4-(7-chloroquinolin-4-yl)piperazin-1-yl)ethanone"},
        {"abbreviation": "E", "definition": "phenyl glycidyl ether"},
        {"abbreviation": "HD", "definition": "HD-tDCS"},
        {"abbreviation": "WAT", "definition": "WAT"},
        {"abbreviation": "Duchenne muscular dystrophy", "definition": "DMD"},
        {"abbreviation": "PC", "definition": "pancreatic cancer"},
        {"abbreviation": "PC", "definition": "phosphatidylcholine"},
    ])
    assert amap == {"PC": "pancreatic cancer"}


def test_mutually_recursive_definitions_fail_loudly():
    amap = {"AB": "alpha CD", "CD": "gamma AB"}
    with pytest.raises(ValueError):
        expand("AB", amap)


TEXT = ("Pseudomonas aeruginosa (Pa) infection in cystic fibrosis (CF) patients is associated with worse disease, "
        "and chronic Pa infection (CPA) with reduced lung function. NGF levels were unchanged.")


@pytest.fixture
def task():
    return load_abbreviation_task("abbreviations_v1")


def test_prompt_renders_the_document(task):
    rendered = [m.template.format(text=TEXT) for m in task.messages]
    assert TEXT in rendered[-1]


def test_pairs_not_in_the_text_are_ungrounded(task):
    answer = json.dumps({"abbreviations": [
        {"abbreviation": "CF", "definition": "Cystic  fibrosis"},  # case and spacing don't matter
        {"abbreviation": "CPA", "definition": "chronic Pa infection"},  # nested, as written
        {"abbreviation": "NGF", "definition": "nerve growth factor"},  # used, not defined in the text
        {"abbreviation": "XYZ", "definition": "cystic fibrosis"},  # abbreviation not in the text
    ]})
    extraction = parse_extraction(task, Document(id="d", text=TEXT), answer)
    assert extraction.status == "ok"
    assert [p["abbreviation"] for p in extraction.abbreviation_map] == ["CF", "CPA"]
    assert [p["abbreviation"] for p in extraction.ungrounded] == ["NGF", "XYZ"]


def test_answers_that_break_the_schema_are_parse_errors(task):
    document = Document(id="d", text=TEXT)
    assert parse_extraction(task, document, "not json").status == "parse_error"
    assert parse_extraction(task, document, json.dumps({"abbreviations": [{"abbreviation": "CF"}]})).status == \
        "parse_error"


def test_maps_load_from_bagel_and_bagelwithcheese_files(tmp_path):
    path = tmp_path / "abbreviations.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        {"document": "craft-1", "abbreviation_map": [{"abbreviation": "ER", "definition": "endoplasmic reticulum"}]},
        {"pmid": 25763772, "abbreviation_map": CF_ABSTRACT}]) + "\n")
    maps = load_abbreviation_maps(path)
    assert maps["craft-1"] == {"ER": "endoplasmic reticulum"}
    assert maps["25763772"]["CF"] == "cystic fibrosis"


def test_resuming_with_a_different_extractor_is_refused(task, tmp_path):
    documents = tmp_path / "documents.jsonl"
    documents.write_text(json.dumps({"id": "d", "text": TEXT}) + "\n")
    out = tmp_path / "abbreviations.jsonl"
    out.write_text(json.dumps({"document": "d", "abbreviation_map": [], "fingerprint": "another"}) + "\n")
    backend = parse_backend_config({"kind": "openai", "base_url": "http://localhost:11434/v1", "model": "m"})
    with pytest.raises(ValueError, match="different task or backend"):
        extract_documents(documents, task, backend, out)


def test_parentheticals_in_both_shapes_and_not_inside_names():
    text = ("Hydrogen peroxide (H2O2) and Fe(III)PPIX were given to patients with CF (cystic fibrosis). "
            "Recordings at upper-thoracic (T4) levels (n = 12; P < 0.05) in 2016 (2016) differed (Smith et al.). "
            "Later, H2O2 (H2O2) was repeated (see (Table 1)).")
    found = find_parentheticals(text)
    assert [f["inner"] for f in found] == ["H2O2", "cystic fibrosis", "T4", "see (Table 1)"]
    assert all(text[f["start"]:f["end"]] == f["inner"] for f in found)
    assert found[2]["before"].endswith("Recordings at upper-thoracic")


CLASSIFY_TEXT = "Hydrogen peroxide (H2O2) opened a potassium channel (KCNA1) in patients with cystic fibrosis (CF)."


def test_classify_request_lists_each_parenthetical_with_an_id():
    task = load_abbreviation_task("abbreviations_classify_v1")
    messages, schema, found = build_abbreviation_request(task, Document(id="d", text=CLASSIFY_TEXT))
    assert [p["inner"] for p in found.values()] == ["H2O2", "KCNA1", "CF"]
    assert schema["properties"]["parentheticals"]["items"]["properties"]["id"]["enum"] == ["p1", "p2", "p3"]
    assert '"id": "p2"' in messages[-1]["content"] and "(KCNA1)" in messages[-1]["content"]


def test_only_same_parentheticals_become_pairs():
    task = load_abbreviation_task("abbreviations_classify_v1")
    document = Document(id="d", text=CLASSIFY_TEXT)
    _, schema, found = build_abbreviation_request(task, document)
    answer = {"parentheticals": [
        {"id": "p1", "label": "same", "short_form": "H2O2", "long_form": "Hydrogen peroxide"},
        {"id": "p2", "label": "instance", "short_form": "", "long_form": ""},
        {"id": "p3", "label": "same", "short_form": "CF", "long_form": "cystic fibrosis"}]}
    extraction = parse_extraction(task, document, json.dumps(answer), schema, found)
    assert extraction.status == "ok"
    assert [p["abbreviation"] for p in extraction.abbreviation_map] == ["H2O2", "CF"]
    assert [c["label"] for c in extraction.classified] == ["same", "instance", "same"]


def test_a_parenthetical_left_out_is_a_parse_error():
    task = load_abbreviation_task("abbreviations_classify_v1")
    document = Document(id="d", text=CLASSIFY_TEXT)
    _, schema, found = build_abbreviation_request(task, document)
    answer = {"parentheticals": [{"id": "p1", "label": "same", "short_form": "H2O2", "long_form": "Hydrogen peroxide"}]}
    assert parse_extraction(task, document, json.dumps(answer), schema, found).status == "parse_error"
