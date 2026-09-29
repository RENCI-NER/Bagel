import pytest

from bagel.abbreviations import clean_abbreviation_map, expand

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
        {"abbreviation": "PC", "definition": "pancreatic cancer"},
        {"abbreviation": "PC", "definition": "phosphatidylcholine"},
    ])
    assert amap == {"PC": "pancreatic cancer"}


def test_mutually_recursive_definitions_fail_loudly():
    amap = {"AB": "alpha CD", "CD": "gamma AB"}
    with pytest.raises(ValueError):
        expand("AB", amap)
