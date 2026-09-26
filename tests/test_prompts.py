import pytest

import prompt
from models import SynonymListContext, Entity


def all_prompt_versions():
    index = prompt._load_index()
    return [(name, commit) for name, value in index.items() for commit in value["commits"]]


@pytest.fixture(autouse=True)
def no_hub(monkeypatch):
    monkeypatch.delenv("BAGEL_PROMPT_HUB_FALLBACK", raising=False)


@pytest.mark.parametrize("name,commit", all_prompt_versions())
def test_every_vendored_prompt_loads_and_formats(name, commit):
    template = prompt.load_prompt(f"bagel/{name}:{commit}")
    assert set(template.input_variables) == {"text", "term", "synonym_list"}
    assert template.metadata["lc_hub_repo"] == name
    assert template.metadata["lc_hub_commit_hash"] == commit
    context = SynonymListContext(text="Aspirin reduces heart disease risk.", entity="Aspirin",
                                 synonyms=[Entity(label="aspirin", identifier="CHEBI:15365",
                                                  entity_type="biolink:SmallMolecule")])
    rendered = template.format_prompt(text=context.text, term=context.entity,
                                      synonym_list=context.pretty_print_synonyms()).to_string()
    assert "Aspirin reduces heart disease risk." in rendered


def test_name_forms_resolve():
    latest = prompt.list_prompts()["ask_classes"]
    assert prompt.load_prompt("bagel/ask_classes").metadata["lc_hub_commit_hash"] == latest
    assert prompt.load_prompt("ask_classes").metadata["lc_hub_commit_hash"] == latest
    assert prompt.load_prompt("bagel/ask_classes:4da4ad77").metadata["lc_hub_commit_hash"].startswith("4da4ad77")


def test_unknown_prompt_or_version_fails_without_network():
    with pytest.raises(KeyError):
        prompt.load_prompt("bagel/does_not_exist")
    with pytest.raises(KeyError):
        prompt.load_prompt("bagel/ask_classes:deadbeef")
    with pytest.raises(ValueError):
        prompt.load_prompt("someone_else/ask_classes")
