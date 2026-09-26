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
    assert {"text", "term"} <= set(template.input_variables) <= {"text", "term", "synonym_list", "candidates_json"}
    assert template.metadata["lc_hub_repo"] == name
    assert template.metadata["lc_hub_commit_hash"] == commit
    context = SynonymListContext(text="Aspirin reduces heart disease risk.", entity="Aspirin",
                                 synonyms=[Entity(label="aspirin", identifier="CHEBI:15365",
                                                  entity_type="biolink:SmallMolecule")])
    rendered = template.format_prompt(**context.prompt_inputs()).to_string()
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


def test_local_prompt_sources_are_built():
    """ prompts/src/*.yaml must be compiled (scripts/build_local_prompts.py) into the latest local version."""
    import subprocess, sys, json
    before = json.loads((prompt.PROMPTS_DIR / "index.json").read_text())
    subprocess.run([sys.executable, str(prompt.PROMPTS_DIR.parent / "scripts" / "build_local_prompts.py")],
                   check=True, capture_output=True)
    after = json.loads((prompt.PROMPTS_DIR / "index.json").read_text())
    assert after == before, "prompts/src changed without rebuilding; run scripts/build_local_prompts.py"
