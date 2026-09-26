import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import FakeListChatModel, FakeListLLM

import server

ANSWER = '[{"synonym": "Aspirin", "synonym_type": "exact", "color_code": "alizarin"}]'
CONTEXT = {
    "text": "The study focused on the effects of Aspirin on heart disease.",
    "entity": "Aspirin",
    "synonyms": [
        {"label": "Aspirin", "identifier": "CHEBI:15365", "description": "A small molecule drug."},
        {"label": "Acetylsalicylic acid", "identifier": "DRUGBANK:DB00945"},
    ],
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "get_ollama_llm", lambda config: FakeListLLM(responses=[ANSWER]))
    monkeypatch.setattr(server, "get_openai_llm", lambda config: FakeListChatModel(responses=[ANSWER]))

    async def fake_get_entity_ids(**kwargs):
        return {"CHEBI:15365": {"name": "aspirin", "category": "biolink:SmallMolecule", "name_res_rank": 1}}
    monkeypatch.setattr(server, "get_entity_ids", fake_get_entity_ids)
    return TestClient(server.app)


def test_readme_group_synonyms_ollama_example(client):
    # README example: no entity_type on synonyms, no top-level text/entity
    response = client.post("/group_synonyms_ollama", json={
        "prompt_name": "bagel/ask_classes_no_system", "context": CONTEXT,
        "config": {"llm_model_name": "llama3.1", "ollama_base_url": "https://ollama.apps.renci.org"}})
    assert response.status_code == 200, response.text
    assert [(r["identifier"], r["synonym_type"]) for r in response.json()] == [("CHEBI:15365", "exact")]


def test_find_curies_openai(client):
    response = client.post("/find_curies_openai", json={
        "prompt_name": "bagel/ask_classes", "text": CONTEXT["text"], "entity": "Aspirin",
        "config": {"url": ""}})
    assert response.status_code == 200, response.text
    assert response.json()["CHEBI:15365"]["synonym_type"] == "exact"


@pytest.mark.parametrize("path,body", [
    ("/find_curies_openai", {"prompt_name": "bagel/ask_classes", "text": "t", "entity": "e",
                             "nodenorm_url": "http://169.254.169.254/latest/meta-data"}),
    ("/group_synonyms_ollama", {"prompt_name": "bagel/ask_classes", "context": CONTEXT,
                                "config": {"ollama_base_url": "http://internal-db:5432"}}),
])
def test_disallowed_hosts_rejected(client, path, body):
    response = client.post(path, json=body)
    assert response.status_code == 400
    assert "not in allowed_url_hosts" in response.json()["error"]
    assert "traceback" not in response.json()


def test_unknown_prompt_is_400(client):
    response = client.post("/group_synonyms_ollama", json={"prompt_name": "bagel/nope", "context": CONTEXT})
    assert response.status_code == 400 and "traceback" not in response.json()
