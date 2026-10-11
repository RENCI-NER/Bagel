import subprocess
import uuid

import openai
import pytest

from bagel.backends import make_backend
from bagel.backends.openai_compatible import AzureOpenAIConfig, make_client
from bagel.backends.secrets import keychain_password, resolve_api_key


@pytest.fixture
def keychain_item():
    """A throwaway item in the login Keychain (macOS), removed afterwards."""
    service = f"bagel-test-{uuid.uuid4().hex[:8]}"
    subprocess.run(["security", "add-generic-password", "-a", "bagel-test", "-s", service, "-w", "not-a-real-key"],
                   check=True)
    yield service
    subprocess.run(["security", "delete-generic-password", "-s", service], check=True, capture_output=True)


def test_keychain_key_is_read_and_takes_precedence(keychain_item, monkeypatch):
    monkeypatch.setenv("BAGEL_TEST_KEY", "from-env")
    assert keychain_password(keychain_item) == "not-a-real-key"
    assert resolve_api_key(keychain_item, "BAGEL_TEST_KEY") == "not-a-real-key"
    assert resolve_api_key(None, "BAGEL_TEST_KEY") == "from-env"


def test_missing_keychain_item_says_how_to_add_it():
    with pytest.raises(ValueError, match="security add-generic-password"):
        keychain_password(f"bagel-missing-{uuid.uuid4().hex[:8]}")


def test_azure_backend_builds_an_azure_client(keychain_item):
    config = {"kind": "azure_openai", "endpoint": "https://example-resource.openai.azure.com",
              "model": "gpt-4o-deployment", "api_key_keychain": keychain_item}
    backend = make_backend(config)
    assert isinstance(backend.config, AzureOpenAIConfig)
    assert isinstance(backend.client, openai.AzureOpenAI)
    assert backend.config.max_tokens_parameter == "max_completion_tokens"


def test_azure_without_a_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No Azure OpenAI key"):
        make_client(AzureOpenAIConfig(endpoint="https://example-resource.openai.azure.com", model="d"))


def test_azure_v1_endpoint_uses_the_standard_client(keychain_item):
    backend = make_backend({"kind": "azure_openai", "endpoint": "https://example-resource.openai.azure.com/openai/v1/",
                            "model": "gpt-4.1", "api_key_keychain": keychain_item})
    assert backend.config.is_v1
    assert type(backend.client) is openai.OpenAI
    assert str(backend.client.base_url) == "https://example-resource.openai.azure.com/openai/v1/"
    assert backend.config.max_tokens_parameter == "max_completion_tokens"
