"""Model backends: rendered messages + output JSON schema in, the model's answer (JSON text) out."""
from typing import Annotated, Protocol

from pydantic import BaseModel, Field

from bagel.backends.claude_cli import ClaudeCLIBackend, ClaudeCLIConfig
from bagel.backends.errors import BackendError, BackendResponse
from bagel.backends.jev import JevBackend, JevConfig
from bagel.backends.openai_compatible import AzureOpenAIConfig, OpenAICompatibleBackend, OpenAIConfig

BackendConfig = Annotated[OpenAIConfig | AzureOpenAIConfig | ClaudeCLIConfig | JevConfig, Field(discriminator="kind")]


class Backend(Protocol):
    def complete(self, messages: list[dict], schema: dict) -> BackendResponse: ...


class _BackendConfigHolder(BaseModel):
    config: BackendConfig


def parse_backend_config(config: dict) -> BackendConfig:
    return _BackendConfigHolder(config=config).config


def make_backend(config: dict | BackendConfig) -> Backend:
    if isinstance(config, dict):
        config = parse_backend_config(config)
    if isinstance(config, OpenAIConfig):
        return OpenAICompatibleBackend(config)
    if isinstance(config, JevConfig):
        return JevBackend(config)  # typed decisions: used with typed task specs, not chat prompts
    return ClaudeCLIBackend(config)


__all__ = ["Backend", "BackendConfig", "BackendError", "BackendResponse", "make_backend", "parse_backend_config"]
