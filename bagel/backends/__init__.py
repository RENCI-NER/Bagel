"""Model backends: rendered messages + output JSON schema in, the model's answer (JSON text) out."""
from typing import Annotated, Protocol

from pydantic import BaseModel, Field

from bagel.backends.claude_cli import ClaudeCLIBackend, ClaudeCLIConfig
from bagel.backends.errors import BackendError, BackendResponse
from bagel.backends.openai_compatible import OpenAICompatibleBackend, OpenAIConfig

BackendConfig = Annotated[OpenAIConfig | ClaudeCLIConfig, Field(discriminator="kind")]


class Backend(Protocol):
    def complete(self, messages: list[dict], schema: dict) -> BackendResponse: ...


class _BackendConfigHolder(BaseModel):
    config: BackendConfig


def make_backend(config: dict | BackendConfig) -> Backend:
    if isinstance(config, dict):
        config = _BackendConfigHolder(config=config).config
    if isinstance(config, OpenAIConfig):
        return OpenAICompatibleBackend(config)
    return ClaudeCLIBackend(config)


__all__ = ["Backend", "BackendConfig", "BackendError", "BackendResponse", "make_backend"]
