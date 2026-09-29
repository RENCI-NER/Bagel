"""OpenAI chat completions: OpenAI, Azure OpenAI, and compatible servers (vLLM, Ollama's /v1, ...)."""
from typing import Literal

import openai
from pydantic import BaseModel, Field

from bagel.backends.errors import BackendError, BackendResponse
from bagel.backends.secrets import resolve_api_key


class OpenAIConfig(BaseModel):
    kind: Literal["openai"] = "openai"
    model: str
    base_url: str | None = None  # None: api.openai.com
    api_key_keychain: str | None = None  # macOS Keychain service holding the key (takes precedence)
    api_key_env: str | None = "OPENAI_API_KEY"
    temperature: float | None = 0  # set to null for reasoning models that reject it (o-series, gpt-5)
    seed: int | None = 0
    structured_output: bool = True  # send the output schema as response_format (json_schema, strict)
    extra_body: dict = Field(default_factory=dict)  # server-specific options, e.g. {"think": false} for Ollama
    max_output_tokens: int | None = 4096  # runaway generation ends as a visible parse_error, not a timeout
    timeout: float = 300
    # the SDK retries twice by default, silently; a failure should surface as api_error (runs can --retry-failed)
    max_retries: int = 0

    @property
    def max_tokens_parameter(self) -> str:
        # OpenAI and Azure want max_completion_tokens (required by reasoning models); local servers take max_tokens
        return "max_tokens" if self.base_url else "max_completion_tokens"


class AzureOpenAIConfig(OpenAIConfig):
    """
    Azure OpenAI. `model` is the deployment name; the model behind it is recorded from each response.
    endpoint: https://<resource>.openai.azure.com
    """
    kind: Literal["azure_openai"] = "azure_openai"
    endpoint: str
    api_version: str = "2024-10-21"  # GA version with structured outputs
    api_key_env: str | None = "AZURE_OPENAI_API_KEY"


def make_client(config: OpenAIConfig) -> openai.OpenAI:
    api_key = resolve_api_key(config.api_key_keychain, config.api_key_env)
    common = {"timeout": config.timeout, "max_retries": config.max_retries}
    if isinstance(config, AzureOpenAIConfig):
        if api_key is None:
            raise ValueError(f"No Azure OpenAI key: set api_key_keychain, or the {config.api_key_env} variable")
        return openai.AzureOpenAI(api_key=api_key, azure_endpoint=config.endpoint, api_version=config.api_version,
                                  **common)
    if api_key is None:
        if config.base_url is None:
            raise ValueError(f"No OpenAI key: set api_key_keychain, or the {config.api_key_env} variable")
        api_key = "unused"  # local servers (Ollama, vLLM) don't check it
    return openai.OpenAI(api_key=api_key, base_url=config.base_url, **common)


class OpenAICompatibleBackend:
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = make_client(config)

    def complete(self, messages: list[dict], schema: dict) -> BackendResponse:
        c = self.config
        kwargs = {"model": c.model, "messages": messages}
        if c.temperature is not None:
            kwargs["temperature"] = c.temperature
        if c.seed is not None:
            kwargs["seed"] = c.seed
        if c.max_output_tokens is not None:
            kwargs[c.max_tokens_parameter] = c.max_output_tokens
        if c.structured_output:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "decision", "schema": schema, "strict": True}}
        if c.extra_body:
            kwargs["extra_body"] = c.extra_body
        try:
            completion = self.client.chat.completions.create(**kwargs)
        except openai.APIError as e:
            raise BackendError(f"{type(e).__name__}: {e}") from e
        choice = completion.choices[0]
        usage = {"model": completion.model, "finish_reason": choice.finish_reason}
        if completion.usage:
            usage.update(input_tokens=completion.usage.prompt_tokens, output_tokens=completion.usage.completion_tokens)
        return BackendResponse(content=choice.message.content or "", usage=usage)
