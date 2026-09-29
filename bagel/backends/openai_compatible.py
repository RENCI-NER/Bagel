"""OpenAI chat completions API and compatible servers (vLLM, Ollama's /v1, ...)."""
import os
from typing import Literal

import openai
from pydantic import BaseModel, Field

from bagel.backends.errors import BackendError, BackendResponse


class OpenAIConfig(BaseModel):
    kind: Literal["openai"] = "openai"
    model: str
    base_url: str | None = None  # None: api.openai.com
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = 0
    seed: int | None = 0
    structured_output: bool = True  # send the output schema as response_format (json_schema, strict)
    extra_body: dict = Field(default_factory=dict)  # server-specific options, e.g. {"think": false} for Ollama
    max_output_tokens: int | None = 4096  # runaway generation ends as a visible parse_error, not a timeout
    timeout: float = 300
    # the SDK retries twice by default, silently; a failure should surface as api_error (runs can --retry-failed)
    max_retries: int = 0


class OpenAICompatibleBackend:
    def __init__(self, config: OpenAIConfig):
        self.config = config
        api_key = os.environ.get(config.api_key_env)
        if api_key is None:
            if config.base_url is None:
                raise ValueError(f"{config.api_key_env} is not set")
            api_key = "unused"  # local servers (Ollama, vLLM) don't check it
        self.client = openai.OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout,
                                    max_retries=config.max_retries)

    def complete(self, messages: list[dict], schema: dict) -> BackendResponse:
        c = self.config
        kwargs = {"model": c.model, "messages": messages}
        if c.temperature is not None:
            kwargs["temperature"] = c.temperature
        if c.seed is not None:
            kwargs["seed"] = c.seed
        if c.max_output_tokens is not None:
            kwargs["max_tokens"] = c.max_output_tokens
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
        usage = {"input_tokens": completion.usage.prompt_tokens, "output_tokens": completion.usage.completion_tokens,
                 "finish_reason": choice.finish_reason} if completion.usage else {"finish_reason": choice.finish_reason}
        return BackendResponse(content=choice.message.content or "", usage=usage)
