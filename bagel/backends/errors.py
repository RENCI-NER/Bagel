from pydantic import BaseModel, Field


class BackendError(Exception):
    """The model service failed to return an answer (network, auth, rate limit, timeout...)."""


class BackendResponse(BaseModel):
    content: str  # the model's answer, expected to be JSON matching the requested schema
    usage: dict = Field(default_factory=dict)
