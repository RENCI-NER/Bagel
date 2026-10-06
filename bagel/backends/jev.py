"""
Jev (TypeSafe System One): a typed-decision model. It answers typed questions about a state (Choice, Score, Noul)
with probabilities and doesn't generate text, so it isn't a chat backend: it has its own config and an `ask` method
instead of `complete`.
"""
from typing import Literal

from pydantic import BaseModel
from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient, TypeSafeError

from bagel.backends.errors import BackendError
from bagel.backends.secrets import resolve_api_key


class JevConfig(BaseModel):
    kind: Literal["jev"] = "jev"
    model: str  # pin a version (e.g. jev-1.13.0); jev-latest moves when TypeSafe releases a new one
    api_key_keychain: str | None = None  # macOS Keychain service holding the key (takes precedence)
    api_key_env: str | None = "TYPESAFE_API_KEY"
    timeout: float = 30
    # the SDK retries twice by default; a failure should be visible, and callers can retry
    max_retries: int = 0


class JevBackend:
    def __init__(self, config: JevConfig):
        api_key = resolve_api_key(config.api_key_keychain, config.api_key_env)
        if not api_key:
            raise ValueError("No Jev API key: set api_key_keychain or the environment variable named by api_key_env")
        self.config = config
        self.client = TypeSafeClient(api_key=api_key, model=config.model, timeout=config.timeout,
                                     retry=RetryPolicy(max_retries=config.max_retries))

    def ask(self, state, questions: dict[str, dict]) -> dict:
        """
        Several Choice questions about one state ({id: {"instructions", "criteria"}}), answered in one request.
        Returns {"answers": {id: {"choice", "probabilities", "confidence"}}, "model", "usage"}. Service failures
        raise BackendError.
        """
        try:
            response = self.client.system_one(state=state, questions={
                qid: Choice(instructions=q["instructions"], criteria=q["criteria"]) for qid, q in questions.items()})
        except TypeSafeError as e:
            raise BackendError(f"{type(e).__name__}: {e}") from e
        return {"answers": {qid: {"choice": a.choice, "probabilities": dict(a.probabilities), "confidence": a.confidence}
                            for qid, a in response.answers.items()},
                "model": response.model, "usage": response.usage.model_dump()}

    def choose(self, state, instructions, criteria: dict) -> dict:
        """One Choice question: {"choice", "probabilities", "confidence", "model", "usage"}."""
        result = self.ask(state, {"q": {"instructions": instructions, "criteria": criteria}})
        return {**result["answers"]["q"], "model": result["model"], "usage": result["usage"]}
