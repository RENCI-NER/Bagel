"""Claude Code in print mode (`claude -p`) as a model backend, e.g. to run on a Claude subscription."""
import json
import subprocess
import tempfile
from typing import Literal

from pydantic import BaseModel

from bagel.backends.errors import BackendError, BackendResponse


class ClaudeCLIConfig(BaseModel):
    kind: Literal["claude_cli"] = "claude_cli"
    model: str = "sonnet"
    executable: str = "claude"
    timeout: float = 300


class ClaudeCLIBackend:
    def __init__(self, config: ClaudeCLIConfig):
        self.config = config

    def command(self, system: str, schema: dict) -> list[str]:
        """No tools, no saved session, no settings; runs in an empty directory so no project CLAUDE.md is read."""
        return [self.config.executable, "-p", "--output-format", "json", "--json-schema", json.dumps(schema),
                "--tools", "", "--no-session-persistence", "--setting-sources", "", "--model", self.config.model,
                "--system-prompt", system]

    def complete(self, messages: list[dict], schema: dict) -> BackendResponse:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or \
            "Answer the user's request using the requested output structure."
        prompt = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        with tempfile.TemporaryDirectory() as cwd:
            try:
                result = subprocess.run(self.command(system, schema), input=prompt, capture_output=True, text=True,
                                        cwd=cwd, timeout=self.config.timeout)
            except subprocess.TimeoutExpired as e:
                raise BackendError(f"claude -p timed out after {self.config.timeout}s") from e
        if result.returncode != 0 and not result.stdout:
            raise BackendError(f"claude -p exited {result.returncode}: {result.stderr.strip()[:500]}")
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise BackendError(f"claude -p printed non-JSON output: {result.stdout[:500]!r}") from e
        # failures (e.g. expired login) come back as subtype "success" with is_error true
        if output.get("is_error"):
            raise BackendError(f"claude -p: {output.get('result')}")
        usage = output.get("usage", {})
        usage = {"input_tokens": usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
                 + usage.get("cache_creation_input_tokens", 0),
                 "output_tokens": usage.get("output_tokens"),
                 "cost_usd_equivalent": output.get("total_cost_usd")}
        structured = output.get("structured_output")
        content = json.dumps(structured) if structured is not None else output.get("result", "")
        return BackendResponse(content=content, usage=usage)
