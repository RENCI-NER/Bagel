"""Claude Code in print mode (`claude -p`) as a model backend, e.g. to run on a Claude subscription."""
import json
import re
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
    # --json-schema makes Claude Code take two turns and load ~29k tokens of its own context per call. Off by
    # default: the schema goes in the prompt instead, and answers are validated against it either way.
    structured_output: bool = False


_FENCED = re.compile(r"\A\s*```(?:json)?\s*\n(.*)\n\s*```\s*\Z", re.DOTALL)


def strip_fence(text: str) -> str:
    """Removes one code fence wrapping the whole answer (```json ... ```); anything else is left as is."""
    match = _FENCED.match(text)
    return match.group(1) if match else text


class ClaudeCLIBackend:
    def __init__(self, config: ClaudeCLIConfig):
        self.config = config

    def command(self, system: str, schema: dict) -> list[str]:
        """No tools, no saved session, no settings; runs in an empty directory so no project CLAUDE.md is read."""
        command = [self.config.executable, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
                   "--setting-sources", "", "--model", self.config.model, "--system-prompt", system]
        if self.config.structured_output:
            command += ["--json-schema", json.dumps(schema)]
        return command

    def complete(self, messages: list[dict], schema: dict) -> BackendResponse:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or \
            "Answer the user's request using the requested output structure."
        prompt = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        if not self.config.structured_output:
            prompt += ("\n\nRespond with only a JSON object, without code fences, that validates against this JSON "
                       f"schema:\n{json.dumps(schema)}")
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
        content = json.dumps(structured) if structured is not None else strip_fence(output.get("result", ""))
        return BackendResponse(content=content, usage=usage)
