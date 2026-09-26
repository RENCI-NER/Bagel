"""
Prompt loading.

Prompts are read from the local ./prompts directory, a snapshot of every commit of the `bagel` namespace on
LangChain Hub (see scripts/sync_prompts_from_hub.py). Names use the hub's format, e.g. "bagel/ask_classes"
(latest commit) or "bagel/ask_classes:4da4ad77" (a specific commit, any unique prefix of its hash).

Set BAGEL_PROMPT_HUB_FALLBACK=true to pull prompts that are not in the local snapshot from LangChain Hub.
"""
import json
import os
import pathlib
from functools import lru_cache

from langchain_core.load import loads
from config import PromptSettings, List, logger

PROMPTS_DIR = pathlib.Path(os.environ.get("BAGEL_PROMPTS_DIR",
                                          pathlib.Path(__file__).resolve().parent.parent / "prompts"))
PROMPT_OWNER = "bagel"


class PromptNotFoundError(KeyError):
    """ The requested prompt or prompt version is not in the local snapshot."""


def parse_prompt_name(prompt_name: str):
    """ Splits "bagel/name:hash", "bagel/name" or "name" into (name, hash_or_empty)."""
    if "/" in prompt_name:
        owner, prompt_name = prompt_name.split("/", 1)
        if owner != PROMPT_OWNER:
            raise ValueError(f"Invalid prompt {owner}/{prompt_name}. "
                             f"Prompts are supported only from {PROMPT_OWNER}/ namespace currently.")
    name, _, version = prompt_name.partition(":")
    return name, version


@lru_cache(maxsize=None)
def _load_index():
    with open(PROMPTS_DIR / "index.json") as stream:
        return json.load(stream)


def list_prompts():
    """ Returns {prompt_name: latest_commit_hash} for all locally available prompts."""
    return {name: value["latest"] for name, value in _load_index().items()}


def resolve_commit(name: str, version: str = "") -> str:
    """ Resolves a (possibly abbreviated or empty) commit hash to a full local commit hash."""
    index = _load_index()
    if name not in index:
        raise PromptNotFoundError(f"Prompt {PROMPT_OWNER}/{name} not found in {PROMPTS_DIR}. Available: {sorted(index)}")
    if not version:
        return index[name]["latest"]
    matches = [c for c in index[name]["commits"] if c.startswith(version)]
    if len(matches) != 1:
        raise PromptNotFoundError(f"Version {version!r} of prompt {PROMPT_OWNER}/{name} "
                       f"{'is ambiguous' if matches else 'not found'}. Available: {index[name]['commits']}")
    return matches[0]


@lru_cache(maxsize=None)
def _load_local(name: str, commit_hash: str):
    with open(PROMPTS_DIR / name / f"{commit_hash}.json") as stream:
        record = json.load(stream)
    obj = loads(json.dumps(record["manifest"]))
    # same metadata hub.pull() attaches; used for naming chains
    obj.metadata = obj.metadata or {}
    obj.metadata.update({
        "lc_hub_owner": record["owner"],
        "lc_hub_repo": record["name"],
        "lc_hub_commit_hash": record["commit_hash"],
    })
    return obj


def load_prompt(prompt_name: str):
    """ Loads a prompt by hub-style name from the local snapshot and returns a prompt object."""
    name, version = parse_prompt_name(str(prompt_name))
    try:
        return _load_local(name, resolve_commit(name, version))
    except PromptNotFoundError:
        if os.environ.get("BAGEL_PROMPT_HUB_FALLBACK", "").lower() not in ("1", "true", "yes"):
            raise
        logger.warning(f"Prompt {prompt_name} not found locally, pulling from LangChain Hub.")
        from langchain import hub
        return hub.pull(f"{PROMPT_OWNER}/{name}" + (f":{version}" if version else ""))


def load_prompts(prompts: List[PromptSettings]):
    # loads all prompts
    prompt_mapping = {}
    for prompt in prompts:
        prompt_mapping[prompt.name] = load_prompt(str(prompt))
    return prompt_mapping
