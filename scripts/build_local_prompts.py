"""
Compiles the prompt sources in prompts/src/*.yaml into prompt records (the same format as the LangChain Hub
snapshots) and registers them in prompts/index.json.

Each distinct prompt content gets its own commit hash (sha256 of the serialized prompt), so editing a source and
rebuilding adds a new version while earlier versions stay loadable by hash.

Usage:
    python scripts/build_local_prompts.py
"""
import glob
import hashlib
import json
import os
from datetime import datetime, timezone

import yaml
from langchain_core.load import dumpd
from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def build(source_path):
    with open(source_path) as stream:
        source = yaml.safe_load(stream)
    template = ChatPromptTemplate.from_messages([(m["role"], m["template"]) for m in source["messages"]])
    manifest = dumpd(template)
    extras = {k: source[k] for k in ("status", "description", "response_format") if k in source}
    commit_hash = hashlib.sha256(json.dumps([manifest, extras], sort_keys=True).encode()).hexdigest()
    name = source["name"]
    os.makedirs(os.path.join(PROMPTS_DIR, name), exist_ok=True)
    record_path = os.path.join(PROMPTS_DIR, name, commit_hash + ".json")
    if not os.path.exists(record_path):
        record = {
            "name": name,
            "owner": "bagel",
            "source": "local",
            "source_file": os.path.relpath(source_path, PROMPTS_DIR),
            "commit_hash": commit_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **extras,
            "manifest": manifest,
        }
        with open(record_path, "w") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    return name, commit_hash


def main():
    index_path = os.path.join(PROMPTS_DIR, "index.json")
    with open(index_path) as stream:
        index = json.load(stream)
    for source_path in sorted(glob.glob(os.path.join(PROMPTS_DIR, "src", "*.yaml"))):
        name, commit_hash = build(source_path)
        entry = index.setdefault(name, {"latest": commit_hash, "commits": [], "source": "local"})
        if commit_hash not in entry["commits"]:
            entry["commits"].insert(0, commit_hash)
        entry["latest"] = commit_hash
        print(f"{name}: {commit_hash[:8]}")
    with open(index_path, "w") as stream:
        json.dump(index, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
