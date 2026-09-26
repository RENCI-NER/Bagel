"""
Snapshot every commit of every prompt in the `bagel` LangChain Hub namespace into ./prompts,
so the server and benchmark scripts can run without network access to smith.langchain.com.

Usage:
    python scripts/sync_prompts_from_hub.py [--owner bagel] [--out prompts]

Layout written:
    prompts/index.json                 {prompt_name: {"latest": <hash>, "commits": [<hash>, ...newest first]}}
    prompts/<prompt_name>/<hash>.json  {"name", "owner", "commit_hash", "parent_commit_hash", "created_at", "manifest"}

`manifest` is the LangChain-serialized ChatPromptTemplate exactly as the hub returns it.
"""
import argparse
import json
import os

import requests

API = "https://api.smith.langchain.com"


def list_repos(owner):
    resp = requests.get(f"{API}/repos/", params={"tenant_handle": owner, "limit": 100})
    resp.raise_for_status()
    return [r["repo_handle"] for r in resp.json()["repos"]]


def list_commits(owner, name):
    resp = requests.get(f"{API}/commits/{owner}/{name}/", params={"limit": 100})
    resp.raise_for_status()
    return resp.json()["commits"]


def get_manifest(owner, name, commit_hash):
    resp = requests.get(f"{API}/commits/{owner}/{name}/{commit_hash}")
    resp.raise_for_status()
    return resp.json()["manifest"]


def main(owner, out_dir):
    index_path = os.path.join(out_dir, "index.json")
    index = {}
    if os.path.exists(index_path):
        # keep prompts that only exist locally (built by scripts/build_local_prompts.py)
        with open(index_path) as stream:
            index = {name: entry for name, entry in json.load(stream).items() if entry.get("source") == "local"}
    for name in sorted(list_repos(owner)):
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        commits = list_commits(owner, name)
        if index.get(name, {}).get("source") == "local":
            raise ValueError(f"Hub prompt {owner}/{name} has the same name as a local prompt; rename one of them.")
        index[name] = {"latest": commits[0]["commit_hash"], "commits": [c["commit_hash"] for c in commits]}
        for commit in commits:
            record = {
                "name": name,
                "owner": owner,
                "commit_hash": commit["commit_hash"],
                "parent_commit_hash": commit.get("parent_commit_hash"),
                "created_at": commit["created_at"],
                "manifest": get_manifest(owner, name, commit["commit_hash"]),
            }
            with open(os.path.join(out_dir, name, commit["commit_hash"] + ".json"), "w") as stream:
                json.dump(record, stream, indent=2)
                stream.write("\n")
        print(f"{name}: {len(commits)} commit(s)")
    with open(index_path, "w") as stream:
        json.dump(index, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", default="bagel")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "prompts"))
    args = parser.parse_args()
    main(args.owner, args.out)
