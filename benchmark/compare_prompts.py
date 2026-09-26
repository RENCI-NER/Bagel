"""
Small side-by-side comparison of prompts on hand-written cases with gold identifiers.

Candidates are retrieved once per case (NameRes + SapBERT + NodeNorm, cached in --cache) so every prompt sees the
same candidate list. Each prompt is then run against an Ollama or OpenAI-compatible chat endpoint.

Scoring per case: correct if exactly one candidate is labeled exact and it is in `gold`, or, when `gold` is empty
(the right concept was not retrieved), if no candidate is labeled exact.

Example:
    python benchmark/compare_prompts.py --cases benchmark/prompt_cases.jsonl \\
        --prompts bagel/ask_classes bagel/ask_classes_no_system bagel/ask_classes_v2 \\
        --ollama-url http://localhost:11434 --model qwen3:14b --out /tmp/compare.jsonl
"""
import argparse
import asyncio
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from llm_output import parse_llm_json, map_color_codes, LLMOutputError  # noqa: E402
from models import SynonymListContext, Entity  # noqa: E402
from prompt import load_prompt, parse_prompt_name, resolve_commit, _load_local  # noqa: E402
from util.ner_util import get_entity_ids  # noqa: E402

NAMERES = "https://name-resolution-sri.renci.org/lookup?autocomplete=false&offset=0&string="
SAPBERT = "https://sap-qdrant.apps.renci.org/annotate/"
NODENORM = "https://nodenormalization-sri.renci.org/get_normalized_nodes"


async def retrieve(cases, cache_path, count):
    cache = json.load(open(cache_path)) if cache_path and os.path.exists(cache_path) else {}
    async with httpx.AsyncClient(timeout=120) as client:
        for case in cases:
            if case["id"] not in cache:
                cache[case["id"]] = await get_entity_ids(case["entity"], name_res_url=NAMERES, sapbert_url=SAPBERT,
                                                         node_norm_url=NODENORM, session=client, count=count)
    if cache_path:
        json.dump(cache, open(cache_path, "w"), indent=1)
    return cache


def context_for(case, candidates):
    return SynonymListContext(text=case["text"], entity=case["entity"], synonyms=[
        Entity(label=v["name"], identifier=k, description=v.get("description", ""),
               entity_type=v.get("category", ""), taxa=v.get("taxa", "")) for k, v in candidates.items()])


def messages_for(template, context):
    role = {"human": "user", "system": "system", "ai": "assistant"}
    return [{"role": role[m.type], "content": m.content}
            for m in template.format_prompt(**context.prompt_inputs()).to_messages()]


def chat(client, args, messages, response_format=None):
    if args.ollama_url:
        body = {"model": args.model, "messages": messages, "stream": False, "think": False,
                "options": {"temperature": 0, "seed": 0, "num_ctx": 16384}}
        if response_format:
            body["format"] = response_format["json_schema"]["schema"]
        response = client.post(f"{args.ollama_url}/api/chat", json=body)
        response.raise_for_status()
        return response.json()["message"]["content"]
    body = {"model": args.model, "messages": messages, "temperature": 0, "seed": 0}
    if response_format:
        body["response_format"] = response_format
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"}
    response = client.post(f"{args.openai_url}/chat/completions", json=body, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def score(case, entities, problems, error):
    exact = [e["identifier"] for e in entities if e["synonym_type"] == "exact"]
    if error:
        correct = False
    elif case["gold"]:
        correct = len(exact) == 1 and exact[0] in case["gold"]
    else:
        correct = not exact
    return {"correct": correct, "exact": exact, "n_exact": len(exact), "error": error,
            "unknown_ids": sum(p["reason"] == "unknown_color" for p in problems),
            "unmapped": len(problems)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="prompt names; append '+schema' to send the prompt's response_format")
    parser.add_argument("--model", required=True)
    parser.add_argument("--ollama-url")
    parser.add_argument("--openai-url", default="https://api.openai.com/v1")
    parser.add_argument("--cache", default=None, help="candidate cache (json)")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cases = [json.loads(line) for line in open(args.cases) if line.strip()]
    candidates = asyncio.run(retrieve(cases, args.cache, args.count))
    results = []
    with httpx.Client(timeout=900) as client, open(args.out, "w") as out:
        for prompt_spec in args.prompts:
            prompt_name, _, flag = prompt_spec.partition("+")
            template = load_prompt(prompt_name)
            response_format = None
            if flag == "schema":
                name, version = parse_prompt_name(prompt_name)
                with open(os.path.join(os.path.dirname(__file__), "..", "prompts", name,
                                       resolve_commit(name, version) + ".json")) as stream:
                    response_format = json.load(stream)["response_format"]
            for case in cases:
                context = context_for(case, candidates[case["id"]])
                start = time.time()
                content, entities, problems, error = "", [], [], None
                try:
                    content = chat(client, args, messages_for(template, context), response_format)
                    entities, problems = map_color_codes(context.synonyms, parse_llm_json(content))
                except LLMOutputError as e:
                    error = f"parse: {e}"
                except httpx.HTTPError as e:
                    error = f"http: {e}"
                row = {"prompt": prompt_spec, "case": case["id"], "seconds": round(time.time() - start, 1),
                       "gold_retrieved": any(g in candidates[case["id"]] for g in case["gold"]) or not case["gold"],
                       **score(case, entities, problems, error), "response": content}
                results.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"{prompt_spec:40s} {case['id']:14s} {'OK ' if row['correct'] else 'BAD'} "
                      f"exact={row['exact']} err={error} {row['seconds']}s", flush=True)
    print("\nprompt                                    correct  errors  multi-exact  unknown-ids  mean-s")
    for prompt_spec in args.prompts:
        rows = [r for r in results if r["prompt"] == prompt_spec]
        print(f"{prompt_spec:40s} {sum(r['correct'] for r in rows):3d}/{len(rows):<3d}  "
              f"{sum(bool(r['error']) for r in rows):5d}  {sum(r['n_exact'] > 1 for r in rows):10d}  "
              f"{sum(r['unknown_ids'] for r in rows):10d}  {sum(r['seconds'] for r in rows) / len(rows):6.1f}")


if __name__ == "__main__":
    main()
