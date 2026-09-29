"""Command line: bagel link | dataset | run | eval."""
import argparse
import json
import sys
from pathlib import Path

import httpx
import yaml

from bagel.adjudicate import load_task
from bagel.backends import make_backend
from bagel.cache import Cache
from bagel.datasets import assessed
from bagel.evaluate import evaluate_baseline, evaluate_run
from bagel.pipeline import PipelineConfig, link
from bagel.runs import RunStore, load_items, load_run_config, run
from bagel.types import Document, Mention


def cmd_link(args):
    backend_config = yaml.safe_load(Path(args.backend).read_text())
    pipeline = PipelineConfig(**yaml.safe_load(Path(args.pipeline).read_text())) if args.pipeline else PipelineConfig()
    text = Path(args.text_file).read_text() if args.text_file else args.text
    abbreviations = json.loads(args.abbreviations) if args.abbreviations else {}
    with httpx.Client(timeout=pipeline.retrieval.timeout, transport=httpx.HTTPTransport(retries=3)) as client:
        result = link(Document(id="input", text=text), Mention(text=args.mention, type_hint=args.type_hint),
                      abbreviations, load_task(args.task), make_backend(backend_config), pipeline, client,
                      Cache(args.cache) if args.cache else None)
    print(result.model_dump_json(indent=2, exclude={"decision": {"raw"}} if not args.raw else None))


def cmd_dataset(args):
    count = assessed.build(args.dump, args.abbreviations, args.out)
    print(f"wrote {count} mentions to {args.out}")


def cmd_run(args):
    config = load_run_config(args.config)
    if args.limit is not None:
        config.limit = args.limit
    run(config, args.out, retry_failed=args.retry_failed)


def _fmt(value):
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def cmd_eval(args):
    items = load_items(args.dataset)
    stores = {Path(path).stem: RunStore(path).results() for path in args.runs}
    if stores:
        # compare everything on the same mentions: those every run has results for
        common = set.intersection(*(set(results) for results in stores.values()))
        items = [i for i in items if i["id"] in common]
        stores = {name: {k: v for k, v in results.items() if k in common} for name, results in stores.items()}
        print(f"evaluating on the {len(items)} mentions all runs have results for\n")
    reports = {name: evaluate_run(results, items) for name, results in stores.items()}
    for model in args.baseline or []:
        reports[f"baseline:{model}"] = evaluate_baseline(model, items)
    if args.json:
        json.dump(reports, sys.stdout, indent=2)
        print()
        return
    rows = [("selection", k) for k in ("n", "linked", "correct", "incorrect", "unassessed", "abstained", "failed",
                                       "coverage", "precision_assessed", "agrees_with_medmentions")]
    rows += [("retrieval", "recall"), ("adjudication_given_retrieved", "exact_on_correct"),
             ("adjudication_given_retrieved", "exact_on_incorrect"), ("adjudication_given_retrieved", "multiple_exact"),
             ("abstention_when_correct_not_retrieved", "abstained")]
    names = list(reports)
    width = max(len(n) for n in names) + 2
    print(f"{'metric':40s}" + "".join(f"{n:>{width}s}" for n in names))
    for section, key in rows:
        print(f"{key:40s}" + "".join(f"{_fmt(reports[n].get(section, {}).get(key)):>{width}s}" for n in names))
    for name in names:
        if "status" in reports[name]:
            print(f"\n{name} status: {reports[name]['status']}  models: {reports[name]['models']}  "
                  f"latency: {reports[name]['latency_seconds']}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bagel", description="Babel Assisted Generative Entity Linker")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("link", help="link one mention")
    p.add_argument("--mention", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    p.add_argument("--task", default="relations_v2", help="task spec name or path")
    p.add_argument("--backend", required=True, help="backend config YAML")
    p.add_argument("--pipeline", help="pipeline config YAML (retrieval, policy)")
    p.add_argument("--type-hint")
    p.add_argument("--abbreviations", help='JSON object, e.g. \'{"CF": "cystic fibrosis"}\'')
    p.add_argument("--cache", help="retrieval cache SQLite file")
    p.add_argument("--raw", action="store_true", help="include the raw model response")
    p.set_defaults(func=cmd_link)

    p = sub.add_parser("dataset", help="build the assessed-MedMentions dataset from a bagel-eval pg_dump")
    p.add_argument("--dump", required=True)
    p.add_argument("--abbreviations", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("run", help="run a batch from a run config YAML")
    p.add_argument("config")
    p.add_argument("--out", required=True, help="run file (SQLite); rerun with the same file to resume")
    p.add_argument("--limit", type=int)
    p.add_argument("--retry-failed", action="store_true", help="rerun mentions that ended in api_error/retrieval_error")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("eval", help="metrics for runs and stored baselines")
    p.add_argument("runs", nargs="*")
    p.add_argument("--dataset", required=True)
    p.add_argument("--baseline", action="append", help="stored baseline model, e.g. gpt-oss (repeatable)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
