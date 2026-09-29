"""Batch runs: link every mention of a dataset and store the results in one SQLite file. Resumable."""
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel, Field

from bagel.adjudicate import load_task
from bagel.backends import BackendConfig, make_backend
from bagel.cache import Cache
from bagel.pipeline import PipelineConfig, link
from bagel.types import Document, Mention, MentionResult


class RunConfig(BaseModel):
    dataset: Path
    task: str  # task spec name (bagel/tasks/<name>.yaml) or path
    backend: BackendConfig
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    cache: Path | None = None  # retrieval cache (SQLite), shareable across runs
    concurrency: int = 4
    limit: int | None = None


def load_run_config(path: str | Path) -> RunConfig:
    return RunConfig(**yaml.safe_load(Path(path).read_text()))


def load_items(path: str | Path, limit: int | None = None) -> list[dict]:
    with open(path) as stream:
        items = [json.loads(line) for line in stream]
    return items[:limit] if limit else items


def _fingerprint(config: RunConfig, task_hash: str) -> str:
    # what determines results; concurrency, limit and cache location don't
    relevant = config.model_dump(mode="json", exclude={"concurrency", "limit", "cache"})
    return hashlib.sha256(json.dumps([relevant, task_hash], sort_keys=True).encode()).hexdigest()


class RunStore:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS results (item_id TEXT PRIMARY KEY, status TEXT NOT NULL, "
                        "selection TEXT NOT NULL, seconds REAL NOT NULL, record TEXT NOT NULL)")
        self.db.commit()

    def meta(self) -> dict:
        return {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM meta")}

    def set_meta(self, **values) -> None:
        self.db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                            [(k, json.dumps(v, default=str)) for k, v in values.items()])
        self.db.commit()

    def done(self) -> set[str]:
        return {row[0] for row in self.db.execute("SELECT item_id FROM results")}

    def add(self, item_id: str, result: MentionResult, seconds: float) -> None:
        self.db.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?)",
                        (item_id, result.status, result.selection.status, seconds, result.model_dump_json()))
        self.db.commit()

    def results(self) -> dict[str, MentionResult]:
        return {item_id: MentionResult.model_validate_json(record)
                for item_id, record in self.db.execute("SELECT item_id, record FROM results")}


def run(config: RunConfig, out_path: str | Path, progress=print) -> RunStore:
    task = load_task(config.task)
    store = RunStore(out_path)
    fingerprint = _fingerprint(config, task.hash)
    existing = store.meta().get("fingerprint")
    if existing is not None and existing != fingerprint:
        raise ValueError(f"{out_path} holds a run with a different configuration; use a new output file")
    store.set_meta(fingerprint=fingerprint, config=config.model_dump(mode="json"), task=task.name,
                   task_hash=task.hash)
    items = [i for i in load_items(config.dataset, config.limit) if i["id"] not in store.done()]
    backend = make_backend(config.backend)
    cache = Cache(config.cache) if config.cache else None

    def work(item):
        start = time.perf_counter()
        result = link(Document(**item["document"]), Mention(**item["mention"]), item.get("abbreviations", {}),
                      task, backend, config.pipeline, client, cache)
        return item["id"], result, time.perf_counter() - start

    with httpx.Client(timeout=config.pipeline.retrieval.timeout, transport=httpx.HTTPTransport(retries=3)) as client, \
            ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures = [pool.submit(work, item) for item in items]
        for count, future in enumerate(as_completed(futures), 1):
            item_id, result, seconds = future.result()
            store.add(item_id, result, seconds)
            progress(f"[{count}/{len(items)}] {item_id}: {result.status} / {result.selection.status} "
                     f"{result.selection.curie or ''} ({seconds:.1f}s)")
    return store
