"""
Stage 1 (prepare): abbreviations. Extraction asks a model which abbreviations a document defines, once per document;
expansion rewrites a mention with that document's map before retrieval.
"""
import copy
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import jsonschema
from pydantic import BaseModel, Field

from bagel.adjudicate import Message, load_task
from bagel.backends import Backend, BackendConfig, BackendError, make_backend
from bagel.types import Document

_WORD_CHAR = r"A-Za-z0-9"


def clean_abbreviation_map(pairs: list[dict]) -> dict[str, str]:
    """
    Builds {abbreviation: definition} from extractor output ([{"abbreviation", "definition"}]).

    Drops entries that can't be expanded safely: single characters and purely numeric/punctuation "abbreviations"
    (extractors return compound numbers like "3"), definitions no longer than the abbreviation (pairs returned the
    wrong way round, "Duchenne muscular dystrophy" -> "DMD"), and definitions that contain the abbreviation as a word
    (expanding them never terminates). When an abbreviation is defined twice, the first definition wins.
    """
    cleaned = {}
    for pair in pairs:
        abbreviation = pair["abbreviation"].strip()
        definition = pair["definition"].strip()
        if len(abbreviation) < 2 or not re.search(r"[A-Za-z]", abbreviation):
            continue
        if len(definition) <= len(abbreviation) or _word_pattern([abbreviation]).search(definition):
            continue
        cleaned.setdefault(abbreviation, definition)
    return cleaned


def load_abbreviation_maps(path: str | Path) -> dict[str, dict[str, str]]:
    """
    Reads a JSONL file of {"document", "abbreviation_map": [...]} (written by `bagel abbreviations`) into
    {document id: cleaned map}. Files from BagelWithCheese, keyed by "pmid", read the same way.
    """
    maps = {}
    with open(path) as stream:
        for line in stream:
            record = json.loads(line)
            document = record["document"] if "document" in record else record["pmid"]
            maps[str(document)] = clean_abbreviation_map(record["abbreviation_map"])
    return maps


_NOT_A_DEFINITION = re.compile(r"[=<>±]|\bet al\b")


def find_parentheticals(text: str) -> list[dict]:
    """
    Every parenthetical that could define an abbreviation, "long form (SF)" or "SF (long form)", in order of first
    occurrence: [{"start", "end", "inner", "before"}], with start/end the offsets of `inner` and `before` the text
    leading up to the parenthesis (up to 80 characters).

    A parenthesis counts only when it follows whitespace or starts the text, so parts of names such as "Fe(III)" and
    "Na(+)" aren't parentheticals; nested parentheses stay inside the outer one. Left out, since they can't define
    an abbreviation: contents with no letter (numbers, years) and statistics or citations (containing =, <, >, ±
    or "et al"). Repeats of the same contents in a document are left out after the first.
    """
    found, seen, depth, open_at = [], set(), 0, None
    for i, char in enumerate(text):
        if char == "(":
            if depth == 0:
                if i > 0 and not text[i - 1].isspace():
                    continue
                open_at = i
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                inner = text[open_at + 1:i].strip()
                if inner and inner not in seen and re.search(r"[A-Za-z]", inner) and \
                        not _NOT_A_DEFINITION.search(inner):
                    seen.add(inner)
                    start = text.index(inner, open_at + 1)
                    found.append({"start": start, "end": start + len(inner), "inner": inner,
                                  "before": text[max(0, open_at - 80):open_at].rstrip()})
    return found


def _word_pattern(words):
    # longest first, so "CPA" is tried before "Pa"
    alternatives = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![{_WORD_CHAR}])(?:{alternatives})(?![{_WORD_CHAR}])")


def expand(text: str, abbreviations: dict[str, str]) -> str:
    """
    Replaces whole-word abbreviations in text with their definitions, repeating until nothing changes so nested
    abbreviations are expanded too ("CPA" -> "chronic Pa infection" -> "chronic Pseudomonas aeruginosa infection").
    A definition directly followed by its own abbreviation in parentheses, e.g. "lactate dehydrogenase (LDH)", becomes
    the definition alone rather than "lactate dehydrogenase (lactate dehydrogenase)".
    """
    if not abbreviations:
        return text
    words = _word_pattern(abbreviations)
    for _ in range(len(abbreviations) + 1):
        for abbreviation, definition in abbreviations.items():
            text = re.sub(rf"{re.escape(definition)}\s*\({re.escape(abbreviation)}\)", definition, text)
        expanded = words.sub(lambda m: abbreviations[m.group(0)], text)
        if expanded == text:
            return text
        text = expanded
    raise ValueError(f"Abbreviation expansion did not terminate for {text!r}")


class AbbreviationTask(BaseModel):
    """
    An abbreviation prompt and the schema of its answer. kind "extract": the model lists the abbreviations a text
    defines ({text}; answer {"abbreviations": [...]}). kind "classify": the model labels each parenthetical found by
    find_parentheticals as same / instance / other ({text}, {parentheticals}; answer {"parentheticals": [...]}).
    """
    name: str
    description: str = ""
    kind: Literal["extract", "classify"] = "extract"
    messages: list[Message]
    output_schema: dict
    hash: str = ""


class Extraction(BaseModel):
    document: str
    status: Literal["ok", "api_error", "parse_error"]
    abbreviation_map: list[dict] = Field(default_factory=list)  # pairs found in the text
    ungrounded: list[dict] = Field(default_factory=list)  # pairs the model returned that the text doesn't contain
    classified: list[dict] = Field(default_factory=list)  # classify tasks: each parenthetical with its label
    error: str | None = None
    raw: str | None = None
    usage: dict = Field(default_factory=dict)
    seconds: float | None = None


def load_abbreviation_task(name_or_path: str = "abbreviations_v1") -> AbbreviationTask:
    return load_task(name_or_path, AbbreviationTask)


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


def grounded(pair: dict, text: str) -> bool:
    """The abbreviation appears in the text as a whole word and the definition appears (ignoring case and spacing)."""
    return bool(_word_pattern([pair["abbreviation"].strip()]).search(text)) and \
        _squash(pair["definition"]) in _squash(text)


def build_abbreviation_request(task: AbbreviationTask, document: Document):
    """Returns (messages, output schema, parentheticals by id); parentheticals only for classify tasks."""
    schema = copy.deepcopy(task.output_schema)
    values = {"text": document.text}
    found = {}
    if task.kind == "classify":
        found = {f"p{i}": p for i, p in enumerate(find_parentheticals(document.text), 1)}
        values["parentheticals"] = "\n".join(
            json.dumps({"id": pid, "context": f"…{p['before'][-60:]} ({p['inner']})"}, ensure_ascii=False)
            for pid, p in found.items())
        schema["properties"]["parentheticals"]["items"]["properties"]["id"]["enum"] = list(found)
    messages = [{"role": m.role, "content": m.template.format(**values)} for m in task.messages]
    return messages, schema, found


def _ground(document: Document, pairs: list[dict], **fields) -> Extraction:
    return Extraction(document=document.id, status="ok", abbreviation_map=[p for p in pairs if grounded(p, document.text)],
                      ungrounded=[p for p in pairs if not grounded(p, document.text)], **fields)


def parse_extraction(task: AbbreviationTask, document: Document, content: str, schema: dict | None = None,
                     found: dict | None = None) -> Extraction:
    """
    Strictly parses an answer: invalid JSON or a schema violation is a parse_error, and so is a classify answer
    that labels a parenthetical twice or leaves one out. Definitions are asked for exactly as written, so pairs not
    found in the text are kept apart (ungrounded) rather than used.
    """
    schema = schema or task.output_schema
    try:
        value = json.loads(content)
        jsonschema.validate(value, schema)
    except (json.JSONDecodeError, jsonschema.ValidationError) as e:
        return Extraction(document=document.id, status="parse_error", error=str(e).splitlines()[0], raw=content)
    if task.kind == "extract":
        pairs = [{"abbreviation": p["abbreviation"], "definition": p["definition"]} for p in value["abbreviations"]]
        return _ground(document, pairs, raw=content)
    ids = [p["id"] for p in value["parentheticals"]]
    if sorted(ids) != sorted(found):
        return Extraction(document=document.id, status="parse_error", raw=content,
                          error=f"labeled {sorted(set(ids))}, expected each of {sorted(found)} once")
    classified, pairs = [], []
    for p in value["parentheticals"]:
        classified.append({**found[p["id"]], "label": p["label"], "short_form": p["short_form"],
                           "long_form": p["long_form"]})
        if p["label"] == "same":
            pairs.append({"abbreviation": p["short_form"], "definition": p["long_form"]})
    classified.sort(key=lambda c: c["start"])
    return _ground(document, pairs, raw=content, classified=classified)


def extract(task: AbbreviationTask, backend: Backend, document: Document) -> Extraction:
    """
    The abbreviations a document defines. Backend failures are status api_error, never an empty map. A classify
    task with no parentheticals to label makes no model call.
    """
    messages, schema, found = build_abbreviation_request(task, document)
    if task.kind == "classify" and not found:
        return Extraction(document=document.id, status="ok", seconds=0.0)
    start = time.perf_counter()
    try:
        response = backend.complete(messages, schema)
    except BackendError as e:
        return Extraction(document=document.id, status="api_error", error=str(e), seconds=time.perf_counter() - start)
    extraction = parse_extraction(task, document, response.content, schema, found)
    extraction.usage, extraction.seconds = response.usage, time.perf_counter() - start
    return extraction


def extractor_fingerprint(task: AbbreviationTask, backend: BackendConfig) -> str:
    return hashlib.sha256(json.dumps([task.hash, backend.model_dump(mode="json")], sort_keys=True).encode()).hexdigest()


def extract_documents(documents: str | Path, task: AbbreviationTask, backend: BackendConfig, out: str | Path,
                      concurrency: int = 4, limit: int | None = None, progress=print) -> list[Extraction]:
    """
    Extracts abbreviations for every document in a JSONL of {"id", "text"} (e.g. an EL_benchmarks documents.jsonl),
    appending one record per document to `out`. Resumable: documents already in `out` are skipped, and `out` must
    come from the same task and backend. Failed documents aren't written, so a rerun retries them; they are returned.
    """
    fingerprint = extractor_fingerprint(task, backend)
    out = Path(out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            record = json.loads(line)
            if record.get("fingerprint") != fingerprint:
                raise ValueError(f"{out} holds extractions from a different task or backend; use a new output file")
            done.add(record["document"])
    with open(documents) as stream:
        todo = [Document(id=str(r["id"]), text=r["text"]) for r in map(json.loads, stream)]
    todo = [d for d in todo if d.id not in done][:limit]
    engine = make_backend(backend)
    failures = []
    with open(out, "a") as sink, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(extract, task, engine, d) for d in todo]
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result.status == "ok":
                sink.write(json.dumps({"document": result.document, "abbreviation_map": result.abbreviation_map,
                                       "ungrounded": result.ungrounded, "classified": result.classified, "task": task.name, "fingerprint": fingerprint,
                                       "model": result.usage.get("model"), "usage": result.usage,
                                       "seconds": result.seconds}) + "\n")
                sink.flush()
            else:
                failures.append(result)
            progress(f"[{count}/{len(todo)}] {result.document}: {result.status} {len(result.abbreviation_map)} "
                     f"abbreviations, {len(result.ungrounded)} ungrounded {result.error or ''}")
    return failures


class AbbreviationCheckTask(BaseModel):
    """A typed question asked of each extracted pair: instructions and Choice criteria (must include "same")."""
    name: str
    description: str = ""
    instructions: str
    criteria: dict
    hash: str = ""


def load_check_task(name_or_path: str = "abbreviations_check_jev_v1") -> AbbreviationCheckTask:
    return load_task(name_or_path, AbbreviationCheckTask)


def defining_sentence(text: str, pair: dict) -> str:
    """The sentence(s) where a pair's definition first appears, extended to include the short form if it follows."""
    words = [re.escape(w) for w in pair["definition"].split()]
    match = re.search(r"\s+".join(words), text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"definition {pair['definition']!r} not in the text; check only grounded pairs")
    previous = text.rfind(". ", 0, match.start())
    start = 0 if previous == -1 else previous + 2
    after = _word_pattern([pair["abbreviation"].strip()]).search(text, match.end())
    end_from = after.end() if after and after.start() - match.end() < 200 else match.end()
    end = text.find(". ", end_from)
    return text[start:len(text) if end == -1 else end + 1].strip()


def check_pairs(task: AbbreviationCheckTask, backend, document: Document, pairs: list[dict],
                concurrency: int = 8) -> list[dict]:
    """Asks the typed backend about each pair; returns [{"abbreviation", "definition", "choice", "probabilities"}]."""
    def ask(pair):
        state = {"text": defining_sentence(document.text, pair), "short_form": pair["abbreviation"],
                 "long_form": pair["definition"]}
        answer = backend.choose(state, task.instructions, task.criteria)
        return {**pair, "choice": answer["choice"], "probabilities": answer["probabilities"],
                "model": answer["model"], "usage": answer["usage"]}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(ask, pairs))


def check_documents(extractions: str | Path, documents: str | Path, task: AbbreviationCheckTask, backend_config,
                    out: str | Path, min_same: float = 0.5, progress=print) -> list[str]:
    """
    Checks every pair in an abbreviation file (from `bagel abbreviations`) and writes a new file in the same format:
    abbreviation_map keeps the pairs whose probability of "same" is at least min_same, and `checked` records every
    pair's answer so other thresholds can be evaluated without asking again. Resumable like extract_documents;
    documents whose checks fail aren't written (rerun to retry) and are returned.
    """
    from bagel.backends.jev import JevBackend
    source = hashlib.sha256(Path(extractions).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps([task.hash, backend_config.model_dump(mode="json"), min_same, source],
                                            sort_keys=True).encode()).hexdigest()
    out = Path(out)
    done = set()
    if out.exists():
        for record in map(json.loads, out.read_text().splitlines()):
            if record.get("fingerprint") != fingerprint:
                raise ValueError(f"{out} holds checks from a different input, task, backend or threshold")
            done.add(record["document"])
    texts = {str(d["id"]): d["text"] for d in map(json.loads, Path(documents).read_text().splitlines())}
    backend = JevBackend(backend_config)
    failures = []
    with open(out, "a") as sink:
        for record in map(json.loads, Path(extractions).read_text().splitlines()):
            if record["document"] in done:
                continue
            document = Document(id=record["document"], text=texts[record["document"]])
            try:
                checked = check_pairs(task, backend, document, record["abbreviation_map"])
            except BackendError as e:
                failures.append(record["document"])
                progress(f"{record['document']}: api_error {e}")
                continue
            kept = [{"abbreviation": c["abbreviation"], "definition": c["definition"]} for c in checked
                    if c["probabilities"]["same"] >= min_same]
            sink.write(json.dumps({**record, "abbreviation_map": kept, "checked": checked, "check_task": task.name,
                                   "min_same": min_same, "fingerprint": fingerprint}) + "\n")
            sink.flush()
            progress(f"{record['document']}: kept {len(kept)} of {len(checked)}")
    return failures
