"""Stage 3 (adjudicate): ask a model how each candidate relates to the mention; parse strictly into SKOS relations."""
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Literal

import jsonschema
import yaml
from pydantic import BaseModel

from bagel.backends import Backend, BackendError
from bagel.types import Candidate, Decision, Document, Judgment, Mention, Relation

TASKS_DIR = Path(__file__).parent / "tasks"

# Candidate ids used by the prompts developed so far. Order matters: ids are assigned by candidate position.
COLORS = ['alizarin', 'amaranth', 'amber', 'amethyst', 'apricot', 'aqua', 'aquamarine', 'asparagus', 'auburn', 'azure',
          'beige', 'bistre', 'black', 'blue', 'blue-green', 'blue-violet', 'bondi-blue', 'brass', 'bronze', 'brown',
          'buff', 'burgundy', 'camouflage-green', 'caput-mortuum', 'cardinal', 'carmine', 'carrot-orange', 'celadon',
          'cerise', 'cerulean', 'champagne', 'charcoal', 'chartreuse', 'cherry-blossom-pink', 'chestnut', 'chocolate',
          'cinnabar', 'cinnamon', 'cobalt', 'copper', 'coral', 'corn', 'cornflower', 'cream', 'crimson', 'cyan']


class Message(BaseModel):
    role: Literal["system", "user"]
    template: str


class TaskSpec(BaseModel):
    """A prompt plus everything needed to interpret its answers."""
    name: str
    description: str = ""
    candidate_ids: Literal["colors", "numbers"]
    # key shown to the model -> candidate attribute; "id" is the assigned id, "taxa" is joined labels
    candidate_fields: dict[str, str]
    messages: list[Message]
    output_schema: dict
    candidates_key: str  # output key holding the list of judged candidates
    id_key: str
    relation_key: str
    rationale_key: str | None = None
    labels: dict[str, Relation]  # the prompt's relation labels -> SKOS
    omitted: Literal["noMatch", "error"]  # what a candidate missing from the answer means
    restrict_ids: bool = True  # constrain the schema's id field to the ids actually offered
    hash: str = ""


def load_task(name_or_path: str, spec_type: type[BaseModel] = TaskSpec):
    """A task spec by name (bagel/tasks/<name>.yaml) or path; spec_type is TaskSpec or another spec with a hash."""
    path = Path(name_or_path)
    if not path.suffix:
        path = TASKS_DIR / f"{name_or_path}.yaml"
    text = path.read_text()
    spec = spec_type(**yaml.safe_load(text))
    spec.hash = hashlib.sha256(text.encode()).hexdigest()
    return spec


def candidate_ids(task: TaskSpec, count: int) -> list[str]:
    if task.candidate_ids == "numbers":
        return [f"c{i:02d}" for i in range(1, count + 1)]
    if count > len(COLORS):
        raise ValueError(f"{count} candidates but only {len(COLORS)} color ids")
    return COLORS[:count]


def render_candidates(task: TaskSpec, candidates: list[Candidate], ids: list[str]) -> str:
    rows = []
    for candidate, cid in zip(candidates, ids):
        row = {}
        for key, attribute in task.candidate_fields.items():
            if attribute == "id":
                value = cid
            elif attribute == "taxa":
                value = ", ".join(candidate.taxa)
            elif attribute == "category":
                value = candidate.category.removeprefix("biolink:")
            else:
                value = getattr(candidate, attribute)
            if value:
                row[key] = value
        rows.append(json.dumps(row, ensure_ascii=False))
    return "[\n" + ",\n".join(rows) + "\n]"


def build_request(task: TaskSpec, document: Document, mention: Mention, candidates: list[Candidate]):
    """Returns (messages, output schema, candidate ids) for one mention."""
    ids = candidate_ids(task, len(candidates))
    values = {"text": document.text, "mention": mention.text, "candidates": render_candidates(task, candidates, ids)}
    messages = [{"role": m.role, "content": m.template.format(**values)} for m in task.messages]
    schema = copy.deepcopy(task.output_schema)
    if task.restrict_ids:
        item = schema["properties"][task.candidates_key]["items"]
        item["properties"][task.id_key]["enum"] = ids
    return messages, schema, ids


def parse_decision(task: TaskSpec, content: str, schema: dict, ids: list[str], candidates: list[Candidate]) -> Decision:
    """Strictly parses a model answer: invalid JSON or a schema violation is a parse_error, never a guess."""
    try:
        value = json.loads(content)
        jsonschema.validate(value, schema)
    except (json.JSONDecodeError, jsonschema.ValidationError) as e:
        return Decision(status="parse_error", error=str(e).splitlines()[0], raw=content)
    by_id = dict(zip(ids, candidates))
    relations: dict[str, Relation] = {}
    for item in value[task.candidates_key]:
        cid = item[task.id_key]
        if cid not in by_id:
            return Decision(status="invalid_candidate", error=f"unknown candidate id {cid!r}", raw=content)
        if cid in relations:
            return Decision(status="parse_error", error=f"candidate id {cid!r} judged twice", raw=content)
        label = item[task.relation_key]
        if label not in task.labels:
            return Decision(status="parse_error", error=f"unknown relation label {label!r}", raw=content)
        relations[cid] = task.labels[label]
    missing = [cid for cid in ids if cid not in relations]
    if missing and task.omitted == "error":
        return Decision(status="incomplete", error=f"no relation for {missing}", raw=content)
    judgments = [Judgment(curie=c.curie, relation=relations.get(cid, Relation.NONE)) for cid, c in zip(ids, candidates)]
    rationale = value.get(task.rationale_key) if task.rationale_key else None
    return Decision(status="ok", judgments=judgments, rationale=rationale, raw=content,
                    usage={"omitted": len(missing)})


def adjudicate(task: TaskSpec, backend: Backend, document: Document, mention: Mention,
               candidates: list[Candidate]) -> Decision:
    messages, schema, ids = build_request(task, document, mention, candidates)
    start = time.perf_counter()
    try:
        response = backend.complete(messages, schema)
    except BackendError as e:
        return Decision(status="api_error", error=str(e), seconds=time.perf_counter() - start)
    decision = parse_decision(task, response.content, schema, ids, candidates)
    decision.usage = {**response.usage, **decision.usage}
    decision.seconds = time.perf_counter() - start
    return decision
