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


class TypedTaskSpec(BaseModel):
    """
    A relation task for a typed-decision backend (Jev): one Choice question per candidate about the state
    {text, mention}, with the candidate described in the question. Answers map to SKOS through `labels`, and the
    chosen answer's probability becomes the judgment's confidence.
    """
    name: str
    description: str = ""
    kind: Literal["typed"]
    # per_candidate: one question per candidate, answered with `criteria`, mapped through `labels`.
    # choose_one: one question whose options are the candidates plus `none`; the chosen candidate is exactMatch with
    #   its probability as confidence, the others noMatch.
    mode: Literal["per_candidate", "choose_one"] = "per_candidate"
    candidate_fields: dict[str, str]
    instructions: str
    criteria: dict = {}  # per_candidate: answer -> description (order matters, Jev leans toward the first)
    labels: dict[str, Relation] = {}
    none: str | None = None  # choose_one: description of the "none of these" option
    none_position: Literal["first", "last"] = "first"  # choose_one: where "none" is offered
    hash: str = ""


def load_task(name_or_path: str, spec_type: type[BaseModel] = TaskSpec):
    """
    A task spec by name (bagel/tasks/<name>.yaml) or path. spec_type is TaskSpec or another spec with a hash; a
    linking spec with `kind: typed` loads as a TypedTaskSpec.
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = TASKS_DIR / f"{name_or_path}.yaml"
    text = path.read_text()
    values = yaml.safe_load(text)
    if spec_type is TaskSpec and values.get("kind") == "typed":
        spec_type = TypedTaskSpec
    spec = spec_type(**values)
    spec.hash = hashlib.sha256(text.encode()).hexdigest()
    return spec


def candidate_ids(task: TaskSpec, count: int) -> list[str]:
    if task.candidate_ids == "numbers":
        return [f"c{i:02d}" for i in range(1, count + 1)]
    if count > len(COLORS):
        raise ValueError(f"{count} candidates but only {len(COLORS)} color ids")
    return COLORS[:count]


def candidate_view(task, candidate: Candidate, cid: str | None = None) -> dict:
    """The candidate as shown to a model: task.candidate_fields maps shown keys to attributes ("id" is cid)."""
    view = {}
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
            view[key] = value
    return view


def render_candidates(task: TaskSpec, candidates: list[Candidate], ids: list[str]) -> str:
    rows = [json.dumps(candidate_view(task, c, cid), ensure_ascii=False) for c, cid in zip(candidates, ids)]
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


def adjudicate_typed(task: TypedTaskSpec, backend, document: Document, mention: Mention,
                     candidates: list[Candidate]) -> Decision:
    """Asks a typed-decision backend one Choice question per candidate; every candidate gets a judgment."""
    state = {"text": document.text, "mention": mention.text}
    ids = [f"c{i:02d}" for i in range(1, len(candidates) + 1)]
    if task.mode == "choose_one":
        options = {cid: candidate_view(task, c) for cid, c in zip(ids, candidates)}
        criteria = {"none": task.none, **options} if task.none_position == "first" else {**options, "none": task.none}
        questions = {"choice": {"instructions": task.instructions, "criteria": criteria}}
    else:
        questions = {cid: {"instructions": {"candidate": candidate_view(task, c), "question": task.instructions},
                           "criteria": task.criteria} for cid, c in zip(ids, candidates)}
    start = time.perf_counter()
    try:
        result = backend.ask(state, questions)
    except BackendError as e:
        return Decision(status="api_error", error=str(e), seconds=time.perf_counter() - start)
    seconds = time.perf_counter() - start
    judgments = []
    if task.mode == "choose_one":
        answer = result["answers"]["choice"]
        for cid, candidate in zip(ids, candidates):
            chosen = answer["choice"] == cid
            judgments.append(Judgment(curie=candidate.curie, relation=Relation.EXACT if chosen else Relation.NONE,
                                      confidence=answer["probabilities"][cid] if chosen else None,
                                      probabilities={Relation.EXACT.value: answer["probabilities"][cid]}))
        return Decision(status="ok", judgments=judgments, raw=json.dumps(result["answers"]),
                        usage={"model": result["model"], **result["usage"]}, seconds=seconds)
    for qid, candidate in zip(questions, candidates):
        answer = result["answers"][qid]
        judgments.append(Judgment(
            curie=candidate.curie, relation=task.labels[answer["choice"]], confidence=answer["probabilities"][answer["choice"]],
            probabilities={task.labels[k].value: v for k, v in answer["probabilities"].items()}))
    return Decision(status="ok", judgments=judgments, raw=json.dumps(result["answers"]),
                    usage={"model": result["model"], **result["usage"]}, seconds=seconds)


def adjudicate(task: TaskSpec | TypedTaskSpec, backend: Backend, document: Document, mention: Mention,
               candidates: list[Candidate]) -> Decision:
    if isinstance(task, TypedTaskSpec):
        return adjudicate_typed(task, backend, document, mention, candidates)
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
