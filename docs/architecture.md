# BAGEL architecture

BAGEL (Babel Assisted Generative Entity Linker) links mentions in text to Babel/NodeNorm identifiers. Named-entity recognition is out of scope: BAGEL takes a document plus mentions, whether from gold annotations (MedMentions, CRAFT) or from an upstream recognizer (e.g. the LitCoin extraction pipeline).

This package replaces five separate linker implementations:
- the old Bagel service;
- ORION's in-process copy;
- BagelWithCheese;
- the ncats/litcoin pipeline's `normalize.py`;
- the Bagel `abbreviations` branch.

ORION, the LitCoin pipeline and the evaluation all use this one package.

## Decisions

1. **One repository.** RENCI-NER/Bagel holds the core library, the batch/evaluation runner and (later) the service. Older code is recovered from git history if needed; nothing is kept as `legacy/`.
2. **Linking strategies.** `link()` runs a *strategy*. The first is the pipeline below. An agentic strategy, where a model does its own retrieval with tools, can be added later behind the same interface and evaluated the same way. CLI harnesses (`claude -p`, `codex exec`) are model backends of the pipeline, not a separate mode.
3. **SKOS relations.** Everything BAGEL emits or stores uses `exactMatch`, `broadMatch`, `narrowMatch`, `relatedMatch` or `noMatch`, with the **mention as subject** and the candidate as object. `broadMatch` means the candidate is broader than the mention. Prompts and backends may use any wording; each task spec maps its labels to SKOS when the response is parsed.
4. **Evaluation first.** The first version is a vertical slice: all stages, OpenAI-compatible, Azure and `claude -p` backends, and the batch runner with metrics on the EL_benchmarks sets. Next come Jev and Laya, the service, and switching ORION and LitCoin over.
5. **Benchmarks keep their source identifiers.** Matching BAGEL's output to a benchmark identifier is an evaluation-time choice (identity, or the same Babel clique in a named release), so a benchmark doesn't change meaning with a Babel release.

## Stages

A mention is its text plus an optional character span and an optional type hint.

| # | Stage | Input → output |
|---|---|---|
| 0 | Recognize (outside the core) | document → mentions |
| 1 | Prepare | document → abbreviation map; mention → query strings (as written, expanded) |
| 2 | Retrieve | query strings (+ type hint) → ordered, normalized candidates |
| 3 | Adjudicate | document + mention + candidates → typed decision (SKOS relation per candidate, optional confidence) |
| 4 | Select | decision → `linked` (CURIE, relation, confidence) or `abstained` (reason) |

**Prepare.**
- Abbreviations are extracted once per document, by a model with its own task spec (`abbreviations_v1`) through the same backends. The prompt asks for definitions exactly as written; pairs not found in the text are kept apart as `ungrounded`, not used.
- Cleaning drops pairs that can't be expanded safely: single characters, numbers, definitions no longer than the abbreviation, and definitions that contain it.
- Expansion replaces whole-word abbreviations in the mention, repeating until nested ones are expanded too. A mention of the form "definition (ABBR)" becomes the definition.
- Extracted pairs can be checked by a typed-decision model (Jev): for each pair and its defining sentence, the probability that the short form stands for the same thing as the long form. This catches instances ("T. vaginalis isolates (TV-LACM6)") and wrong long forms at a fraction of a cent per hundred abstracts.
- Batch runs read a precomputed abbreviation file (`bagel abbreviations`), since one document has many mentions and the map shouldn't vary between them.

**Retrieval.**
- NameRes and SapBERT results are merged by CURIE, then normalized with NodeNorm (gene/protein and drug/chemical conflation; taxa).
- CURIEs that normalize to the same clique are merged, keeping each source's best rank.
- Candidates are ordered by best source rank, so the order the model sees is deterministic.

**Adjudication.**
- A *task spec* (prompt template, output JSON schema, label → SKOS mapping, candidate id scheme) is paired with a *backend*.
- Backends take a rendered request and an output schema, and return the parsed value with usage and the raw response. Chat models get a prompt and structured output; a typed-decision model such as Jev would implement the same contract natively.

**Selection.** The policy is data, not code:
- an ordered list of accepted relations (e.g. `exactMatch`, then `broadMatch`);
- a tie-break by retrieval rank;
- an optional minimum confidence.

It covers what ORION's `extract_best_match`, the LitCoin pipeline and exact-only evaluation each did.

## Failures are visible

Every mention ends with an explicit status for each stage:
- `ok`
- `retrieval_empty`
- `api_error`
- `parse_error`: the response isn't valid JSON or violates the schema
- `invalid_candidate`: the response names an id that wasn't offered
- `incomplete`: the task requires a relation for every candidate and some are missing

Nothing is silently turned into "no match". Parsing is strict: a response that doesn't match the task's schema is a `parse_error`, with the raw response kept.

## Caching and runs

- **Retrieval cache:** keyed by query string, type hint and service URLs. Changing the prompt or model reruns only adjudication.
- **Run store:** a batch run is one SQLite file holding one row per mention, with each stage's output, status and provenance: task spec hash, backend, model, parameters, timing and usage. Runs are resumable, since finished mentions are skipped.

## Evaluation

Metrics are computed from run files, stage by stage:
- **Retrieval recall:** is a correct CURIE among the candidates?
- **Adjudication:** of the mentions where a correct CURIE was retrieved, how often it was labeled `exactMatch`.
- **Selection:**
  - accuracy of linked mentions;
  - coverage (fraction linked);
  - abstention when no correct CURIE was retrieved;
  - how often several candidates were labeled `exactMatch`.
- **Operational:** status counts, latency, token usage.

Benchmarks come from the EL_benchmarks repository: a `truth.jsonl` of mentions with judged identifiers, in the source's own vocabulary, plus a `documents.jsonl` of full texts.
- **Matching:** a *matcher* decides whether a produced identifier matches a judged one. The identity matcher needs the same identifier. The Babel matcher also accepts the same NodeNorm clique, with or without conflation, and records the Babel release.
- **Open and closed world:** each item says which it is. In an open-world benchmark (MedMentions-assessed), a link nobody judged is **unjudged**, never counted as right or wrong. In a closed-world one (CRAFT), it is wrong.
- **Unreachable gold:** a judged identifier the matcher can't resolve, e.g. one missing from the Babel release, is counted separately so it isn't mistaken for a retrieval failure.

## Planned backends

- **Typed-decision models: Jev and Laya.** They return typed values (choices, scores) with calibrated confidences instead of text, so each needs a native backend returning a Decision with `confidence`, not a chat prompt. Both fit the "which candidate is the mention, or none" decision shape.
  - [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (TypeSafe) is a hosted API.
  - [Laya](https://huggingface.co/convaiinnovations/laya) (ConvAI Innovations) is open source (Apache 2.0): ModernBERT-large, 421M parameters, `pip install laya`, CPU or GPU. Its model card reports 0.36 zero-shot vs 0.77 fine-tuned on its own benchmark, so plan on fine-tuning it on the human assessments, keeping training and evaluation mentions separate.
