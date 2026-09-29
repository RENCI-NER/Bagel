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
4. **Evaluation first.** The first version is a vertical slice: all stages, one API backend plus `claude -p`, and the batch runner with metrics on the human-assessed MedMentions items. Next come full MedMentions, Azure and Jev. After that, the service, and switching ORION and LitCoin over.

## Stages

A mention is its text plus an optional character span and an optional type hint.

| # | Stage | Input → output |
|---|---|---|
| 0 | Recognize (outside the core) | document → mentions |
| 1 | Prepare | document → abbreviation map; mention → query strings (as written, expanded) |
| 2 | Retrieve | query strings (+ type hint) → ordered, normalized candidates |
| 3 | Adjudicate | document + mention + candidates → typed decision (SKOS relation per candidate, optional confidence) |
| 4 | Select | decision → `linked` (CURIE, relation, confidence) or `abstained` (reason) |

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

Gold comes from human assessments (agree/disagree per mention × CURIE) where available, then from MedMentions. A selected CURIE nobody has assessed is reported as **unassessed**, never counted as right or wrong. Assessment data contains assessor names and stays outside this repository; runs read it from a path.
