# BAGEL

Babel Assisted Generative Entity Linker. Given a document and mentions in it, BAGEL links each mention to a Babel/NodeNorm identifier:

1. **Prepare:** find the abbreviations the document defines, and expand them in the mention.
2. **Retrieve:** get candidate identifiers from NameRes and SapBERT, normalized with NodeNorm.
3. **Adjudicate:** ask a model how each candidate relates to the mention.
4. **Select:** pick the link, or abstain, by a configurable policy.

Relations are SKOS (`exactMatch`, `broadMatch`, `narrowMatch`, `relatedMatch`, `noMatch`) with the mention as subject: `broadMatch` means the candidate is broader than the mention.

Named-entity recognition is not part of BAGEL. Mentions come from gold annotations or an upstream recognizer. See [docs/architecture.md](docs/architecture.md) for the design.

This is version 2, a rewrite in progress. The previous service is in the git history.

## Workflows

| Use case | How | Status |
|---|---|---|
| Try a mention, debug a prompt or model | `bagel link` | works |
| Link the mentions an NER found (LitCoin, ORION) | the Python functions `extract` and `link` | works; ORION and LitCoin not yet switched over |
| Evaluate a model or prompt on a benchmark | `bagel abbreviations`, `bagel run`, `bagel eval` | works |
| Tune a prompt | copy a task spec, rerun, compare with `bagel eval` | works |
| Link over HTTP, or from a web page | a service around `link` | not built yet |

The same choices apply to every workflow:
- a **backend**: which model to call;
- a **task spec**: the prompt, and how to read its answers;
- a **selection policy**: which relations count as a link.

A benchmark run and a production call with the same three choices give the same answers.

## Setup

```bash
uv sync
```

## Backends

A backend config is a small YAML file saying which model to call. It works the same way in every workflow.

A local model, or any OpenAI-compatible server (Ollama, vLLM):

```yaml
# ollama.yaml
kind: openai
base_url: http://localhost:11434/v1
model: qwen3:14b
extra_body: {think: false}
```

A Claude subscription, through Claude Code (log in with `claude` first). There's no API key; usage counts against the subscription:

```yaml
kind: claude_cli
model: claude-sonnet-4-6
```

Azure OpenAI, where `model` is your deployment name:

```yaml
kind: azure_openai
endpoint: https://<resource>.openai.azure.com/openai/v1/
model: <deployment name>
api_key_keychain: azure-openai
```

The OpenAI API itself is `kind: openai` with no `base_url`.

Notes:
- **Azure versions:** the v1 endpoint above needs no `api_version`. The older form, `endpoint: https://<resource>.openai.azure.com`, also works, together with `api_version`.
- **Model names:** pin full names rather than aliases like `sonnet`, because an alias can point to a different model after an update. Each answer records the model the backend reports, and `bagel eval` lists them.
- **Reasoning models** that reject `temperature` (o-series, gpt-5): set `temperature: null`.
- **Retries:** failed requests aren't retried inside a call. A failure is recorded as `api_error`, and batch commands can retry it later. Azure's per-minute token limits are easy to hit with long documents; lower the concurrency if you see 429s.

### API keys

Configs name where a key lives, never the key itself, so configs can be shared and committed. Keep keys out of chat with coding assistants, out of files in any repository, and out of your shell environment.

On a Mac, store the key in the Keychain from your own terminal. `-w` with no value makes `security` prompt for it, so it isn't echoed or saved in shell history:

```bash
security add-generic-password -U -a "$USER" -s azure-openai -w
```

Then set `api_key_keychain: azure-openai` in the backend config. BAGEL reads the key only when it creates the backend, inside its own process. The first read may show a Keychain dialog; allow it. To replace the key, run the same command again; to remove it:

```bash
security delete-generic-password -s azure-openai
```

On servers and in containers, use an environment variable instead (`api_key_env`, default `AZURE_OPENAI_API_KEY` or `OPENAI_API_KEY`), supplied by the platform, e.g. a Kubernetes secret. If a key is ever pasted somewhere it shouldn't be, rotate it.

## Link one mention

```bash
uv run bagel link --backend ollama.yaml --mention Tnf --text "Hepatic Tnf mRNA rose in Sprague-Dawley rats."
```

The output has:
- the query as searched, after abbreviation expansion;
- the candidates, and each candidate's relation;
- the selection: `linked` with a CURIE, `abstained`, or `failed` with a reason.

Abbreviations are either extracted from the text with the same backend (`--extract-abbreviations`, one extra model call), or given directly:

```bash
uv run bagel link --backend ollama.yaml --mention "CF" --text-file abstract.txt --extract-abbreviations
uv run bagel link --backend ollama.yaml --mention "CF" --text-file abstract.txt --abbreviations '{"CF": "cystic fibrosis"}'
```

Other options:
- `--task`: a task spec name or path; the default is `relations_v2`.
- `--pipeline`: a YAML file with retrieval settings and the selection policy.
- `--type-hint`: a Biolink type for the mention.
- `--cache`: a retrieval cache file.
- `--raw`: include the raw model response.

## Link from Python, after an NER

A pipeline that already has mentions calls BAGEL per document. It extracts abbreviations once per document, then links each mention:

```python
import httpx
from bagel.abbreviations import clean_abbreviation_map, extract, load_abbreviation_task
from bagel.adjudicate import load_task
from bagel.backends import make_backend
from bagel.cache import Cache
from bagel.pipeline import PipelineConfig, link
from bagel.types import Document, Mention

backend = make_backend({"kind": "azure_openai", "endpoint": "https://<resource>.openai.azure.com/openai/v1/",
                        "model": "gpt-4o-mini", "api_key_keychain": "azure-openai"})
task, abbreviation_task = load_task("relations_v2"), load_abbreviation_task()
config = PipelineConfig(policy={"accept": ["exactMatch"]})  # exact matches only
cache = Cache("retrieval-cache.sqlite")

document = Document(id="PMID:25763772", text=abstract)
extraction = extract(abbreviation_task, backend, document)
if extraction.status != "ok":
    raise RuntimeError(extraction.error)
abbreviations = clean_abbreviation_map(extraction.abbreviation_map)

with httpx.Client(timeout=60) as client:
    for text, start, end in recognized_mentions:  # from the NER
        result = link(document, Mention(text=text, start=start, end=end), abbreviations, task, backend, config,
                      client, cache)
        if result.selection.status == "linked":
            print(text, result.selection.curie, result.selection.relation)
```

What to know when integrating:
- **Results:** `result.status` says how each stage went (`ok`, `api_error`, `parse_error`, `retrieval_error`, ...). `result.selection` is the answer: `linked`, `abstained` or `failed`. A failure is never turned into "no match", so decide what your pipeline does with each.
- **The selection policy is where use cases differ.** For example:
  - a knowledge graph that needs precise links: `accept: [exactMatch]`;
  - the LitCoin pipeline: `[exactMatch, broadMatch]`;
  - ORION's old behavior: all four relations, best retrieval rank first.
- **Concurrency:** `link` is thread-safe given a shared backend, client and cache; `bagel run` uses a thread pool.
- **Type hints:** `Mention.type_hint` (a Biolink type from the NER) is only used to filter retrieval when `PipelineConfig(use_type_hint=True)`.

ORION and the ncats/litcoin pipeline each still have their own linking code. Switching them to this package is planned.

## Benchmarks and evaluation

Benchmarks live in the separate EL_benchmarks repository. Each one has:
- a `truth.jsonl` of items, each a mention with its context and judged identifiers;
- a `documents.jsonl` with the full text of each source document.

The judged identifiers are the benchmark's own, e.g. UMLS or MONDO. Whether BAGEL's answer matches one is decided when evaluating. The benchmarks:
- `medmentions_assessed`: MedMentions mentions with human-judged links. It is *open world*: a link nobody judged is **unjudged**, not wrong.
- `craft`: CRAFT concept annotations. It is *closed world*: any link other than the annotated concept is wrong.

A benchmark evaluation has three steps.

**1. Extract abbreviations**, once per benchmark, backend and abbreviation prompt:

```bash
uv run bagel abbreviations --documents EL_benchmarks/craft/documents.jsonl --backend azure-4omini.yaml \
    --out abbreviations-craft-4omini.jsonl
```

This writes one line per document:
- the abbreviations found in the text;
- `ungrounded`: pairs the model returned that aren't in the text, kept for inspection and not used;
- the model and usage.

Failed documents aren't written, so running the same command again retries them. Output files from a different task or backend are refused.

**2. Run.** A run config names the benchmark, abbreviations, task, backend, selection policy and retrieval cache:

```yaml
dataset: EL_benchmarks/craft/truth.jsonl
abbreviations: abbreviations-craft-4omini.jsonl
task: relations_v2
backend: {kind: azure_openai, endpoint: https://<resource>.openai.azure.com/openai/v1/, model: gpt-4o-mini,
          api_key_keychain: azure-openai}
cache: retrieval-cache.sqlite
pipeline:
  policy: {accept: [exactMatch, broadMatch], on_multiple: best_rank}
concurrency: 4
limit: 100  # optional: the first 100 items
```

```bash
uv run bagel run craft-4omini.yaml --out craft-4omini.sqlite
uv run bagel run craft-4omini.yaml --out craft-4omini.sqlite --retry-failed
```

A run is one SQLite file. Running again with the same `--out` resumes, and `--retry-failed` reruns the mentions that ended in `api_error` or `retrieval_error`. A run file belongs to one configuration, including the content of the abbreviation file; changing any of it needs a new `--out`.

**3. Evaluate** one or more runs:

```bash
uv run bagel eval craft-4omini.sqlite craft-claude.sqlite --dataset EL_benchmarks/craft/truth.jsonl --cache matcher-cache.sqlite
```

- **Matching:** by default (`--matcher babel`), a linked identifier matches a judged one if it is the same identifier or in the same NodeNorm clique. `--no-conflate` turns off gene/protein and drug/chemical conflation. `--matcher identity` requires the same identifier. The report records the Babel release, since clique membership changes between releases.
- **Fair comparisons:** runs are compared on the mentions they all have results for.
- **Other options:**
  - `--proposers` adds the precision of the systems whose proposals were judged, such as MedMentions' own links;
  - `--json` prints the full report.

The report has, for each run:
- **Selection:**
  - linked, correct, incorrect, unjudged, conflicting, abstained, failed;
  - coverage;
  - precision on judged links;
  - accuracy over all mentions.
- **Gold:** how many mentions have a correct identifier, and how many of those the matcher can resolve.
- **Retrieval recall:** a correct identifier was among the candidates.
- **Adjudication, given retrieval:**
  - how often the correct candidate was labeled `exactMatch`;
  - how often an incorrect one was;
  - how often several candidates were.
- **Abstention:** how often BAGEL abstained when no correct candidate was retrieved.
- **Operations:** status counts, models, latency and token usage.

## Prompt tuning

Task specs in `bagel/tasks/` hold the prompts. Each linking task spec has:
- the message templates, with `{text}`, `{mention}` and `{candidates}`;
- the output JSON schema;
- how the prompt's relation labels map to SKOS;
- how candidates are shown and identified.

The ones included:
- `relations_bwc`: the BagelWithCheese prompt, used for run_5 and by the ncats/litcoin pipeline.
- `relations_v2`: the draft successor prompt.
- `abbreviations_v1`: abbreviation extraction. It asks for definitions exactly as written, so they can be checked against the text; nested abbreviations are expanded afterwards.

To try a change:
1. Copy a task spec to a new file and edit it.
2. Point a run config's `task:` at the file's path, and run with a new `--out`.
3. Compare the runs with `bagel eval`.

The retrieval cache means only the model calls are repeated. Run files record the task spec's hash, so each result can be traced to the exact prompt that produced it.

Two cautions:
- **Tune on one benchmark and check on the other,** or tune on part of one, so the prompt isn't fit to the test.
- **Hosted models aren't deterministic,** even at temperature 0 with a fixed seed. Two identical gpt-4o-mini runs on 100 MedMentions mentions differed on 7 of them. Compare prompts on several hundred mentions, or repeat the runs, before trusting a difference of a few points.

## Tests

```bash
uv run pytest -m "not integration"   # offline
uv run pytest                        # also calls NameRes, SapBERT and NodeNorm
```
