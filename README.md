# BAGEL

Babel Assisted Generative Entity Linker. Given a document and mentions in it, BAGEL:

1. retrieves candidate identifiers from NameRes and SapBERT, normalized with NodeNorm;
2. asks a model how each candidate relates to the mention;
3. picks the link.

Relations are SKOS (`exactMatch`, `broadMatch`, `narrowMatch`, `relatedMatch`, `noMatch`) with the mention as subject: `broadMatch` means the candidate is broader than the mention.

Named-entity recognition is not part of BAGEL. Mentions come from gold annotations or an upstream recognizer. See [docs/architecture.md](docs/architecture.md) for the design.

This is version 2, a rewrite in progress. The previous service is in the git history.

## Setup

```bash
uv sync
```

## Link one mention

A backend config says which model to call. For example, a local Ollama:

```yaml
# ollama.yaml
kind: openai
base_url: http://localhost:11434/v1
model: qwen3:14b
extra_body: {think: false}
```

Or a Claude subscription through Claude Code (log in first):

```yaml
kind: claude_cli
model: claude-sonnet-4-6
```

Or Azure OpenAI, where `model` is your deployment name:

```yaml
kind: azure_openai
endpoint: https://<resource>.openai.azure.com
model: <deployment name>
api_key_keychain: bagel-azure-openai
```

Pin full model names rather than aliases like `sonnet`: an alias can point to a different model after an update. Each decision also records the model the backend reports, and `bagel eval` lists them. For reasoning models that reject `temperature` (o-series, gpt-5), set `temperature: null`.

```bash
uv run bagel link --backend ollama.yaml --mention Tnf --text "Hepatic Tnf mRNA rose in Sprague-Dawley rats."
```

The output has the query as searched (after abbreviation expansion), the candidates, each candidate's relation, and the selection (`linked` with a CURIE, `abstained`, or `failed` with a reason).

### API keys

Configs name where a key lives, never the key itself, so configs can be shared and committed. Keep keys out of chat with coding assistants, out of files in any repository, and out of your shell environment.

On a Mac, store the key in the Keychain from your own terminal. `-w` with no value makes `security` prompt for it, so it isn't echoed or saved in shell history:

```bash
security add-generic-password -U -a "$USER" -s bagel-azure-openai -w
```

Then set `api_key_keychain: bagel-azure-openai` in the backend config. BAGEL reads the key only when it creates the backend, inside its own process. The first read may show a Keychain dialog; allow it. To replace the key, run the same command again; to remove it:

```bash
security delete-generic-password -s bagel-azure-openai
```

On servers and in containers, use an environment variable instead (`api_key_env`, default `AZURE_OPENAI_API_KEY` or `OPENAI_API_KEY`), supplied by the platform, e.g. a Kubernetes secret. If a key is ever pasted somewhere it shouldn't be, rotate it in the Azure portal.

## Task specs

`bagel/tasks/*.yaml` hold the prompts. Each task spec has:
- the message templates, with `{text}`, `{mention}` and `{candidates}`;
- the output JSON schema;
- how the prompt's relation labels map to SKOS;
- how candidates are shown and identified.

Two are included:

- `relations_bwc`: the BagelWithCheese prompt, used for run_5 and by the ncats/litcoin pipeline.
- `relations_v2`: the draft successor prompt.

## Batch runs and evaluation

Build the human-assessed MedMentions set from a dump of the bagel-eval database. The dump holds assessor names, so keep it and the dataset outside this repository.

```bash
uv run bagel dataset --dump dump.sql --abbreviations abbreviation_llm_results.jsonl --out assessed.jsonl
```

A run config names the dataset, task, backend, selection policy and retrieval cache:

```yaml
dataset: /data/bagel/assessed.jsonl
task: relations_v2
backend: {kind: openai, base_url: http://localhost:11434/v1, model: qwen3:14b, extra_body: {think: false}}
cache: /data/bagel/retrieval-cache.sqlite
pipeline:
  policy: {accept: [exactMatch, broadMatch], on_multiple: best_rank}
concurrency: 4
```

Run it, then evaluate. Rerunning with the same `--out` resumes.

```bash
uv run bagel run run.yaml --out run.sqlite
uv run bagel eval run.sqlite --dataset assessed.jsonl --baseline gpt-oss
```

`bagel eval` reports the following. A linked CURIE nobody assessed counts as **unassessed**, never as right or wrong.

- retrieval recall;
- how the model labeled the correct candidate when it was retrieved;
- abstention when it wasn't retrieved;
- selection accuracy and coverage;
- agreement with MedMentions.

## Tests

```bash
uv run pytest -m "not integration"   # offline
uv run pytest                        # also calls NameRes, SapBERT and NodeNorm
```
