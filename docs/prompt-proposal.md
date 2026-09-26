# Proposal: updating the Bagel prompts

Status: draft for discussion. The proposed prompt is `bagel/ask_classes_v2` (source: [`prompts/src/ask_classes_v2.yaml`](../prompts/src/ask_classes_v2.yaml)). It's registered as a local prompt; no default or deployed config has been changed.

## What's wrong with the current prompts

These come from reading every version in [`prompts/`](../prompts/) and how the code uses them.

**1. The output contract contradicts itself.** `ask_classes` and `ask_classes_no_system` define the keys `synonym_type` / `color_code` / `vocabulary_class`, but their worked example uses `synonymType` / `color-code` / `vocabulary class`. Models copy examples, so responses came back in either form, and the old parser counted the hyphenated form as a hallucination.

**2. The newest prompt (`ask_classes_cot`) breaks downstream consumers.** It renames the relation types to `subclass` / `superclass` / `None`. ORION's `extract_best_match` only knows `exact` / `narrow` / `broad` / `related`, so every CoT result would be discarded as no match. The prompt also says its input is "a JSON object containing three keys", but it receives the same bulleted text as the other prompts. And it requires the output to have as many entries as there are candidates, which means about 20 JSON objects per call.

**3. `ask_classes_no_system_simplified` contains a leftover test answer and a wrong rule.** It includes `Prioritize exact matches (human BMI1 = "blue-violet")`. Color codes are assigned by position, so "blue-violet" is whichever candidate happens to be 16th in every request. It also says exact matching is case-insensitive ("Bmi1" = "BMI1"), but for genes, case often distinguishes the mouse gene from the human one.

**4. The candidate list is hard to parse.** Each candidate is written as `label [taxa] (class) (color) : definition`, but labels and taxa contain their own parentheses and colons (`Heart attack (myocardial infarction)`, `Pongo abelii (Sumatran orangutan)`, `Hypertension, CTCAE`). Before the fix in this branch, a candidate with no class also had no color code, so it could never be selected.

**5. Missing guidance for things the pipeline actually does:**
- **Species.** Only `ask_classes_cot` mentions it. NameRes/SapBERT return the same gene symbol for many species, and before this branch, SapBERT-only candidates had no taxon at all.
- **Conflation.** Candidates are normalized with gene/protein and drug/chemical conflation. The prompts never tell the model that a Gene candidate stands for the protein too, so "p53 protein" vs a Gene candidate looks like a class mismatch.
- **The right concept may not have been retrieved.** For "HD" (Huntington's disease), "MS" (multiple sclerosis) and "p53", NameRes and SapBERT return mostly insect genes and none of the correct concepts. The prompts say "it is also possible there are no matches" but give no guidance, and "at most one exact" invites the model to pick the closest string.
- **Which occurrence.** The model gets the mention string, not its position, so it can't tell repeated or ambiguous occurrences apart.
- **Abbreviations defined in the abstract** ("Huntington's disease (HD)") are not mentioned.

**6. Instructions that do nothing.** "Strictly ensure that your output is going to stay the same every time you are asked this input" has no effect; determinism comes from temperature and seed. "Your output must strictly follow the JSON format … without including anything other than the JSON" can now be *enforced* with structured output (below) instead of requested.

**7. The narrow/broad direction is easy to misread.** The labels are defined relative to the mention ("the query term is a more specific form of the list term"), but they're stored on the candidate. So `MONDO:diabetes mellitus → narrow` means the *candidate is broader*. That's workable but needs saying explicitly, especially since ORION prefers `narrow` over `broad` for exactly this reason: linking to a broader concept is safe, linking to a narrower one asserts more than the text says.

## What `ask_classes_v2` changes

| | Current prompts | `ask_classes_v2` |
|---|---|---|
| Output | JSON array; key names inconsistent | One object: `{"interpretation": "...", "matches": [{"id", "relation"}]}` |
| Relation labels | exact/narrow/broad/related, or subclass/superclass/None (CoT) | exact/narrow/broad/related, same as ORION; non-matches are left out |
| Direction | Relative to the "query term" | Stated both ways: "the mention is more specific, i.e. the candidate is broader" |
| Candidates | Bulleted text with nested parentheses | JSON, one object per line: `id`, `label`, `category`, `taxon`, `definition` (empty fields omitted) |
| Reasoning | None, or a long `<reasoning>` block over every candidate (CoT) | One-sentence `interpretation` written before the matches, as a short chain of thought |
| Species | Only in CoT | Human by default; genes need the right taxon to be exact; other species' genes are related |
| Conflation | Not mentioned | A gene and its protein, and a drug and its active ingredient, count as exact |
| Nothing fits | "It is also possible…" | `"matches": []`, and no forced exact |
| Near misses | Not addressed | Groupings ("X and related disorders"), look-alikes ("X-like syndrome"), subtypes and tests/records about X are not exact |
| Mention position | Not available | `[[mention]]` marks the occurrence when offsets are known (batch: `"mark_mentions": true`) |
| Worked example | Wrong key names | One example with the right keys and a species decision (mouse Tnf vs human TNF) |
| Message layout | Mixed | Static instructions in the system message, then abstract, mention, candidates, so repeated calls on the same abstract can reuse the cached prefix (OpenAI prompt caching, vLLM prefix caching) |
| Structured output | Not possible | A JSON schema is shipped with the prompt (`response_format` in the prompt record) for OpenAI `json_schema` (strict), vLLM guided decoding, or Ollama `format` |

To stay a drop-in replacement it takes the same inputs as before (`text`, `term`) plus `candidates_json`, which `SynonymListContext.prompt_inputs()` now provides for every prompt. Older prompts just ignore it.

### Using the schema

- **Batch API:** add `"response_format": <record's response_format>` to `model_args` in the run config.
- **Server:** pass `"llm_model_args": {"model_kwargs": {"response_format": ...}}` in the OpenAI config.
- **Ollama:** use `format: <schema>`, as `benchmark/compare_prompts.py` does with the `+schema` suffix.

## Results so far

A smoke test, not an evaluation: 13 hand-written cases ([`benchmark/prompt_cases.jsonl`](../benchmark/prompt_cases.jsonl)), candidates retrieved live from NameRes/SapBERT/NodeNorm (top 10 each), `qwen3:14b` via local Ollama (similar size to the gemma-3-12b ORION uses), temperature 0, thinking off. Three cases have no correct candidate, because retrieval missed Huntington disease, multiple sclerosis and human TP53; for those, "correct" means no candidate labeled exact. Rerun with:

```bash
python benchmark/compare_prompts.py --cases benchmark/prompt_cases.jsonl --model qwen3:14b --ollama-url http://localhost:11434 --prompts bagel/ask_classes bagel/ask_classes_no_system bagel/ask_classes_cot bagel/ask_classes_v2 bagel/ask_classes_v2+schema --out compare.jsonl
```

| Case | Gold | ask_classes | no_system | cot | v2 | v2 + schema |
|---|---|---|---|---|---|---|
| tnf-rat | NCBIGene:24835 | ✓ | ✓ | ✓ | ✓ | ✓ |
| hd-abbrev | *not retrieved* | ✓ | ✓ | ✓ | ✗ MONDO:0000167 | ✗ MONDO:0000167 |
| diabetic-adj | MONDO:0005015 | ✗ UMLS:C0241863 | ✗ UMLS:C0241863 | ✗ 3 exacts | ✓ | ✓ |
| heart-attack | MONDO:0005068 | ✓ | ✓ | ✗ 3 exacts | ✓ | ✓ |
| cold-disease | MONDO:0005709 | ✓ | ✓ | ✓ | ✓ | ✓ |
| aspirin | CHEBI:15365 | ✗ 3 exacts | ✗ 2 exacts | ✓ | ✓ | ✓ |
| t2d | MONDO:0005148 | ✗ 2 exacts | ✗ 2 exacts | ✓ | ✓ | ✓ |
| brca1-human | NCBIGene:672 | ✓ | ✓ | ✓ | ✓ | ✓ |
| fever | HP:0001945 | ✓ | ✓ | ✓ | ✓ | ✓ |
| ms-abbrev | *not retrieved* | ✓ | ✓ | ✓ | ✗ UMLS:C0236169 | ✗ UMLS:C0236169 |
| p53-human | *not retrieved* | ✗ UMLS:C0872074 | ✗ UMLS:C0872074 | ✗ UMLS:C0872074 | ✓ | ✓ |
| ace-enzyme | NCBIGene:1636 | ✓ | ✓ | ✓ | ✓ | ✓ |
| t2d-short | MONDO:0005148 | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Correct** | | **9/13** | **9/13** | **10/13** | **11/13** | **11/13** |
| Cases with >1 exact | | 2 | 2 | 2 | 0 | 0 |
| Mean seconds/call | | 6 | 6 | 77 | 6 | 4 |

Notes:
- **v2 fixes the multiple-exact problem** (0 cases vs 2 for every other prompt), maps "diabetic" to the disease rather than the UMLS "Diabetic" finding, and declines to link p53 to a "p53 gene/protein" information-content entity. Its answers were the same with and without the schema, and the schema makes calls faster because the output is shorter.
- **v2 is worse at abstaining when retrieval misses.** For HD and MS it labels the nearest grouping ("Huntington disease and related disorders") or look-alike ("MS-like syndrome") as exact, even though its own `interpretation` names the right disease. Rule 6 (near-misses aren't exact) didn't change that. Adding a worked "nothing matches" example made things worse (10/13, multiple exacts came back), so I left it out. This is the main open issue, and it needs a real evaluation set to tune against.
- **CoT is about 13× slower** because of its per-candidate reasoning, and it still produces multiple exacts.
- **Caveats.** The prompt's worked example (mouse Tnf) is close to the tnf-rat case, and 13 cases can't separate prompts that differ by one or two answers. Treat this as a check that v2 works end to end with a small open model, not as evidence it's better.

## Suggested next steps

1. **Evaluate on MedMentions**, not 13 hand-picked cases: run `ask_classes_no_system` (current benchmark prompt), `ask_classes` (ORION) and `ask_classes_v2` with and without the schema, all with the same candidates, on a fixed sample of about 2k mentions. Report accuracy separately for mentions whose gold concept was retrieved and those where it wasn't, since the prompts differ most on the second group.
2. **Replace color names with neutral IDs** (`c01`, `c02`, …) as a separate change. Many color names are biomedical words (copper, gold, silver, cobalt, lime, saffron, tomato, alizarin, amaranth), which matters when the mention is a chemical or a food. The mapping code doesn't care what the IDs are.
3. **Restrict IDs per request:** generate the schema's `id` enum from the candidate list, so strict structured output makes a hallucinated ID impossible.
4. **Measure position bias:** candidates are now in a stable order, so it's easy to shuffle them and see whether answers change.
5. **Deduplicate (abstract, mention) pairs** in the batch script. Repeated mentions get identical prompts unless they're marked.
6. After a decision, **publish `ask_classes_v2` to the Hub** (or drop the Hub entirely) and switch ORION's `call_bagel_service` to a pinned version, e.g. `bagel/ask_classes_v2:<hash>`.
