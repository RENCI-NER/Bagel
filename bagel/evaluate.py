"""
Metrics for runs against a benchmark's judgments. Whether a produced identifier matches a judged one is decided by a
matcher (bagel.matching), so benchmarks can keep their source identifiers.
"""
import statistics
from collections import Counter

from bagel.abbreviations import _word_pattern, clean_abbreviation_map
from bagel.matching import judged_verdict
from bagel.types import MentionResult, Relation


def _candidate_ids(result: MentionResult) -> set[str]:
    return {c for cand in result.candidates for c in [cand.curie, *cand.source_curies]}


def selection_metrics(outcomes: list[str]) -> dict:
    counts = Counter(outcomes)
    n = len(outcomes)
    linked = sum(counts[k] for k in ("correct", "incorrect", "unjudged", "conflicting"))
    judged = counts["correct"] + counts["incorrect"]
    return {
        "n": n,
        "linked": linked,
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "unjudged": counts["unjudged"],  # only in open-world benchmarks
        "conflicting": counts["conflicting"],
        "abstained": counts["abstained"],
        "failed": counts["failed"],
        "coverage": linked / n if n else None,
        "precision_judged": counts["correct"] / judged if judged else None,
        "accuracy": counts["correct"] / n if n else None,  # correct links over all mentions
    }


def prefetch(matcher, results: dict[str, MentionResult], items: list[dict]) -> None:
    """Looks up every identifier the evaluation will compare, in bulk."""
    ids = {j["curie"] for i in items for j in i["judgments"]}
    for r in results.values():
        ids |= _candidate_ids(r)
        if r.selection.curie:
            ids.add(r.selection.curie)
    matcher.prefetch(ids)


def evaluate_run(results: dict[str, MentionResult], items: list[dict], matcher) -> dict:
    items = [i for i in items if i["id"] in results]
    prefetch(matcher, results, items)
    outcomes = []
    confusion = Counter()
    has_gold = reachable = retrieved = exact_correct = exact_incorrect = multi_exact = 0
    missed = missed_abstained = 0
    for item in items:
        result = results[item["id"]]
        judgments, closed = item["judgments"], item["closed_world"]
        sel = result.selection
        outcomes.append(judged_verdict(sel.curie, judgments, matcher, closed) if sel.status == "linked"
                        else sel.status)
        correct = [j["curie"] for j in judgments if j["verdict"] == "correct"]
        gold_offered = any(matcher.same(o, c) for o in _candidate_ids(result) for c in correct)
        if sel.status == "linked":
            confusion[{"correct": "tp", "incorrect": "fp"}.get(outcomes[-1], outcomes[-1])] += 1
        else:
            confusion["fn" if gold_offered else "tn"] += 1
        if not correct:
            continue
        has_gold += 1
        reachable += any(matcher.known(c) for c in correct)
        if gold_offered:
            retrieved += 1
            if result.decision and result.decision.status == "ok":
                exact = [j.curie for j in result.decision.with_relation(Relation.EXACT)]
                verdicts = [judged_verdict(e, judgments, matcher, closed) for e in exact]
                exact_correct += "correct" in verdicts
                exact_incorrect += "incorrect" in verdicts
                multi_exact += len(exact) > 1
        else:
            missed += 1
            missed_abstained += sel.status == "abstained"
    seconds = [r.decision.seconds for r in results.values() if r.decision and r.decision.seconds is not None]
    usage = Counter()
    for r in results.values():
        if r.decision:
            usage.update({k: v for k, v in r.decision.usage.items() if isinstance(v, (int, float)) and v is not None})
    models = Counter(str(r.decision.usage.get("model")) for r in results.values() if r.decision)
    return {
        "matcher": matcher.describe(),
        "selection": selection_metrics(outcomes),
        # tp/fp: linked to a correct/incorrect concept; fn: abstained or failed although a correct concept was among
        # the candidates; tn: abstained or failed with no correct concept among them. Open-world benchmarks also
        # have links nobody judged (unjudged), which are none of the four.
        "confusion": {"tp": confusion["tp"], "fp": confusion["fp"], "fn": confusion["fn"], "tn": confusion["tn"],
                      "unjudged": confusion["unjudged"], "conflicting": confusion["conflicting"],
                      "precision": confusion["tp"] / (confusion["tp"] + confusion["fp"])
                      if confusion["tp"] + confusion["fp"] else None,
                      # of mentions with a correct concept, end to end (retrieval and linking)
                      "end_to_end_recall": confusion["tp"] / has_gold if has_gold else None},
        "status": dict(Counter(r.status for r in results.values())),
        "gold": {"mentions_with_correct_gold": has_gold,
                 # correct gold the matcher can't resolve (e.g. identifiers missing from this Babel release)
                 "mentions_with_reachable_gold": reachable},
        "retrieval": {"correct_retrieved": retrieved, "recall": retrieved / has_gold if has_gold else None},
        "adjudication_given_retrieved": {
            "n": retrieved,
            "exact_on_correct": exact_correct / retrieved if retrieved else None,
            "exact_on_incorrect": exact_incorrect / retrieved if retrieved else None,
            "multiple_exact": multi_exact / retrieved if retrieved else None,
        },
        "abstention_when_correct_not_retrieved": {"n": missed,
                                                  "abstained": missed_abstained / missed if missed else None},
        "latency_seconds": {"mean": statistics.mean(seconds), "p50": statistics.median(seconds),
                            "p95": statistics.quantiles(seconds, n=20)[-1] if len(seconds) >= 20 else max(seconds)}
        if seconds else None,
        "usage": dict(usage),
        "models": dict(models),
    }


def evaluate_proposers(items: list[dict]) -> dict:
    """Precision of each system whose proposals were judged (judgments' `proposed_by`), e.g. MedMentions, gpt-oss."""
    counts: dict[str, Counter] = {}
    for item in items:
        for j in item["judgments"]:
            for system in j.get("proposed_by", []):
                counts.setdefault(system, Counter())[j["verdict"]] += 1
    return {system: {"judged": sum(c.values()), "precision": c["correct"] / sum(c.values())}
            for system, c in sorted(counts.items())}


def _pair_matches(pair: dict, inner: str, before: str) -> bool:
    """A pair accounts for a parenthetical: "long form (SF)" or "SF (long form)", with the other part just before."""
    abbreviation, definition = pair["abbreviation"].strip(), " ".join(pair["definition"].split()).casefold()
    near = " ".join(before.split()).casefold()
    word = _word_pattern([abbreviation])
    return bool(word.search(inner) and definition in near) or \
        bool(definition and definition in " ".join(inner.split()).casefold() and word.search(before[-60:]))


def evaluate_abbreviations(truth: list[dict], texts: dict[str, str], records: list[dict]) -> dict:
    """
    Scores an abbreviation file (from `bagel abbreviations`) against labeled parentheticals (EL_benchmarks
    abbreviations/truth.jsonl: same / instance / other). An item counts as predicted "same" when a pair in the
    cleaned map accounts for it, so extract and classify tasks are scored on what linking would actually use.
    A pair accounts for at most one item, the first in the text it matches, since an abbreviation is defined once
    ("odds ratio (OR)" defines OR; a later "(OR = 3.6)" doesn't). Pairs that account for no labeled item are
    counted as unmatched (defined without parentheses, or spurious).
    """
    by_document = {r["document"]: r for r in records}
    missing = sorted({t["document"] for t in truth} - set(by_document))
    if missing:
        raise ValueError(f"no extraction for {len(missing)} labeled documents, e.g. {missing[:3]}")
    counts = Counter()
    errors = {"instance_as_same": [], "other_as_same": [], "same_missed": []}
    used = set()
    for item in sorted(truth, key=lambda t: (t["document"], t["start"])):
        text = texts[item["document"]]
        before = text[max(0, item["start"] - 150):item["start"]].rstrip(" (")
        pairs = [{"abbreviation": a, "definition": d}
                 for a, d in clean_abbreviation_map(by_document[item["document"]]["abbreviation_map"]).items()
                 if (item["document"], a) not in used]
        hits = [p for p in pairs if _pair_matches(p, item["inner"], before)]
        used |= {(item["document"], p["abbreviation"]) for p in hits}
        predicted = "same" if hits else "not"
        counts[(item["label"], predicted)] += 1
        if hits and item["label"] != "same":
            errors[f"{item['label']}_as_same"].append((item["inner"], hits[0]["abbreviation"], hits[0]["definition"]))
        elif not hits and item["label"] == "same":
            errors["same_missed"].append(item["inner"])
    all_pairs = {(d, a) for d in {t["document"] for t in truth}
                 for a in clean_abbreviation_map(by_document[d]["abbreviation_map"])}
    tp, fn = counts[("same", "same")], counts[("same", "not")]
    fp = counts[("instance", "same")] + counts[("other", "same")]
    return {
        "same": {"labeled": tp + fn, "found": tp, "missed": fn, "false": fp,
                 "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None},
        "instance_as_same": counts[("instance", "same")],
        "instances": counts[("instance", "same")] + counts[("instance", "not")],
        "other_as_same": counts[("other", "same")],
        "unmatched_pairs": len(all_pairs - used),
        "errors": errors,
    }
