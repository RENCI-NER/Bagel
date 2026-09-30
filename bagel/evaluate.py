"""
Metrics for runs against a benchmark's judgments. Whether a produced identifier matches a judged one is decided by a
matcher (bagel.matching), so benchmarks can keep their source identifiers.
"""
import statistics
from collections import Counter

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
    has_gold = reachable = retrieved = exact_correct = exact_incorrect = multi_exact = 0
    missed = missed_abstained = 0
    for item in items:
        result = results[item["id"]]
        judgments, closed = item["judgments"], item["closed_world"]
        sel = result.selection
        outcomes.append(judged_verdict(sel.curie, judgments, matcher, closed) if sel.status == "linked"
                        else sel.status)
        correct = [j["curie"] for j in judgments if j["verdict"] == "correct"]
        if not correct:
            continue
        has_gold += 1
        reachable += any(matcher.known(c) for c in correct)
        offered = _candidate_ids(result)
        if any(matcher.same(o, c) for o in offered for c in correct):
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
