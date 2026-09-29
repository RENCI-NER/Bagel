"""Metrics for runs (and stored baselines) against the human-assessed gold."""
import statistics
from collections import Counter

from bagel.types import MentionResult, Relation


def _outcome(curie: str | None, status: str, gold: dict[str, str]) -> str:
    """correct | incorrect | unassessed (linked to a CURIE without a correct/incorrect verdict) | abstained | failed"""
    if status != "linked":
        return status
    verdict = gold.get(curie)
    return verdict if verdict in ("correct", "incorrect") else "unassessed"


def selection_metrics(outcomes: list[str], medmentions_agree: list[bool]) -> dict:
    counts = Counter(outcomes)
    n = len(outcomes)
    linked = counts["correct"] + counts["incorrect"] + counts["unassessed"]
    judged = counts["correct"] + counts["incorrect"]
    return {
        "n": n,
        "linked": linked,
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "unassessed": counts["unassessed"],
        "abstained": counts["abstained"],
        "failed": counts["failed"],
        "coverage": linked / n if n else None,
        "precision_assessed": counts["correct"] / judged if judged else None,  # among links with a verdict
        "agrees_with_medmentions": sum(medmentions_agree) / n if n else None,
    }


def evaluate_run(results: dict[str, MentionResult], items: list[dict]) -> dict:
    items = [i for i in items if i["id"] in results]
    outcomes, mm_agree = [], []
    has_gold = retrieved = exact_correct = exact_incorrect = multi_exact = 0
    missed_abstained = missed = 0
    for item in items:
        result, gold = results[item["id"]], item["gold"]
        outcome = _outcome(result.selection.curie, result.selection.status, gold)
        outcomes.append(outcome)
        mm_agree.append(result.selection.status == "linked" and result.selection.curie == item["medmentions"])
        correct = {c for c, v in gold.items() if v == "correct"}
        incorrect = {c for c, v in gold.items() if v == "incorrect"}
        if not correct:
            continue
        has_gold += 1
        offered = {c for cand in result.candidates for c in [cand.curie, *cand.source_curies]}
        if correct & offered:
            retrieved += 1
            if result.decision and result.decision.status == "ok":
                exact = {j.curie for j in result.decision.with_relation(Relation.EXACT)}
                exact_correct += bool(exact & correct)
                exact_incorrect += bool(exact & incorrect)
                multi_exact += len(exact) > 1
        else:
            missed += 1
            missed_abstained += result.selection.status == "abstained"
    seconds = [r.decision.seconds for r in results.values() if r.decision and r.decision.seconds is not None]
    usage = Counter()
    for r in results.values():
        if r.decision:
            usage.update({k: v for k, v in r.decision.usage.items() if isinstance(v, (int, float)) and v is not None})
    return {
        "selection": selection_metrics(outcomes, mm_agree),
        "status": dict(Counter(r.status for r in results.values())),
        "retrieval": {"mentions_with_correct_gold": has_gold, "correct_retrieved": retrieved,
                      "recall": retrieved / has_gold if has_gold else None},
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
    }


def evaluate_baseline(model: str, items: list[dict]) -> dict:
    """Selection metrics for a stored baseline result (e.g. run_5's gpt-oss); None means it gave no single exact."""
    items = [i for i in items if model in i["baselines"]]
    outcomes = [_outcome(i["baselines"][model], "linked" if i["baselines"][model] else "abstained", i["gold"])
                for i in items]
    mm = [i["baselines"][model] is not None and i["baselines"][model] == i["medmentions"] for i in items]
    return {"selection": selection_metrics(outcomes, mm)}
