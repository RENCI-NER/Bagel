"""Stage 4 (select): choose the link, or abstain, from the adjudicated relations."""
from typing import Literal

from pydantic import BaseModel, Field

from bagel.types import Candidate, Decision, Relation, Selection


class SelectionPolicy(BaseModel):
    """
    accept: relations that may be linked, in order of preference; the first relation with any candidate wins.
    on_multiple: when several candidates share the winning relation, take the best retrieval rank, the highest
        confidence (backends that report one), or abstain.
    min_confidence: abstain if the chosen judgment's confidence is below this (only for backends that report one).

    Examples: exact only = accept [exactMatch]; ORION's extract_best_match = accept [exactMatch, broadMatch,
    narrowMatch, relatedMatch] with best_rank; the LitCoin pipeline = accept [exactMatch, broadMatch].
    """
    accept: list[Relation] = Field(default_factory=lambda: [Relation.EXACT, Relation.BROAD])
    on_multiple: Literal["best_rank", "best_confidence", "abstain"] = "best_rank"
    min_confidence: float | None = None


def select(decision: Decision, candidates: list[Candidate], policy: SelectionPolicy) -> Selection:
    if decision.status != "ok":
        return Selection(status="failed", reason=decision.status)
    rank = {c.curie: position for position, c in enumerate(candidates)}  # candidates are in retrieval order
    for relation in policy.accept:
        matches = sorted(decision.with_relation(relation), key=lambda j: rank[j.curie])
        if not matches:
            continue
        if len(matches) > 1 and policy.on_multiple == "abstain":
            return Selection(status="abstained", reason=f"multiple {relation.value}")
        if len(matches) > 1 and policy.on_multiple == "best_confidence":
            if any(m.confidence is None for m in matches):
                raise ValueError("on_multiple is best_confidence but the backend reports no confidence")
            matches = sorted(matches, key=lambda j: -j.confidence)
        chosen = matches[0]
        if policy.min_confidence is not None:
            if chosen.confidence is None:
                raise ValueError("min_confidence is set but the backend reports no confidence")
            if chosen.confidence < policy.min_confidence:
                return Selection(status="abstained", reason="low confidence", curie=chosen.curie,
                                 relation=relation, confidence=chosen.confidence)
        return Selection(status="linked", curie=chosen.curie, relation=relation, confidence=chosen.confidence)
    return Selection(status="abstained", reason="no accepted relation")
