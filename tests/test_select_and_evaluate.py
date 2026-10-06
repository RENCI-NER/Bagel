from bagel.evaluate import evaluate_proposers, evaluate_run
from bagel.matching import IdentityMatcher
from bagel.select import SelectionPolicy, select
from bagel.types import Candidate, Decision, Judgment, Mention, MentionResult, Relation

CANDIDATES = [Candidate(curie=c, label=c) for c in ("A:1", "A:2", "A:3")]  # retrieval order


def decision(**relations):
    return Decision(status="ok", judgments=[Judgment(curie=c.curie, relation=relations.get(c.curie.replace(":", ""),
                                                                                          Relation.NONE))
                                            for c in CANDIDATES])


def test_first_accepted_relation_wins():
    d = decision(A2=Relation.BROAD, A3=Relation.EXACT)
    assert select(d, CANDIDATES, SelectionPolicy()).curie == "A:3"
    assert select(d, CANDIDATES, SelectionPolicy(accept=[Relation.BROAD, Relation.EXACT])).curie == "A:2"


def test_multiple_exact_best_rank_or_abstain():
    d = decision(A2=Relation.EXACT, A3=Relation.EXACT)
    assert select(d, CANDIDATES, SelectionPolicy()).curie == "A:2"
    assert select(d, CANDIDATES, SelectionPolicy(on_multiple="abstain")).status == "abstained"


def test_no_accepted_relation_abstains_and_failures_are_failures():
    assert select(decision(A1=Relation.RELATED), CANDIDATES, SelectionPolicy()).status == "abstained"
    assert select(Decision(status="parse_error"), CANDIDATES, SelectionPolicy()).status == "failed"


def _result(selection_curie, exact=()):
    d = Decision(status="ok", judgments=[Judgment(curie=c.curie, relation=Relation.EXACT if c.curie in exact
                                                  else Relation.NONE) for c in CANDIDATES])
    return MentionResult(mention=Mention(text="x"), query="x", status="ok", candidates=CANDIDATES, decision=d,
                         selection=select(d, CANDIDATES, SelectionPolicy()))


ITEMS = [  # benchmark items as bagel.benchmarks produces them
    {"id": "m1", "closed_world": False, "judgments": [
        {"curie": "A:1", "verdict": "correct", "proposed_by": ["medmentions", "old"]}]},
    {"id": "m2", "closed_world": False, "judgments": [
        {"curie": "A:2", "verdict": "incorrect", "proposed_by": ["old"]},
        {"curie": "A:9", "verdict": "correct", "proposed_by": ["medmentions"]}]},
    {"id": "m3", "closed_world": False, "judgments": [{"curie": "A:3", "verdict": "incorrect"}]},
]


def test_run_metrics():
    results = {"m1": _result("A:1", exact=("A:1",)),
               "m2": _result("A:2", exact=("A:2",)),  # links a known-incorrect CURIE; the correct one wasn't retrieved
               "m3": _result("A:1", exact=("A:1",))}  # links a CURIE nobody judged
    report = evaluate_run(results, ITEMS, IdentityMatcher())
    s = report["selection"]
    assert (s["correct"], s["incorrect"], s["unjudged"]) == (1, 1, 1)
    assert s["precision_judged"] == 0.5
    assert report["gold"] == {"mentions_with_correct_gold": 2, "mentions_with_reachable_gold": 2}
    assert report["retrieval"] == {"correct_retrieved": 1, "recall": 0.5}
    assert report["adjudication_given_retrieved"]["exact_on_correct"] == 1.0
    assert report["abstention_when_correct_not_retrieved"] == {"n": 1, "abstained": 0.0}


def test_closed_world_makes_unjudged_links_incorrect():
    closed = [dict(i, closed_world=True) for i in ITEMS]
    s = evaluate_run({"m3": _result("A:1", exact=("A:1",))}, closed, IdentityMatcher())["selection"]
    assert (s["incorrect"], s["unjudged"]) == (1, 0)


def test_baseline_metrics():
    # precision of the systems whose proposals were judged
    assert evaluate_proposers(ITEMS) == {"medmentions": {"judged": 2, "precision": 1.0},
                                         "old": {"judged": 2, "precision": 0.5}}


def test_best_confidence_breaks_ties_between_exact_matches():
    candidates = [Candidate(curie="A:1", label="a"), Candidate(curie="B:2", label="b")]
    decision = Decision(status="ok", judgments=[Judgment(curie="A:1", relation=Relation.EXACT, confidence=0.6),
                                                Judgment(curie="B:2", relation=Relation.EXACT, confidence=0.9)])
    assert select(decision, candidates, SelectionPolicy(on_multiple="best_confidence")).curie == "B:2"
    assert select(decision, candidates, SelectionPolicy(on_multiple="best_rank")).curie == "A:1"
    low = SelectionPolicy(accept=["exactMatch"], on_multiple="best_confidence", min_confidence=0.95)
    assert select(decision, candidates, low).status == "abstained"
