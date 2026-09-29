from bagel.evaluate import evaluate_baseline, evaluate_run
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


ITEMS = [
    {"id": "m1", "gold": {"A:1": "correct"}, "medmentions": "A:1", "baselines": {"old": "A:1"}},
    {"id": "m2", "gold": {"A:2": "incorrect", "A:9": "correct"}, "medmentions": "A:9", "baselines": {"old": None}},
    {"id": "m3", "gold": {"A:3": "unsure"}, "medmentions": "A:3", "baselines": {"old": "A:2"}},
]


def test_run_metrics():
    results = {"m1": _result("A:1", exact=("A:1",)),
               "m2": _result("A:2", exact=("A:2",)),  # links a known-incorrect CURIE; the correct one wasn't retrieved
               "m3": _result("A:1", exact=("A:1",))}  # links a CURIE nobody assessed
    report = evaluate_run(results, ITEMS)
    s = report["selection"]
    assert (s["correct"], s["incorrect"], s["unassessed"]) == (1, 1, 1)
    assert s["precision_assessed"] == 0.5
    assert report["retrieval"] == {"mentions_with_correct_gold": 2, "correct_retrieved": 1, "recall": 0.5}
    assert report["adjudication_given_retrieved"]["exact_on_correct"] == 1.0
    assert report["abstention_when_correct_not_retrieved"] == {"n": 1, "abstained": 0.0}


def test_baseline_metrics():
    s = evaluate_baseline("old", ITEMS)["selection"]
    assert (s["correct"], s["abstained"], s["unassessed"]) == (1, 1, 1)
