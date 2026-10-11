import pytest

from bagel.matching import BabelMatcher, IdentityMatcher, judged_verdict

JUDGMENTS = [{"curie": "MONDO:0004979", "verdict": "correct"}, {"curie": "HP:0002099", "verdict": "incorrect"}]


def test_identity_verdicts_open_and_closed_world():
    m = IdentityMatcher()
    assert judged_verdict("MONDO:0004979", JUDGMENTS, m, closed_world=False) == "correct"
    assert judged_verdict("HP:0002099", JUDGMENTS, m, closed_world=False) == "incorrect"
    assert judged_verdict("DOID:2841", JUDGMENTS, m, closed_world=False) == "unjudged"
    assert judged_verdict("DOID:2841", JUDGMENTS, m, closed_world=True) == "incorrect"


def test_conflicting_when_an_identifier_matches_both_verdicts():
    m = IdentityMatcher()
    judgments = [{"curie": "A:1", "verdict": "correct"}, {"curie": "A:1", "verdict": "incorrect"}]
    assert judged_verdict("A:1", judgments, m, closed_world=False) == "conflicting"


@pytest.mark.integration
def test_babel_matches_other_identifiers_for_the_same_concept():
    m = BabelMatcher()
    assert m.babel_version  # recorded with every evaluation
    m.prefetch(["DOID:2841", "MONDO:0004979", "NCBIGene:7124", "UniProtKB:P01375", "UMLS:C3494254"])
    assert m.same("DOID:2841", "MONDO:0004979")  # asthma
    assert judged_verdict("DOID:2841", JUDGMENTS, m, closed_world=True) == "correct"
    assert m.same("NCBIGene:7124", "UniProtKB:P01375")  # TNF gene and protein, with conflation
    assert not BabelMatcher(conflate=False, drug_chemical_conflate=False).same("NCBIGene:7124", "UniProtKB:P01375")
    assert not m.known("UMLS:C3494254")  # not in Babel
