import httpx
import pytest

from bagel.retrieval import RetrievalConfig, merge_hits, normalize_candidates, retrieve

# shapes as returned by NameRes /lookup, SapBERT /annotate and NodeNorm for "asthma"
NAMERES = [
    {"curie": "MONDO:0004979", "label": "asthma", "score": 50.0, "types": ["biolink:Disease"], "taxa": []},
    {"curie": "MONDO:0850282", "label": "asthma, susceptibility", "score": 40.0, "types": ["biolink:Disease"],
     "taxa": []},
]
SAPBERT = [
    {"curie": "UMLS:C0877430", "name": "Asthma susceptibility", "score": 0.99, "category": "biolink:Disease"},
    {"curie": "UMLS:C9999999", "name": "asthmatic", "score": 0.80, "category": "biolink:Phenomenon"},
]
NORMALIZED = {
    "MONDO:0004979": {"id": {"identifier": "MONDO:0004979", "label": "asthma", "description": "A disease."},
                      "type": ["biolink:Disease"]},
    "MONDO:0850282": {"id": {"identifier": "MONDO:0850282", "label": "asthma susceptibility"},
                      "type": ["biolink:Disease"]},
    "UMLS:C0877430": {"id": {"identifier": "MONDO:0850282", "label": "asthma susceptibility"},
                      "type": ["biolink:Disease"]},
    "UMLS:C9999999": None,
}


def test_cliques_are_combined_keeping_each_sources_best_rank():
    candidates = normalize_candidates(merge_hits(NAMERES, SAPBERT), NORMALIZED, {})
    by_curie = {c.curie: c for c in candidates}
    merged = by_curie["MONDO:0850282"]
    assert (merged.name_res_rank, merged.sapbert_rank) == (2, 1)
    assert sorted(merged.source_curies) == ["MONDO:0850282", "UMLS:C0877430"]


def test_unnormalized_curies_are_kept():
    candidates = normalize_candidates(merge_hits(NAMERES, SAPBERT), NORMALIZED, {})
    assert "UMLS:C9999999" in {c.curie for c in candidates}


def test_order_is_by_best_rank_then_name_res_rank():
    candidates = normalize_candidates(merge_hits(NAMERES, SAPBERT), NORMALIZED, {})
    # MONDO:0004979 and MONDO:0850282 both have best rank 1; name_res rank breaks the tie
    assert [c.curie for c in candidates] == ["MONDO:0004979", "MONDO:0850282", "UMLS:C9999999"]


def test_order_does_not_depend_on_input_order():
    forward = normalize_candidates(merge_hits(NAMERES, SAPBERT), NORMALIZED, {})
    shuffled = normalize_candidates(dict(reversed(list(merge_hits(NAMERES, SAPBERT).items()))), NORMALIZED, {})
    assert [c.curie for c in forward] == [c.curie for c in shuffled]


@pytest.mark.integration
def test_live_retrieval_gives_species_for_every_gene():
    with httpx.Client(timeout=60) as client:
        candidates = retrieve(client, RetrievalConfig(), "Tnf")
    genes = [c for c in candidates if c.category == "biolink:Gene"]
    assert genes and all(c.taxa for c in genes)
    assert "NCBIGene:24835" in {c.curie for c in genes}  # rat Tnf
