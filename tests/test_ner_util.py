import asyncio

from util.ner_util import get_entity_ids

NN_URL = "http://nodenorm/get_normalized_nodes"


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data, self.status_code = data, status_code

    def json(self):
        return self._data


class FakeSession:
    """ Stands in for httpx.AsyncClient. sapbert/nameres return lists in the given order; nodenorm answers
    keys in `nodenorm_order` (or reversed request order) to show output order doesn't depend on it. """

    def __init__(self, sapbert, nameres, normalized):
        self.sapbert, self.nameres, self.normalized = sapbert, nameres, normalized

    async def post(self, url, json):
        if url == NN_URL:
            return FakeResponse({c: self.normalized.get(c) for c in reversed(json["curies"])})
        return FakeResponse(self.sapbert)

    async def get(self, url):
        return FakeResponse(self.nameres)


def sap(curie, name, score):
    return {"curie": curie, "name": name, "score": score, "category": "biolink:Disease"}


def nr(curie, label, score):
    return {"curie": curie, "label": label, "score": score, "taxa": [], "types": ["biolink:Disease"]}


def nn(identifier, label):
    return {"id": {"identifier": identifier, "label": label, "description": f"{label} desc"},
            "type": ["biolink:Disease"]}


def run(session):
    return asyncio.run(get_entity_ids("asthma", name_res_url="http://nameres/?string=", sapbert_url="http://sapbert",
                                      node_norm_url=NN_URL, session=session))


NORMALIZED = {
    "MONDO:1": nn("MONDO:1", "asthma"),
    "DOID:1": nn("MONDO:1", "asthma"),       # same clique as MONDO:1
    "MONDO:2": nn("MONDO:2", "allergic asthma"),
    "MONDO:3": nn("MONDO:3", "cough variant asthma"),
    "UMLS:9": None,                            # does not normalize
}


def test_order_is_by_best_rank_and_stable():
    sapbert = [sap("DOID:1", "asthma", 0.99), sap("MONDO:3", "cva", 0.9), sap("UMLS:9", "asthmatic", 0.8)]
    nameres = [nr("MONDO:2", "allergic asthma", 50), nr("MONDO:1", "asthma", 40)]
    result = run(FakeSession(sapbert, nameres, NORMALIZED))
    # MONDO:1 best rank 1 (sapbert via DOID:1); MONDO:2 nr rank 1 -> tie on best rank 1, broken by nameres rank
    assert list(result) == ["MONDO:2", "MONDO:1", "MONDO:3", "UMLS:9"]
    # same inputs in a different arrival order (as a set/dict could produce) give the same result
    assert list(run(FakeSession(sapbert, nameres, dict(reversed(list(NORMALIZED.items())))))) == list(result)


def test_clique_collision_merges_ranks_from_both_sources():
    sapbert = [sap("DOID:1", "asthma", 0.99)]
    nameres = [nr("MONDO:2", "allergic asthma", 50), nr("MONDO:1", "asthma", 40)]
    result = run(FakeSession(sapbert, nameres, NORMALIZED))
    merged = result["MONDO:1"]
    assert (merged["sapbert_rank"], merged["sapbert_score"]) == (1, 0.99)
    assert (merged["name_res_rank"], merged["name_res_score"]) == (2, 40)
    assert sorted(merged["source_curies"]) == ["DOID:1", "MONDO:1"]
    assert merged["name"] == "asthma" and merged["description"] == "asthma desc"


def test_collision_keeps_better_rank_per_source():
    sapbert = [sap("MONDO:1", "asthma", 0.99), sap("DOID:1", "asthma (disorder)", 0.95)]
    result = run(FakeSession(sapbert, [], NORMALIZED))
    assert (result["MONDO:1"]["sapbert_rank"], result["MONDO:1"]["sapbert_score"]) == (1, 0.99)


def test_nameres_url_encodes_entity_and_replaces_params():
    from urllib.parse import urlsplit, parse_qs
    from util.ner_util import build_nameres_url
    # server default (already has limit=10) and ORION's base (stray quote after string=)
    for base in ["https://nr/lookup?autocomplete=false&offset=0&limit=10&string=",
                 'https://nr/lookup?autocomplete=false&offset=0&limit=10&string="']:
        url = build_nameres_url(base, "Na+/K+ ATPase & T#1", count=20, entity_type="biolink:Protein")
        query = parse_qs(urlsplit(url).query)
        assert query["string"] == ["Na+/K+ ATPase & T#1"]
        assert query["limit"] == ["20"]
        assert query["biolink_type"] == ["biolink:Protein"]
        assert query["autocomplete"] == ["false"]
    assert "biolink_type" not in build_nameres_url("https://nr/lookup?string=", "x", entity_type="")


def test_sapbert_only_gene_gets_taxa_from_nodenorm():
    from util import ner_util
    ner_util.taxa_cache["NCBITaxon:10090"] = "Mus musculus"
    normalized = {"NCBIGene:21926": {"id": {"identifier": "NCBIGene:21926", "label": "Tnf"},
                                     "type": ["biolink:Gene"], "taxa": ["NCBITaxon:10090"]}}
    result = run(FakeSession([sap("NCBIGene:21926", "Tnf", 0.9)], [], normalized))
    assert result["NCBIGene:21926"]["taxa"] == "Mus musculus"
    assert result["NCBIGene:21926"]["taxa_id"] == ["NCBITaxon:10090"]
