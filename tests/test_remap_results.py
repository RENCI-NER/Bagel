import json

from openai_batch_files import remap_results

ANNOTATIONS = {
    "MONDO:1": {"name": "asthma", "category": "biolink:Disease"},
    "MONDO:2": {"name": "allergic asthma", "category": "biolink:Disease"},
}  # colors: alizarin, amaranth


def batch_line(req_id, content=None, error=None):
    if error:
        return {"custom_id": req_id, "response": None, "error": error}
    return {"custom_id": req_id, "response": {"status_code": 200, "body": {"choices": [{"message": {"content": content}}]}}}


def test_remap_statuses(tmp_path):
    ids = ["1-0-6", "2-0-6", "3-0-6", "4-0-6", "5-0-6"]
    with open(tmp_path / "ann.jsonl", "w") as f:
        for i in ids:
            pmid, start, end = i.split("-")
            f.write(json.dumps({"pmid": pmid, "start_index": start, "end_index": end, "entity": "asthma",
                                "annotations": ANNOTATIONS}) + "\n")
    lines = [
        batch_line(ids[0], '```json\n[{"synonymType": "exact", "color-code": "alizarin"}]\n```'),
        batch_line(ids[1], "Sorry, I can't help with that."),   # json error right after a good record
        batch_line(ids[2], '[{"synonym_type": "exact", "color_code": "alizarin"},'
                           ' {"synonym_type": "broad", "color_code": "vermilion"}]'),
        batch_line(ids[3], error={"code": "server_error"}),
        batch_line(ids[4], "[]"),
    ]
    with open(tmp_path / "batch.jsonl", "w") as f:
        f.writelines(json.dumps(l) + "\n" for l in lines)

    stats = remap_results(str(tmp_path / "ann.jsonl"), str(tmp_path / "batch.jsonl"), str(tmp_path / "out.jsonl"))
    out = {r["pmid"]: r for r in map(json.loads, open(tmp_path / "out.jsonl"))}

    assert out["1"]["llm_status"] == "ok"
    assert out["1"]["llm_response_remapped"]["MONDO:1"]["synonym_type"] == "exact"
    # the failed record must not inherit record 1's answer
    assert out["2"]["llm_status"] == "json_error" and out["2"]["llm_response_remapped"] == {}
    assert out["3"]["llm_status"] == "partial" and list(out["3"]["llm_response_remapped"]) == ["MONDO:1"]
    assert out["4"]["llm_status"] == "api_error"
    assert out["5"]["llm_status"] == "ok" and out["5"]["llm_response_remapped"] == {}
    assert stats == {"hallucinations": 1, "hallucinated_items": 1, "unmapped_items": 0, "json_errors": 1,
                     "api_errors": 1, "total_lines": 5}


def test_taxa_labels_handles_old_and_new_annotation_formats():
    from openai_batch_files import taxa_labels
    from util import ner_util
    ner_util.taxa_cache["NCBITaxon:9606"] = "Homo sapiens"
    assert taxa_labels({"taxa": "Homo sapiens", "taxa_id": ["NCBITaxon:9606"]}, "http://unused") == "Homo sapiens"
    assert taxa_labels({"taxa": ["NCBITaxon:9606"]}, "http://unused") == "Homo sapiens"   # pre 2025-07-29 files
    assert taxa_labels({"name": "Tnf", "sapbert_rank": 1}, "http://unused") == ""          # sapbert-only
