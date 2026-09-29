from collections import Counter

from bagel.datasets.assessed import read_copy_tables, verdict

DUMP = "\n".join([
    "-- PostgreSQL database dump",
    "COPY public.recognized_entities (id, pmid, expanded_text, original_text) FROM stdin;",
    "7\t25763772\tcystic fibrosis\tCF",
    "8\t25763772\tlong\\ttab\\\\slash\t\\N",
    "\\.",
    "COPY public.assignments (id, assessor) FROM stdin;",
    "1\tsomeone",
    "\\.",
    "",
])


def test_copy_blocks_are_parsed_and_unescaped(tmp_path):
    path = tmp_path / "dump.sql"
    path.write_text(DUMP)
    tables = read_copy_tables(path, tables=("recognized_entities",))
    assert tables["recognized_entities"] == [
        {"id": "7", "pmid": "25763772", "expanded_text": "cystic fibrosis", "original_text": "CF"},
        {"id": "8", "pmid": "25763772", "expanded_text": "long\ttab\\slash", "original_text": None},
    ]


def test_verdict_is_by_majority_with_ties_unsure():
    assert verdict(Counter(agree=2, disagree=1)) == "correct"
    assert verdict(Counter(disagree=1)) == "incorrect"
    assert verdict(Counter(agree=1, disagree=1)) == "unsure"
    assert verdict(Counter(unsure=2)) == "unsure"
