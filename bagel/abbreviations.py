"""Stage 1 (prepare): abbreviation maps and mention expansion."""
import json
import re
from pathlib import Path

_WORD_CHAR = r"A-Za-z0-9"


def clean_abbreviation_map(pairs: list[dict]) -> dict[str, str]:
    """
    Builds {abbreviation: definition} from extractor output ([{"abbreviation", "definition"}]).

    Drops entries that can't be expanded safely: single characters and purely numeric/punctuation "abbreviations"
    (extractors return compound numbers like "3"), definitions equal to the abbreviation, and definitions that contain
    the abbreviation as a word (expanding them never terminates). When an abbreviation is defined twice, the first
    definition wins.
    """
    cleaned = {}
    for pair in pairs:
        abbreviation = pair["abbreviation"].strip()
        definition = pair["definition"].strip()
        if len(abbreviation) < 2 or not re.search(r"[A-Za-z]", abbreviation):
            continue
        if definition == abbreviation or _word_pattern([abbreviation]).search(definition):
            continue
        cleaned.setdefault(abbreviation, definition)
    return cleaned


def load_abbreviation_maps(path: str | Path) -> dict[str, dict[str, str]]:
    """Reads a JSONL file of {"pmid", "abbreviation_map": [...]} into {document id: cleaned map}."""
    maps = {}
    with open(path) as stream:
        for line in stream:
            record = json.loads(line)
            maps[str(record["pmid"])] = clean_abbreviation_map(record["abbreviation_map"])
    return maps


def _word_pattern(words):
    # longest first, so "CPA" is tried before "Pa"
    alternatives = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![{_WORD_CHAR}])(?:{alternatives})(?![{_WORD_CHAR}])")


def expand(text: str, abbreviations: dict[str, str]) -> str:
    """
    Replaces whole-word abbreviations in text with their definitions, repeating until nothing changes so nested
    abbreviations are expanded too ("CPA" -> "chronic Pa infection" -> "chronic Pseudomonas aeruginosa infection").
    A definition directly followed by its own abbreviation in parentheses, e.g. "lactate dehydrogenase (LDH)", becomes
    the definition alone rather than "lactate dehydrogenase (lactate dehydrogenase)".
    """
    if not abbreviations:
        return text
    words = _word_pattern(abbreviations)
    for _ in range(len(abbreviations) + 1):
        for abbreviation, definition in abbreviations.items():
            text = re.sub(rf"{re.escape(definition)}\s*\({re.escape(abbreviation)}\)", definition, text)
        expanded = words.sub(lambda m: abbreviations[m.group(0)], text)
        if expanded == text:
            return text
        text = expanded
    raise ValueError(f"Abbreviation expansion did not terminate for {text!r}")
