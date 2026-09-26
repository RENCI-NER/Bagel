"""
Parsing of LLM re-ranking responses.

The bagel prompts are inconsistent about output keys: e.g. ask_classes / ask_classes_no_system specify
`synonym_type` / `color_code` but their worked example uses `synonymType` / `color-code`, and ask_classes_cot
uses `relation_type` and emits a <reasoning> block before the JSON. Everything here is tolerant of those variants
so that formatting differences are not mistaken for hallucinated identifiers.
"""
import json
import re
from typing import Any, Dict, List, Tuple

COLOR_KEYS = {"colorcode", "color", "id", "candidateid"}
TYPE_KEYS = {"synonymtype", "relationtype", "synonymoustype", "relation"}
# types meaning "this candidate does not match"; dropped from results (same as the candidate not being listed)
NO_MATCH_TYPES = {"none", "nomatch", "unrelated", "null"}

_FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMOutputError(ValueError):
    """ The LLM response could not be parsed into a list of classified candidates."""


def _normalize(value: str) -> str:
    return re.sub(r"[\s_\-]", "", str(value)).lower()


def _first_json_value(text: str):
    """ Returns the first JSON array/object found in text, preferring an array."""
    decoder = json.JSONDecoder()
    for opener in ("[", "{"):
        for match in re.finditer(re.escape(opener), text):
            try:
                value, _ = decoder.raw_decode(text[match.start():])
                return value
            except json.JSONDecodeError:
                continue
    raise LLMOutputError(f"No JSON found in LLM response: {text[:200]!r}")


def parse_llm_json(response: Any) -> List[Dict]:
    """
    Extracts the list of classified candidates from an LLM response. Accepts a string (optionally with ```json
    fences and/or leading reasoning text), a message object with .content, or already-parsed JSON.
    """
    if hasattr(response, "content"):
        response = response.content
    if isinstance(response, str):
        text = response.strip()
        # drop reasoning that precedes the answer: <reasoning> (ask_classes_cot) and <think> (Qwen3, DeepSeek-R1...)
        for closing_tag in ("</reasoning>", "</think>"):
            if closing_tag in text:
                text = text.split(closing_tag)[-1]
        fenced = _FENCED.findall(text)
        candidates = fenced[::-1] + [text]  # last fenced block first; reasoning may contain earlier ones
        value, error = None, None
        for candidate in candidates:
            try:
                value = json.loads(candidate)
                break
            except json.JSONDecodeError:
                try:
                    value = _first_json_value(candidate)
                    break
                except LLMOutputError as e:
                    error = e
        if value is None:
            raise error or LLMOutputError("Empty LLM response")
        response = value
    if isinstance(response, dict):
        # single object, or a wrapper like {"synonyms": [...]}
        lists = [v for v in response.values() if isinstance(v, list)]
        response = lists[0] if len(lists) == 1 else [response]
    if not isinstance(response, list):
        raise LLMOutputError(f"Expected a JSON list from LLM, got {type(response).__name__}")
    return response


def _get(item: Dict, keys: set):
    for key, value in item.items():
        if _normalize(key) in keys:
            return value
    return None


def map_color_codes(synonym_list, llm_items: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Maps parsed LLM items back to entities by color code.
    :param synonym_list: the Entity objects (with color_code set) that were sent to the LLM
    :param llm_items: output of parse_llm_json
    :return: (entities, problems). entities are Entity dicts plus "synonym_type" (color_code removed); problems
      lists items that could not be mapped, each {"item": ..., "reason": "unknown_color" | "missing_color" |
      "missing_type" | "not_an_object"}.
    """
    by_color = {_normalize(entity.color_code): entity for entity in synonym_list}
    entities, problems = [], []
    for item in llm_items:
        if not isinstance(item, dict):
            problems.append({"item": item, "reason": "not_an_object"})
            continue
        color = _get(item, COLOR_KEYS)
        synonym_type = _get(item, TYPE_KEYS)
        if not color:
            problems.append({"item": item, "reason": "missing_color"})
            continue
        if _normalize(color) not in by_color:
            problems.append({"item": item, "reason": "unknown_color"})
            continue
        if synonym_type is None or not str(synonym_type).strip():
            problems.append({"item": item, "reason": "missing_type"})
            continue
        synonym_type = str(synonym_type).strip().lower()
        if _normalize(synonym_type) in NO_MATCH_TYPES:
            continue
        final = by_color[_normalize(color)].dict()
        final["synonym_type"] = synonym_type
        del final["color_code"]
        entities.append(final)
    return entities, problems
