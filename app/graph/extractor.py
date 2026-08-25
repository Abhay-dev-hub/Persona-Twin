"""
Uses an LLM (via OpenRouter) to pull structured facts/opinions/events/
relationships out of a single text chunk, per the schema in `schema.py`.

Requires OPENROUTER_API_KEY in the environment.
"""

import json

from app.graph.schema import EXTRACTION_SCHEMA_DESCRIPTION
from app.llm.openrouter_client import chat_completion

_SYSTEM_PROMPT = (
    "You extract structured information from a piece of text about a person "
    "(the persona), for a knowledge graph. Be precise and conservative: only "
    "extract what is explicitly stated. " + EXTRACTION_SCHEMA_DESCRIPTION
)

_EMPTY_RESULT = {"facts": [], "opinions": [], "events": [], "relationships": []}


def _parse_response(raw_text: str) -> dict:
    """Parse the model's JSON response, tolerating stray markdown fences
    and, defensively, any leading/trailing prose a model might emit
    around the JSON (e.g. a reasoning model that didn't fully respect
    `reasoning.exclude`)."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: extract the outermost {...} block, in case the model
        # wrapped the JSON in prose/reasoning despite instructions.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"Model did not return valid JSON and no JSON object could be "
                f"found. Raw response: {raw_text!r}"
            )
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"Model did not return valid JSON: {e}\nRaw response: {raw_text!r}")

    result = dict(_EMPTY_RESULT)
    for key in result:
        value = data.get(key, [])
        if not isinstance(value, list):
            raise ValueError(f"Expected a list for '{key}', got {type(value).__name__}")
        result[key] = value
    return result


def extract_from_chunk(chunk_text: str, model: str | None = None) -> dict:
    """
    Extract facts/opinions/events/relationships from a single chunk of text.

    Returns a dict: {"facts": [...], "opinions": [...], "events": [...], "relationships": [...]}
    Any category with nothing found comes back as an empty list.
    """
    if not chunk_text.strip():
        return dict(_EMPTY_RESULT)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": chunk_text},
    ]
    # exclude_reasoning strips chain-of-thought for models that expose one
    # (e.g. "thinking" models); max_tokens raised to give room for both
    # reasoning (if not fully excludable) and the JSON payload itself.
    raw_text = chat_completion(messages, model=model, max_tokens=4000, exclude_reasoning=True)
    return _parse_response(raw_text)
