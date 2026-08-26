# """
# Reads chunks.jsonl (produced by Step 1), runs each chunk through the
# LLM extractor, and writes the results into the Neo4j persona graph.
# """
#
# import json
# import uuid
# from pathlib import Path
#
# from app.graph.extractor import extract_from_chunk
# from app.graph.neo4j_client import GraphClient
#
#
# def load_chunks(jsonl_path: str | Path) -> list[dict]:
#     jsonl_path = Path(jsonl_path)
#     if not jsonl_path.exists():
#         raise FileNotFoundError(f"No such file: {jsonl_path}")
#     chunks = []
#     with jsonl_path.open(encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 chunks.append(json.loads(line))
#     return chunks
#
#
# def build_graph(
#     jsonl_path: str | Path,
#     persona_name: str,
#     client: GraphClient,
#     model: str | None = None,
#     verbose: bool = True,
# ) -> dict:
#     """
#     Process every chunk in `jsonl_path` and write extracted
#     facts/opinions/events/relationships into the graph, attached to
#     `persona_name`.
#
#     Returns a summary dict with counts of each type written.
#     """
#     chunks = load_chunks(jsonl_path)
#     client.ensure_constraints()
#     client.upsert_person(persona_name, is_persona=True)
#
#     totals = {"facts": 0, "opinions": 0, "events": 0, "relationships": 0, "chunks_failed": 0}
#
#     for i, chunk in enumerate(chunks):
#         text = chunk.get("text", "")
#         source_id = chunk.get("source_id", "")
#         source_path = chunk.get("source_path", "")
#
#         if verbose:
#             print(f"  [{i + 1}/{len(chunks)}] extracting from {source_path} (chunk {chunk.get('index')})")
#
#         try:
#             extracted = extract_from_chunk(text, model=model)
#         except Exception as e:
#             print(f"    [skip] extraction failed: {e}")
#             totals["chunks_failed"] += 1
#             continue
#
#         for fact in extracted["facts"]:
#             client.upsert_fact(
#                 fact_id=str(uuid.uuid4()),
#                 text=fact.get("text", ""),
#                 category=fact.get("category", "other"),
#                 source_id=source_id,
#                 source_path=source_path,
#                 persona_name=persona_name,
#             )
#             totals["facts"] += 1
#
#         for opinion in extracted["opinions"]:
#             client.upsert_opinion(
#                 opinion_id=str(uuid.uuid4()),
#                 text=opinion.get("text", ""),
#                 topic=opinion.get("topic", ""),
#                 sentiment=opinion.get("sentiment", "neutral"),
#                 source_id=source_id,
#                 source_path=source_path,
#                 persona_name=persona_name,
#             )
#             totals["opinions"] += 1
#
#         for event in extracted["events"]:
#             client.upsert_event(
#                 event_id=str(uuid.uuid4()),
#                 text=event.get("text", ""),
#                 date=event.get("date", ""),
#                 location=event.get("location", ""),
#                 source_id=source_id,
#                 source_path=source_path,
#                 persona_name=persona_name,
#             )
#             totals["events"] += 1
#
#         for rel in extracted["relationships"]:
#             other = rel.get("person", "").strip()
#             relation_type = rel.get("relation_type", "unknown").strip()
#             if other:
#                 client.upsert_relationship(persona_name, other, relation_type)
#                 totals["relationships"] += 1
#
#     return totals
"""
Reads chunks.jsonl (produced by Step 1), runs each chunk through the
LLM extractor, and writes the results into the Neo4j persona graph.
"""

import json
import uuid
from pathlib import Path

from app.graph.extractor import extract_from_chunk
from app.graph.neo4j_client import GraphClient


def load_chunks(jsonl_path: str | Path) -> list[dict]:
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No such file: {jsonl_path}")
    chunks = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_graph(
    jsonl_path: str | Path,
    persona_name: str,
    client: GraphClient,
    model: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Process every chunk in `jsonl_path` and write extracted
    facts/opinions/events/relationships into the graph, attached to
    `persona_name`.

    Returns a summary dict with counts of each type written.
    """
    chunks = load_chunks(jsonl_path)
    client.ensure_constraints()
    client.upsert_person(persona_name, is_persona=True)

    totals = {"facts": 0, "opinions": 0, "events": 0, "relationships": 0, "chunks_failed": 0}

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        source_id = chunk.get("source_id", "")
        source_path = chunk.get("source_path", "")

        if verbose:
            print(f"  [{i + 1}/{len(chunks)}] extracting from {source_path} (chunk {chunk.get('index')})")

        try:
            extracted = extract_from_chunk(text, model=model)
        except Exception as e:
            print(f"    [skip] extraction failed: {e}")
            totals["chunks_failed"] += 1
            continue

        # Defensive: extract_from_chunk should always return a well-formed
        # dict, but guard anyway rather than crash the whole run on a
        # single unexpected chunk.
        if not isinstance(extracted, dict):
            print(f"    [skip] extraction returned unexpected type: {type(extracted).__name__}")
            totals["chunks_failed"] += 1
            continue

        for fact in extracted.get("facts") or []:
            if not isinstance(fact, dict) or not fact.get("text"):
                continue
            client.upsert_fact(
                fact_id=str(uuid.uuid4()),
                text=fact.get("text", ""),
                category=fact.get("category", "other"),
                source_id=source_id,
                source_path=source_path,
                persona_name=persona_name,
            )
            totals["facts"] += 1

        for opinion in extracted.get("opinions") or []:
            if not isinstance(opinion, dict) or not opinion.get("text"):
                continue
            client.upsert_opinion(
                opinion_id=str(uuid.uuid4()),
                text=opinion.get("text", ""),
                topic=opinion.get("topic", ""),
                sentiment=opinion.get("sentiment", "neutral"),
                source_id=source_id,
                source_path=source_path,
                persona_name=persona_name,
            )
            totals["opinions"] += 1

        for event in extracted.get("events") or []:
            if not isinstance(event, dict) or not event.get("text"):
                continue
            client.upsert_event(
                event_id=str(uuid.uuid4()),
                text=event.get("text", ""),
                date=event.get("date", ""),
                location=event.get("location", ""),
                source_id=source_id,
                source_path=source_path,
                persona_name=persona_name,
            )
            totals["events"] += 1

        for rel in extracted.get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            other = (rel.get("person") or "").strip()
            relation_type = (rel.get("relation_type") or "unknown").strip()
            if other:
                client.upsert_relationship(persona_name, other, relation_type)
                totals["relationships"] += 1

    return totals