"""
Orchestrates the full persona-creation pipeline (extract -> chunk ->
graph -> embed) triggered from the UI, running in a background thread
so the upload request returns immediately and the frontend can poll
status instead of blocking on what might be a multi-minute job.
"""

import re
import threading
import traceback
import uuid
from pathlib import Path

from app.graph.neo4j_client import GraphClient
from app.graph.pipeline import build_graph
from app.ingestion.pipeline import process_directory, write_jsonl
from app.storage import db
from app.vector.pipeline import embed_and_store
from app.vector.qdrant_client import VectorClient

UPLOADS_ROOT = Path("data") / "personas"
CHUNKS_ROOT = Path("data") / "output" / "personas"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return slug or "persona"


def unique_collection_name(name: str) -> str:
    """Qdrant collection names double as the persona's slug identity.
    Append a short suffix if the slug is already taken."""
    base = slugify(name)
    existing = {p["collection_name"] for p in db.list_personas()}
    if base not in existing:
        return base
    return f"{base}_{uuid.uuid4().hex[:6]}"


def register_persona(name: str) -> dict:
    """Creates the DB record only (status='pending'). Caller is
    responsible for saving uploaded files to UPLOADS_ROOT / persona['id']
    before calling start_background()."""
    collection_name = unique_collection_name(name)
    return db.create_persona(name, collection_name)


def start_background(persona_id: str, name: str, collection_name: str) -> None:
    """Kicks off the extract -> graph -> embed pipeline in a background
    thread. Assumes source files already exist under
    UPLOADS_ROOT / persona_id (see register_persona)."""
    thread = threading.Thread(
        target=_run_pipeline, args=(persona_id, name, collection_name), daemon=True
    )
    thread.start()


def _run_pipeline(persona_id: str, name: str, collection_name: str) -> None:
    try:
        db.update_persona_status(persona_id, "processing")

        raw_dir = UPLOADS_ROOT / persona_id
        chunks_path = CHUNKS_ROOT / f"{persona_id}.jsonl"
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        if chunks_path.exists():
            chunks_path.unlink()

        chunks = process_directory(raw_dir)
        if not chunks:
            raise RuntimeError("No text could be extracted from the uploaded file(s).")
        write_jsonl(chunks, chunks_path)

        with GraphClient() as graph_client:
            build_graph(chunks_path, name, graph_client, verbose=False)

        with VectorClient() as vector_client:
            embed_and_store(chunks_path, collection_name, vector_client, verbose=False)

        db.update_persona_status(persona_id, "ready")

    except Exception as e:
        traceback.print_exc()
        db.update_persona_status(persona_id, "error", error_message=str(e))