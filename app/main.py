# """
# Step 5: the chat backend now retrieves context (Neo4j graph profile +
# Qdrant vector search) for each message and responds *as* the persona,
# grounded in that context — not as a generic assistant.
#
# Configured via .env:
#     PERSONA_NAME       — required for persona-grounded responses (e.g. "Jane Doe")
#     QDRANT_COLLECTION  — required for vector retrieval (e.g. "jane_doe")
#
# If Neo4j/Qdrant aren't configured or aren't reachable, the backend
# degrades gracefully — it still responds, just without that source of
# grounding (see app/persona/retrieval.py).
# """
#
# import logging
# import os
# from contextlib import asynccontextmanager
# from pathlib import Path
#
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
#
# from app.graph.neo4j_client import GraphClient
# from app.llm.openrouter_client import chat_completion
# from app.persona.prompt_builder import build_persona_prompt
# from app.persona.retrieval import retrieve_context
# from app.vector.qdrant_client import VectorClient
#
# load_dotenv()
#
# logger = logging.getLogger("persona_twin")
#
# PERSONA_NAME = os.environ.get("PERSONA_NAME")
# QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION")
#
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Connect once at startup, not per-request. Each client is optional —
#     # if it can't connect, we log a warning and keep running in degraded
#     # mode rather than refusing to start the whole app.
#     graph_client = None
#     vector_client = None
#
#     try:
#         graph_client = GraphClient()
#         graph_client.verify_connectivity()
#         logger.info("Connected to Neo4j.")
#     except Exception as e:
#         logger.warning("Neo4j not available, graph context will be skipped: %s", e)
#         graph_client = None
#
#     try:
#         vector_client = VectorClient()
#         vector_client.verify_connectivity()
#         logger.info("Connected to Qdrant.")
#     except Exception as e:
#         logger.warning("Qdrant not available, vector context will be skipped: %s", e)
#         vector_client = None
#
#     app.state.graph_client = graph_client
#     app.state.vector_client = vector_client
#
#     yield
#
#     if graph_client is not None:
#         graph_client.close()
#     if vector_client is not None:
#         vector_client.close()
#
#
# app = FastAPI(title="Persona Twin — Chat (Step 5)", lifespan=lifespan)
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # fine for local dev; tighten before any real deployment
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# STATIC_DIR = Path(__file__).parent / "static"
#
#
# class ChatMessage(BaseModel):
#     role: str  # "user" | "assistant"
#     content: str
#
#
# class ChatRequest(BaseModel):
#     message: str
#     history: list[ChatMessage] = []
#
#
# class ChatResponse(BaseModel):
#     reply: str
#
#
# @app.get("/health")
# def health() -> dict:
#     return {
#         "status": "ok",
#         "persona_configured": bool(PERSONA_NAME and QDRANT_COLLECTION),
#         "graph_connected": app.state.graph_client is not None if hasattr(app.state, "graph_client") else False,
#         "vector_connected": app.state.vector_client is not None if hasattr(app.state, "vector_client") else False,
#     }
#
#
# @app.post("/api/chat", response_model=ChatResponse)
# def chat(request: ChatRequest) -> ChatResponse:
#     if not request.message.strip():
#         raise HTTPException(status_code=400, detail="message must not be empty")
#
#     messages = []
#
#     if PERSONA_NAME and QDRANT_COLLECTION:
#         context = retrieve_context(
#             query=request.message,
#             persona_name=PERSONA_NAME,
#             collection_name=QDRANT_COLLECTION,
#             graph_client=app.state.graph_client,
#             vector_client=app.state.vector_client,
#         )
#         system_prompt = build_persona_prompt(PERSONA_NAME, context)
#         messages.append({"role": "system", "content": system_prompt})
#     else:
#         logger.warning(
#             "PERSONA_NAME/QDRANT_COLLECTION not set in .env — responding without persona grounding."
#         )
#
#     messages.extend({"role": m.role, "content": m.content} for m in request.history)
#     messages.append({"role": "user", "content": request.message})
#
#     try:
#         reply = chat_completion(messages)
#     except RuntimeError as e:
#         # covers both "missing API key" and "OpenRouter request failed"
#         raise HTTPException(status_code=502, detail=str(e))
#
#     return ChatResponse(reply=reply)
#
#
# @app.get("/")
# def index() -> FileResponse:
#     return FileResponse(STATIC_DIR / "index.html")
#
#
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

"""
Step 6+: multi-persona chat backend with persistent chat history.

Personas are created via the UI (upload source files -> background
pipeline: extract -> graph -> embed), stored in a local SQLite DB
alongside their Neo4j/Qdrant identifiers. Each persona can have
multiple chat sessions, each with saved message history.
"""

import logging
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.graph.neo4j_client import GraphClient
from app.llm.openrouter_client import chat_completion
from app.persona.manager import UPLOADS_ROOT, register_persona, start_background
from app.persona.prompt_builder import build_persona_prompt
from app.persona.retrieval import retrieve_context
from app.storage import db
from app.vector.qdrant_client import VectorClient

load_dotenv()

logger = logging.getLogger("persona_twin")

db.init_db()


def _connect_graph() -> GraphClient | None:
    try:
        client = GraphClient()
        client.verify_connectivity()
        return client
    except Exception as e:
        logger.warning("Neo4j not available: %s", e)
        return None


def _connect_vector() -> VectorClient | None:
    try:
        client = VectorClient()
        client.verify_connectivity()
        return client
    except Exception as e:
        logger.warning("Qdrant not available: %s", e)
        return None


app = FastAPI(title="Persona Twin — Multi-Persona Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"

# One shared connection to each store, reused across personas/requests —
# retrieval always passes the specific persona name / collection name
# per call, so a single connection works for all personas.
_graph_client = _connect_graph()
_vector_client = _connect_vector()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "graph_connected": _graph_client is not None,
        "vector_connected": _vector_client is not None,
    }


# ---------- personas ----------

class CreatePersonaResponse(BaseModel):
    id: str
    name: str
    collection_name: str
    status: str


@app.get("/api/personas")
def api_list_personas() -> list[dict]:
    return db.list_personas()


@app.get("/api/personas/{persona_id}")
def api_get_persona(persona_id: str) -> dict:
    persona = db.get_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona


@app.post("/api/personas", response_model=CreatePersonaResponse)
async def api_create_persona(name: str, files: list[UploadFile]) -> dict:
    if not name.strip():
        raise HTTPException(status_code=400, detail="name must not be empty")
    if not files:
        raise HTTPException(status_code=400, detail="at least one file is required")

    persona = register_persona(name)

    raw_dir = UPLOADS_ROOT / persona["id"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = raw_dir / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)

    start_background(persona["id"], persona["name"], persona["collection_name"])

    return persona


@app.delete("/api/personas/{persona_id}")
def api_delete_persona(persona_id: str) -> dict:
    persona = db.get_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    db.delete_persona(persona_id)
    shutil.rmtree(UPLOADS_ROOT / persona_id, ignore_errors=True)
    return {"deleted": True}


# ---------- chats ----------

class CreateChatRequest(BaseModel):
    persona_id: str
    title: str = "New chat"


@app.get("/api/chats")
def api_list_chats(persona_id: str) -> list[dict]:
    return db.list_chats(persona_id)


@app.post("/api/chats")
def api_create_chat(request: CreateChatRequest) -> dict:
    persona = db.get_persona(request.persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return db.create_chat(request.persona_id, request.title)


@app.get("/api/chats/{chat_id}/messages")
def api_get_messages(chat_id: str) -> list[dict]:
    if not db.get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return db.list_messages(chat_id)


@app.delete("/api/chats/{chat_id}")
def api_delete_chat(chat_id: str) -> dict:
    if not db.get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete_chat(chat_id)
    return {"deleted": True}


# ---------- chat (send a message) ----------

class ChatRequest(BaseModel):
    chat_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    chat_row = db.get_chat(request.chat_id)
    if not chat_row:
        raise HTTPException(status_code=404, detail="Chat not found")

    persona = db.get_persona(chat_row["persona_id"])
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    if persona["status"] != "ready":
        raise HTTPException(status_code=409, detail=f"Persona is not ready yet (status: {persona['status']})")

    history = db.list_messages(request.chat_id)

    context = retrieve_context(
        query=request.message,
        persona_name=persona["name"],
        collection_name=persona["collection_name"],
        graph_client=_graph_client,
        vector_client=_vector_client,
    )
    system_prompt = build_persona_prompt(persona["name"], context)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend({"role": m["role"], "content": m["content"]} for m in history)
    messages.append({"role": "user", "content": request.message})

    try:
        reply = chat_completion(messages)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    db.add_message(request.chat_id, "user", request.message)
    db.add_message(request.chat_id, "assistant", reply)

    # Auto-title the chat from the first message, so the sidebar isn't
    # just a wall of "New chat" entries.
    if not history:
        title = request.message.strip()[:50]
        db.rename_chat(request.chat_id, title)

    return ChatResponse(reply=reply)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")