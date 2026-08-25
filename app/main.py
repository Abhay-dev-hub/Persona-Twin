"""
Step 5: the chat backend now retrieves context (Neo4j graph profile +
Qdrant vector search) for each message and responds *as* the persona,
grounded in that context — not as a generic assistant.

Configured via .env:
    PERSONA_NAME       — required for persona-grounded responses (e.g. "Jane Doe")
    QDRANT_COLLECTION  — required for vector retrieval (e.g. "jane_doe")

If Neo4j/Qdrant aren't configured or aren't reachable, the backend
degrades gracefully — it still responds, just without that source of
grounding (see app/persona/retrieval.py).
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.graph.neo4j_client import GraphClient
from app.llm.openrouter_client import chat_completion
from app.persona.prompt_builder import build_persona_prompt
from app.persona.retrieval import retrieve_context
from app.vector.qdrant_client import VectorClient

load_dotenv()

logger = logging.getLogger("persona_twin")

PERSONA_NAME = os.environ.get("PERSONA_NAME")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect once at startup, not per-request. Each client is optional —
    # if it can't connect, we log a warning and keep running in degraded
    # mode rather than refusing to start the whole app.
    graph_client = None
    vector_client = None

    try:
        graph_client = GraphClient()
        graph_client.verify_connectivity()
        logger.info("Connected to Neo4j.")
    except Exception as e:
        logger.warning("Neo4j not available, graph context will be skipped: %s", e)
        graph_client = None

    try:
        vector_client = VectorClient()
        vector_client.verify_connectivity()
        logger.info("Connected to Qdrant.")
    except Exception as e:
        logger.warning("Qdrant not available, vector context will be skipped: %s", e)
        vector_client = None

    app.state.graph_client = graph_client
    app.state.vector_client = vector_client

    yield

    if graph_client is not None:
        graph_client.close()
    if vector_client is not None:
        vector_client.close()


app = FastAPI(title="Persona Twin — Chat (Step 5)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "persona_configured": bool(PERSONA_NAME and QDRANT_COLLECTION),
        "graph_connected": app.state.graph_client is not None if hasattr(app.state, "graph_client") else False,
        "vector_connected": app.state.vector_client is not None if hasattr(app.state, "vector_client") else False,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    messages = []

    if PERSONA_NAME and QDRANT_COLLECTION:
        context = retrieve_context(
            query=request.message,
            persona_name=PERSONA_NAME,
            collection_name=QDRANT_COLLECTION,
            graph_client=app.state.graph_client,
            vector_client=app.state.vector_client,
        )
        system_prompt = build_persona_prompt(PERSONA_NAME, context)
        messages.append({"role": "system", "content": system_prompt})
    else:
        logger.warning(
            "PERSONA_NAME/QDRANT_COLLECTION not set in .env — responding without persona grounding."
        )

    messages.extend({"role": m.role, "content": m.content} for m in request.history)
    messages.append({"role": "user", "content": request.message})

    try:
        reply = chat_completion(messages)
    except RuntimeError as e:
        # covers both "missing API key" and "OpenRouter request failed"
        raise HTTPException(status_code=502, detail=str(e))

    return ChatResponse(reply=reply)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
