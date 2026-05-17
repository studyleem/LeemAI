"""
main.py — Leem AI Backend
FastAPI server combining TF-IDF retrieval + Flan-T5 generation.
Built by UE Developers | Owned by Makaram MS
"""

import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from retriever import Retriever, init_db
from generator import load_model, generate_answer, is_ready

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("leemai")

# ── Global retriever ───────────────────────────────────────────────────────────
retriever = Retriever()

# ── Startup: init DB, load data, start model ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Leem AI starting up...")
    # 1. Create DB tables if they don't exist
    init_db()
    # 2. Load training data from PostgreSQL into memory
    retriever.load()
    # 3. Load Flan-T5 model in background thread (non-blocking)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_model)
    yield
    logger.info("Leem AI shutting down.")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Leem AI",
    description="Leem AI Backend — Built by UE Developers, owned by Makaram MS",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Vercel frontend will call this
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4

class TrainTextRequest(BaseModel):
    text: str
    source_name: Optional[str] = "manual"

class ResetRequest(BaseModel):
    confirm: bool = False

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "Leem AI",
        "version": "2.0.0",
        "built_by": "UE Developers",
        "owner": "Makaram MS",
        "status": "running",
        "model_ready": is_ready(),
        "docs": "/docs",
    }


@app.get("/api/health")
async def health():
    stats = retriever.stats()
    return {
        "status": "ok",
        "model_ready": is_ready(),
        "trained": stats["trained"],
        "total_chunks": stats["total_chunks"],
    }


@app.get("/api/warmup")
async def warmup():
    """
    Call this on page load from your Vercel frontend.
    Triggers model load if not already loaded so first question is fast.
    """
    if is_ready():
        return {"status": "ready", "message": "Model already loaded."}
    # Trigger load in background
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_model)
    return {"status": "loading", "message": "Model is loading, will be ready in ~20-40s."}


@app.get("/api/stats")
async def stats():
    s = retriever.stats()
    s["model_ready"] = is_ready()
    s["version"] = "2.0.0"
    s["built_by"] = "UE Developers"
    s["owner"] = "Makaram MS"
    return s


@app.post("/api/ask")
async def ask(req: AskRequest):
    """
    Main Q&A endpoint.
    1. Retrieves top_k relevant chunks via TF-IDF
    2. Feeds them as context to Flan-T5-small
    3. Returns a generated answer
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > 800:
        raise HTTPException(status_code=400, detail="Question too long (max 800 chars).")

    if not retriever.trained:
        return {
            "answer": "I haven't been trained on any content yet. Please upload study material using the admin panel.",
            "confidence": 0.0,
            "model_ready": is_ready(),
            "fallback": True,
        }

    # Step 1: Retrieve
    top_chunks = retriever.retrieve(question, top_k=req.top_k or 4)

    # Step 2: Generate
    result = generate_answer(question, top_chunks)

    # Step 3: Compute confidence from retrieval scores
    if top_chunks:
        best_score = top_chunks[0][1]
        if best_score > 0.35:
            conf_label = "high"
        elif best_score > 0.12:
            conf_label = "medium"
        else:
            conf_label = "low"
    else:
        best_score = 0.0
        conf_label = "low"

    return {
        "answer": result["answer"],
        "confidence": best_score,
        "confidence_label": conf_label,
        "generation_time": result.get("generation_time", 0),
        "model_ready": result["model_ready"],
        "fallback": result.get("fallback", False),
        "sources_count": len(top_chunks),
    }


@app.post("/api/train/text")
async def train_text(req: TrainTextRequest):
    """Train Leem AI on raw plain text."""
    text = req.text.strip()
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Text too short (min 50 chars).")
    if len(text) > 600_000:
        raise HTTPException(status_code=400, detail="Text too large. Split into parts under 600,000 chars.")

    result = retriever.add_text(text, source_name=req.source_name or "manual")
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])

    retriever.save()
    return result


@app.post("/api/train/file")
async def train_file(file: UploadFile = File(...)):
    """Train by uploading a .txt or .json file."""
    if not file.filename.endswith((".txt", ".json")):
        raise HTTPException(status_code=400, detail="Only .txt and .json files supported.")

    content = await file.read()

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except Exception:
            raise HTTPException(status_code=400, detail="Cannot decode file. Use UTF-8.")

    if file.filename.endswith(".json"):
        try:
            data = json.loads(text)
            text = _extract_text_from_json(data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file.")

    if len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="File has too little content.")

    result = retriever.add_text(text, source_name=file.filename)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])

    retriever.save()
    result["filename"] = file.filename
    return result


@app.post("/api/reset")
async def reset(req: ResetRequest):
    """Wipe all training data. Requires confirm=true."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to reset all data.")
    retriever.reset()
    return {"status": "ok", "message": "All training data cleared."}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _extract_text_from_json(obj, depth: int = 0) -> str:
    if depth > 12:
        return ""
    if isinstance(obj, str):
        return obj + "\n"
    if isinstance(obj, list):
        return "\n".join(_extract_text_from_json(i, depth+1) for i in obj)
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            label = str(k).replace("_", " ")
            parts.append(f"{label}: {_extract_text_from_json(v, depth+1)}")
        return "\n".join(parts)
    return str(obj)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
