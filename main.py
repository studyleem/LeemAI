"""
Leem AI — FastAPI Backend
Built by UE Developers | Owned by Makaram MS
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import os

from brain import LeemBrain

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Leem AI",
    description="Leem AI — Built by UE Developers, owned by Makaram MS",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global brain instance ──────────────────────────────────────────────────────
brain = LeemBrain()
brain.load()   # load saved data if exists

# ── Serve static frontend ──────────────────────────────────────────────────────
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Schemas ───────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class TrainRequest(BaseModel):
    text: str
    source_name: Optional[str] = "manual"

class ResetRequest(BaseModel):
    confirm: bool = False

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    """Serve the Leem AI chat interface."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "Leem AI is running. UI not found."})


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "name": "Leem AI",
        "built_by": "UE Developers",
        "owner": "Makaram MS",
        "version": "1.0.0"
    }


@app.get("/api/stats")
async def stats():
    """Return brain training stats."""
    return brain.stats()


@app.post("/api/ask")
async def ask(req: QuestionRequest):
    """Ask Leem AI a question."""
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(q) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 chars).")
    result = brain.answer(q)
    return result


@app.post("/api/train/text")
async def train_text(req: TrainRequest):
    """Train Leem AI on plain text."""
    text = req.text.strip()
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Text too short. Need at least 50 characters.")
    if len(text) > 500_000:
        raise HTTPException(status_code=400, detail="Text too large (max 500,000 chars). Split into parts.")
    result = brain.train(text, source_name=req.source_name or "manual")
    return result


@app.post("/api/train/file")
async def train_file(file: UploadFile = File(...)):
    """Train Leem AI by uploading a .txt or .json file."""
    if not file.filename.endswith((".txt", ".json")):
        raise HTTPException(status_code=400, detail="Only .txt and .json files supported.")

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode file. Use UTF-8 encoding.")

    # If JSON, extract text values
    if file.filename.endswith(".json"):
        import json
        try:
            data = json.loads(text)
            text = extract_text_from_json(data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file.")

    if len(text.strip()) < 50:
        raise HTTPException(status_code=400, detail="File content too short.")

    result = brain.train(text, source_name=file.filename)
    return result


def extract_text_from_json(obj, depth=0) -> str:
    """Recursively extract all string values from JSON."""
    if depth > 10:
        return ""
    if isinstance(obj, str):
        return obj + "\n"
    if isinstance(obj, list):
        return "\n".join(extract_text_from_json(item, depth+1) for item in obj)
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            parts.append(str(k).replace("_", " ") + ": " + extract_text_from_json(v, depth+1))
        return "\n".join(parts)
    return str(obj)


@app.post("/api/reset")
async def reset(req: ResetRequest):
    """Reset all training data."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true to reset.")
    brain.reset()
    return {"status": "ok", "message": "All training data cleared."}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
