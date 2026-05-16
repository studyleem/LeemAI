# 🧠 Leem AI v2 — Backend
**Built by UE Developers · Owned by Makaram MS**

Self-contained AI backend for StudyLeem.  
**No external AI API.** Uses Flan-T5-small (Google) locally for answer generation.

---

## How It Works

```
User Question
     │
     ▼
TF-IDF Retriever  ──► finds top 4 relevant chunks from trained data
     │
     ▼
Flan-T5-small     ──► reads chunks as context, GENERATES a real answer
     │
     ▼
JSON Response     ──► sent to your Vercel frontend
```

- **Retriever:** pure Python TF-IDF — no ML libs, ultra-fast
- **Generator:** `google/flan-t5-small` — 300MB RAM, instruction-tuned, CPU-only
- **Storage:** `leemai_data.pkl` — pickle file (add Railway Volume to persist)

---

## API Reference

### `GET /api/health`
```json
{ "status": "ok", "model_ready": true, "trained": true, "total_chunks": 120 }
```

### `GET /api/warmup`
Call this when your Vercel page loads. Pre-loads model so first answer is fast.
```json
{ "status": "ready", "message": "Model already loaded." }
```

### `GET /api/stats`
```json
{
  "trained": true,
  "total_chunks": 120,
  "vocabulary_size": 3400,
  "training_sources": ["Chapter3.txt", "leemai_clean.json"],
  "model_ready": true
}
```

### `POST /api/ask`
```json
// Request
{ "question": "What is a buffer solution?", "top_k": 4 }

// Response
{
  "answer": "A buffer solution resists changes in pH when small amounts of acid or base are added. It typically consists of a weak acid and its conjugate base.",
  "confidence": 0.38,
  "confidence_label": "high",
  "generation_time": 4.2,
  "model_ready": true,
  "fallback": false,
  "sources_count": 3
}
```

### `POST /api/train/text`
```json
// Request
{ "text": "A buffer solution is...", "source_name": "Chapter 3" }

// Response
{ "status": "ok", "chunks_added": 14, "total_chunks": 134 }
```

### `POST /api/train/file`
Multipart form upload. Accepts `.txt` and `.json`.
```
curl -X POST https://YOUR_RAILWAY_URL/api/train/file \
  -F "file=@chapter3.txt"
```

### `POST /api/reset`
```json
{ "confirm": true }
```

---

## Deploy to Railway

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Leem AI v2"
git remote add origin https://github.com/YOUR_USERNAME/leemai-backend.git
git push -u origin main
```

### Step 2 — Create Railway project
1. Go to **railway.com** → New Project → Deploy from GitHub
2. Select `leemai-backend` repo
3. Railway auto-detects Python via Nixpacks

### Step 3 — Add Volume (important for data persistence)
1. In Railway project → **+ New** → **Volume**
2. Mount path: `/app`
3. This persists `leemai_data.pkl` across restarts

### Step 4 — Get your URL
Settings → Domains → Generate domain  
Example: `https://leemai-backend-production.up.railway.app`

**Give this URL to the frontend developer to connect your Vercel site.**

---

## Run Locally

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart transformers sentencepiece
uvicorn main:app --reload --port 8000
```

First run downloads Flan-T5-small (~300MB). Cached after that.

---

## RAM Budget (Railway Free = 512MB)

| Component | RAM |
|-----------|-----|
| Python + FastAPI | ~40 MB |
| Flan-T5-small model | ~240 MB |
| Torch (CPU) runtime | ~30 MB |
| Retriever + data | ~10–30 MB |
| **Total** | **~320–340 MB** ✅ |

Leaves ~170MB headroom — safe on free tier.

---

## Notes

- **Cold start:** model loads in 20–40s after Railway wakes the container
- Call `/api/warmup` on page load to pre-load the model
- Training data survives restarts only if a Railway Volume is mounted at `/app`
- `.json` files from the chunk converter are directly supported

---

*Leem AI v2 · StudyLeem · UE Developers*
