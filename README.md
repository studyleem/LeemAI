# 🧠 Leem AI
**Built by UE Developers · Owned by Makaram MS**

A fully self-contained AI assistant for StudyLeem. No external APIs. No GPU needed.
Trained on plain text using TF-IDF retrieval — runs comfortably on Railway's free tier.

---

## Project Structure

```
leemai/
├── brain.py          ← TF-IDF AI engine (no external ML libs)
├── main.py           ← FastAPI backend + REST API
├── static/
│   └── index.html    ← Chat UI (served by FastAPI)
├── requirements.txt
├── Procfile
├── railway.toml
└── .gitignore
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Chat UI |
| GET | `/api/health` | Server health check |
| GET | `/api/stats` | Brain stats (chunks, vocab, sources) |
| POST | `/api/ask` | Ask a question |
| POST | `/api/train/text` | Train on plain text |
| POST | `/api/train/file` | Train by uploading .txt or .json file |
| POST | `/api/reset` | Reset all training data |

### Ask a question
```json
POST /api/ask
{ "question": "What is a buffer solution?" }
```

### Train on text
```json
POST /api/train/text
{ "text": "A buffer solution is...", "source_name": "Chapter 3" }
```

---

## Deploy to Railway (Step by Step)

### Step 1 — Push to GitHub
```bash
cd leemai
git init
git add .
git commit -m "Leem AI v1.0"
git remote add origin https://github.com/YOUR_USERNAME/leemai.git
git push -u origin main
```

### Step 2 — Create Railway project
1. Go to **railway.com** → Login → **New Project**
2. Select **Deploy from GitHub repo**
3. Connect your GitHub and select the `leemai` repo
4. Railway auto-detects Python via Nixpacks

### Step 3 — Set environment (optional)
Railway reads `$PORT` automatically. No extra env vars needed.

### Step 4 — Deploy
Railway builds and deploys automatically on every `git push`.

### Step 5 — Get your URL
Go to your Railway project → **Settings** → **Domains** → Generate domain.
Your Leem AI will be live at: `https://leemai-xxxx.up.railway.app`

---

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Open: http://localhost:8000
```

---

## How Training Works

1. You paste text or upload a `.txt`/`.json` file
2. Leem AI splits it into overlapping chunks of ~4 sentences
3. Each chunk is tokenized and TF-IDF weighted (no ML library needed)
4. Data is saved to `leemai_data.pkl` (persists across restarts on Railway volumes)

## How Answering Works

1. Your question is tokenized the same way
2. Cosine similarity is computed against all stored chunks
3. Top-matching chunks are merged into a clean answer
4. Confidence is reported: high / medium / low

---

## Limitations

- Best for factual Q&A on content you've trained it with
- Does not generate new sentences — retrieves and combines from trained text
- For GPU-based generation, use Hugging Face Spaces instead
- Training data resets on Railway free tier if no persistent volume is attached
  → Add a Railway Volume mounted at `/app` to persist `leemai_data.pkl`

---

*Leem AI · StudyLeem · UE Developers*
