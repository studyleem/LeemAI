"""
api/ask.py — LeemAI Vercel Serverless Endpoint
================================================
Route: POST /api/ask  →  { "query": "your question" }
Route: GET  /api/ask  →  health check

Deploy structure:
  /
  ├── api/
  │   └── ask.py              ← this file
  ├── leemAI_engine.py
  ├── leemai_all_chunks.json
  ├── requirements.txt
  └── vercel.json
"""

import sys
import os

# Make root importable from inside /api/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from http.server import BaseHTTPRequestHandler

# ── Module-level singleton ──────────────────────────────────────────────────
# Vercel reuses warm containers; loading once here saves ~200ms per request.
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        from leemAI_engine import LeemAIEngine
        _engine = LeemAIEngine()
    return _engine

# Pre-warm on cold start
try:
    _get_engine()
except Exception as e:
    print(f"[LeemAI] Warm-up failed: {e}")


# ── Vercel handler ──────────────────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):

    # ── CORS preflight ──
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    # ── Health check ──
    def do_GET(self):
        self._send(200, {
            'status':   'LeemAI is running',
            'version':  '2.0',
            'engine':   'BM25 + Extractive QA',
            'endpoint': 'POST /api/ask — body: {"query": "your question"}',
        })

    # ── Main query endpoint ──
    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._send(400, {'error': 'Empty request body'})
                return

            body = self.rfile.read(length)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send(400, {'error': 'Invalid JSON'})
                return

            query = (data.get('query') or '').strip()

            if not query:
                self._send(400, {'error': 'No query provided'})
                return

            if len(query) > 500:
                self._send(400, {'error': 'Query too long (max 500 characters)'})
                return

            # ── Get answer ──
            engine = _get_engine()
            result = engine.answer(query)
            self._send(200, result)

        except Exception as e:
            self._send(500, {'error': 'Server error', 'detail': str(e)})

    # ── Helpers ──
    def _send(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self._cors()
        self.send_header('Content-Type',   'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        # Allow calls from StudyLeem frontend
        self.send_header('Access-Control-Allow-Origin',  '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, format, *args):
        pass  # Suppress default noisy logging
