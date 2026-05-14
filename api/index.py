import json
import re
import os
import math
from http.server import BaseHTTPRequestHandler
from rank_bm25 import BM25Okapi

# ── Secret API Key ────────────────────────────────────────────────────────────
SECRET_KEY = "leemai_studyleem_2025"   # Change this to anything secret

# ── Stopwords ─────────────────────────────────────────────────────────────────
STOPWORDS = {
    'the','a','an','is','are','was','were','in','on','at','to','for',
    'of','and','or','but','with','this','that','it','be','by','from',
    'as','what','how','why','when','where','who','which','do','does',
    'did','have','has','had','will','would','could','should','may',
    'might','shall','can','its','their','our','your','my','his','her',
    'we','they','i','you','he','she','not','no','any','all','some',
    'if','then','so','than','more','most','very','just','also','about',
    'into','out','up','down','over','such','each','both','after','before',
    'kya','hai','ka','ki','ke','mein','se','ko','aur','ya','nahi','tha',
}

# ── Question type patterns ────────────────────────────────────────────────────
QUESTION_PATTERNS = {
    'definition':  [r'^what (is|are|was|were)\b', r'^define\b', r'^definition of\b',
                    r'^meaning of\b', r'^what do you mean by\b'],
    'explanation': [r'^how (does|do|is|are|can|could)\b', r'^explain\b',
                    r'^describe\b', r'^elaborate\b'],
    'reason':      [r'^why\b', r'^what (causes|makes|leads|results)\b',
                    r'^give (the )?reason\b'],
    'list':        [r'^list\b', r'^name\b', r'^give.*(types|examples|kinds|uses|properties)\b',
                    r'^what are the (types|kinds|examples|properties|characteristics|uses|factors)\b',
                    r'^mention\b', r'^state\b'],
    'example':     [r'^give.*(example|instance)\b', r'^example of\b', r'^examples of\b'],
    'formula':     [r'\bformula\b', r'\bequation\b', r'\bexpression for\b'],
    'comparison':  [r'\bdifference between\b', r'\bcompare\b', r'\bvs\b',
                    r'\bdistinguish\b', r'\bsimilarities\b'],
    'process':     [r'^how (to|do you|can you)\b', r'\bprocess of\b', r'\bsteps\b',
                    r'\bprocedure\b'],
}

# ── Engine ────────────────────────────────────────────────────────────────────
class LeemAIEngine:

    def __init__(self):
        # Load chunks from same folder as this file
        base = os.path.dirname(os.path.abspath(__file__))
        chunks_path = os.path.join(base, 'leemai_all_chunks.json')

        with open(chunks_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        self.chunks = []
        self.meta   = []
        for item in raw:
            if isinstance(item, str):
                self.chunks.append(item)
                self.meta.append({})
            elif isinstance(item, dict):
                self.chunks.append(item.get('text', ''))
                self.meta.append({k: v for k, v in item.items() if k != 'text'})

        tokenized = [self._tokenize(c) for c in self.chunks]
        self.bm25  = BM25Okapi(tokenized)
        print(f"[LeemAI] Ready — {len(self.chunks)} chunks indexed.")

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = []
        for t in text.split():
            if t in STOPWORDS or len(t) < 3:
                continue
            if   t.endswith('ing')  and len(t) > 6: t = t[:-3]
            elif t.endswith('tion') and len(t) > 7: t = t[:-4]
            elif t.endswith('ity')  and len(t) > 6: t = t[:-3]
            elif t.endswith('ies')  and len(t) > 5: t = t[:-3] + 'y'
            elif t.endswith('ed')   and len(t) > 5: t = t[:-2]
            elif t.endswith('es')   and len(t) > 4: t = t[:-2]
            elif t.endswith('s')    and len(t) > 4: t = t[:-1]
            tokens.append(t)
        return tokens

    def _classify(self, query):
        q = query.lower().strip()
        for qtype, patterns in QUESTION_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, q):
                    return qtype
        return 'general'

    def _topic(self, query):
        q = re.sub(
            r'^(what (is|are|was|were|do you mean by)|how (does|do|is|are|can|to)|'
            r'why (is|are|does)|define|explain|describe|list|name|give|tell me about|'
            r'what causes|difference between|compare|formula for|equation for|'
            r'example of|examples of|mention|state|elaborate)\s+',
            '', query.strip(), flags=re.IGNORECASE
        )
        return re.sub(r'\?+$', '', q).strip() or query.strip('?').strip()

    def _score_sentences(self, chunk, query_tokens):
        parts = re.split(r'(?<=[.!?])\s+|\n+', chunk)
        sentences = [(i, s.strip()) for i, s in enumerate(parts) if len(s.strip()) > 25]
        if not sentences:
            return []
        qset   = set(query_tokens)
        scored = []
        for idx, sent in sentences:
            stok   = set(self._tokenize(sent))
            if not stok: continue
            overlap = len(stok & qset)
            density = overlap / (len(stok) + 1)
            wc      = len(sent.split())
            lscore  = 1.0 if 10 <= wc <= 45 else (wc/10 if wc < 10 else min(45/wc, 1.0))
            pscore  = 1 / math.log(idx + 2)
            score   = overlap*2.5 + density*3.0 + lscore*0.6 + pscore*0.4
            scored.append((score, idx, sent))
        scored.sort(key=lambda x: -x[0])
        return scored

    def _best_sentences(self, results, query, n=4):
        qtok = self._tokenize(query)
        pool = []
        for rank, (chunk, bscore) in enumerate(results):
            boost = bscore / (rank + 1)
            for score, _, sent in self._score_sentences(chunk, qtok):
                pool.append((score * boost, sent))
        pool.sort(key=lambda x: -x[0])
        seen, final = set(), []
        for _, sent in pool:
            sig = sent[:60].lower()
            if sig not in seen:
                seen.add(sig)
                final.append(sent)
            if len(final) >= n:
                break
        return final

    def _list_items(self, chunk, qtok):
        items = []
        for line in chunk.split('\n'):
            line = line.strip()
            if re.match(r'^(\d+[\.\):]|[-–•*]|\([a-z]\))\s+\w', line) and len(line) > 15:
                items.append(re.sub(r'^(\d+[\.\):]|[-–•*]|\([a-z]\))\s+', '', line))
        if not items:
            items = [s for _, _, s in self._score_sentences(chunk, qtok)[:5]]
        return items[:6]

    def _format(self, qtype, topic, content, meta):
        title = topic.title()
        body  = ' '.join(content)
        chtag = f'\n\n_Chapter: {meta["chapter"]}_' if meta and meta.get('chapter') else ''
        t = {
            'definition':  f'**{title}**\n\n{body}{chtag}',
            'explanation': f'**How {topic} works:**\n\n{body}{chtag}',
            'reason':      f'**Why {topic}:**\n\n{body}{chtag}',
            'formula':     f'**Formula — {title}:**\n\n{body}{chtag}',
            'comparison':  f'**Comparison — {title}:**\n\n{body}{chtag}',
            'process':     f'**Steps — {title}:**\n\n{body}{chtag}',
            'example':     f'**Examples of {topic}:**\n\n{body}{chtag}',
            'list':        f'**{title}:**\n\n' + '\n'.join(f'• {s}' for s in content) + chtag,
            'general':     body + chtag,
        }
        return t.get(qtype, body + chtag)

    def answer(self, query):
        query = query.strip()
        qtok  = self._tokenize(query)
        if not query or not qtok:
            return {'answer': 'Please ask a more specific question.', 'confidence': 0,
                    'type': 'error', 'topic': ''}

        scores     = self.bm25.get_scores(qtok)
        top_idx    = scores.argsort()[-5:][::-1]
        results    = [(self.chunks[i], float(scores[i]), self.meta[i])
                      for i in top_idx if scores[i] >= 0.4]

        topic = self._topic(query)
        qtype = self._classify(query)

        if not results:
            return {
                'answer': f"I don't have information about **{topic}** yet.\nPlease check your textbook.",
                'confidence': 0, 'type': 'not_found', 'topic': topic,
            }

        cr      = [(c, s) for c, s, _ in results]
        tmeta   = results[0][2]
        qtok2   = self._tokenize(query)

        if qtype == 'list':
            items = self._list_items(results[0][0], qtok2)
            if len(items) < 2 and len(results) > 1:
                items += self._list_items(results[1][0], qtok2)
            answer_text = self._format(qtype, topic, items, tmeta)
        else:
            n     = 5 if qtype in ('explanation', 'process') else 3
            sents = self._best_sentences(cr, query, n)
            answer_text = self._format(qtype, topic, sents or [results[0][0][:600]], tmeta)

        return {
            'answer':     answer_text,
            'confidence': min(int((results[0][1] / 12.0) * 100), 95),
            'type':       qtype,
            'topic':      topic,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = LeemAIEngine()
    return _engine

try:
    get_engine()
except Exception as e:
    print(f"[LeemAI] Init error: {e}")


# ── Vercel Handler ────────────────────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        self._send(200, {
            'status':  'LeemAI is running',
            'version': '2.0',
            'usage':   'POST /api — body: {"query": "your question"}',
        })

    def do_POST(self):
        try:
            # ── API Key check ──
            key = self.headers.get('x-api-key', '')
            if key != SECRET_KEY:
                self._send(401, {'error': 'Unauthorized'})
                return

            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                self._send(400, {'error': 'Empty body'})
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
                self._send(400, {'error': 'Query too long'})
                return

            result = get_engine().answer(query)
            self._send(200, result)

        except Exception as e:
            self._send(500, {'error': 'Server error', 'detail': str(e)})

    def _send(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self._cors()
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin',  '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, x-api-key')

    def log_message(self, fmt, *args):
        pass
