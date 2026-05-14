"""
LeemAI Engine v2
================
Lightweight, independent AI for FBISE curriculum.
No external API. No heavy ML model. Pure BM25 + smart extraction.

Architecture:
  1. BM25 retrieval  → finds most relevant knowledge chunks
  2. Sentence scorer → picks best sentences from those chunks
  3. Answer builder  → formats answer based on question type

To expand the knowledge base, add more entries to leemai_all_chunks.json.
Each entry can be a plain string OR an object:
  { "text": "...", "subject": "chemistry", "class": 11, "chapter": "..." }
"""

import json
import re
import os
import math
from rank_bm25 import BM25Okapi

# ─────────────────────────────────────────────
# Stopwords (English + common Urdu romanized)
# ─────────────────────────────────────────────
STOPWORDS = {
    'the','a','an','is','are','was','were','in','on','at','to','for',
    'of','and','or','but','with','this','that','it','be','by','from',
    'as','what','how','why','when','where','who','which','do','does',
    'did','have','has','had','will','would','could','should','may',
    'might','shall','can','its','their','our','your','my','his','her',
    'we','they','i','you','he','she','not','no','any','all','some',
    'if','then','so','than','more','most','very','just','also','about',
    'into','out','up','down','over','such','each','both','after','before',
    # Common Urdu romanized words students might use
    'kya','hai','ka','ki','ke','mein','se','ko','aur','ya','nahi','tha',
}

# ─────────────────────────────────────────────
# Question type → regex patterns
# ─────────────────────────────────────────────
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


class LeemAIEngine:
    """
    Main engine class. Instantiate once, call .answer(query) repeatedly.
    Thread-safe for read-only operations after __init__.
    """

    def __init__(self, chunks_path: str = None):
        if chunks_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            chunks_path = os.path.join(base, 'leemai_all_chunks.json')

        with open(chunks_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # Support both plain strings and rich objects
        self.chunks: list[str] = []
        self.meta: list[dict] = []
        for item in raw:
            if isinstance(item, str):
                self.chunks.append(item)
                self.meta.append({})
            elif isinstance(item, dict):
                self.chunks.append(item.get('text', ''))
                self.meta.append({k: v for k, v in item.items() if k != 'text'})

        # Build BM25 index once
        tokenized_corpus = [self._tokenize(c) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"[LeemAI] Ready — {len(self.chunks)} chunks indexed.")

    # ──────────────────────────────────────────
    # Text processing
    # ──────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, strip non-alphanumeric, remove stopwords, light stemming."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = []
        for t in text.split():
            if t in STOPWORDS or len(t) < 3:
                continue
            # Light suffix stemming (keeps related forms together)
            if t.endswith('ing') and len(t) > 6:   t = t[:-3]
            elif t.endswith('tion') and len(t) > 7: t = t[:-4]
            elif t.endswith('ity') and len(t) > 6:  t = t[:-3]
            elif t.endswith('ies') and len(t) > 5:  t = t[:-3] + 'y'
            elif t.endswith('ed') and len(t) > 5:   t = t[:-2]
            elif t.endswith('es') and len(t) > 4:   t = t[:-2]
            elif t.endswith('s') and len(t) > 4:    t = t[:-1]
            tokens.append(t)
        return tokens

    def _classify_question(self, query: str) -> str:
        q = query.lower().strip()
        for qtype, patterns in QUESTION_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, q):
                    return qtype
        return 'general'

    def _extract_topic(self, query: str) -> str:
        """Strip question prefix → get the core topic."""
        q = query.strip()
        q = re.sub(
            r'^(what (is|are|was|were|do you mean by)|how (does|do|is|are|can|to)|'
            r'why (is|are|does)|define|explain|describe|list|name|give|tell me about|'
            r'what causes|difference between|compare|formula for|equation for|'
            r'example of|examples of|mention|state|elaborate)\s+',
            '', q, flags=re.IGNORECASE
        )
        q = re.sub(r'\?+$', '', q).strip()
        return q if q else query.strip('?').strip()

    # ──────────────────────────────────────────
    # Sentence scoring
    # ──────────────────────────────────────────

    def _score_sentences(self, chunk: str, query_tokens: list[str]) -> list[tuple]:
        """Return list of (score, original_index, sentence) sorted by score."""
        # Split on sentence endings OR newlines
        raw_sents = re.split(r'(?<=[.!?])\s+|\n{1,}', chunk)
        sentences = [(i, s.strip()) for i, s in enumerate(raw_sents)
                     if len(s.strip()) > 25]

        if not sentences:
            return []

        query_set = set(query_tokens)
        scored = []

        for orig_idx, sent in sentences:
            s_tokens = set(self._tokenize(sent))
            if not s_tokens:
                continue

            overlap     = len(s_tokens & query_set)
            density     = overlap / (len(s_tokens) + 1)
            word_count  = len(sent.split())

            # Prefer 10–45 word sentences
            if 10 <= word_count <= 45:
                length_score = 1.0
            elif word_count < 10:
                length_score = word_count / 10
            else:
                length_score = min(45 / word_count, 1.0)

            # Earlier sentences carry more definitional weight
            position_score = 1 / math.log(orig_idx + 2)

            score = (overlap * 2.5) + (density * 3.0) + (length_score * 0.6) + (position_score * 0.4)
            scored.append((score, orig_idx, sent))

        scored.sort(key=lambda x: -x[0])
        return scored

    # ──────────────────────────────────────────
    # Answer extraction
    # ──────────────────────────────────────────

    def _get_best_sentences(self, results: list[tuple], query: str,
                             n: int = 4) -> list[str]:
        """Collect top-n sentences across all result chunks."""
        query_tokens = self._tokenize(query)
        pool = []

        for rank, (chunk, bm25_score) in enumerate(results):
            boost = bm25_score / (rank + 1)
            for score, _, sent in self._score_sentences(chunk, query_tokens):
                pool.append((score * boost, sent))

        pool.sort(key=lambda x: -x[0])

        # Deduplicate near-identical sentences
        seen, final = set(), []
        for _, sent in pool:
            sig = sent[:60].lower()
            if sig not in seen:
                seen.add(sig)
                final.append(sent)
            if len(final) >= n:
                break

        return final

    def _extract_list_items(self, chunk: str, query_tokens: list[str]) -> list[str]:
        """Pull bullet/numbered items from a chunk."""
        items = []
        for line in chunk.split('\n'):
            line = line.strip()
            if re.match(r'^(\d+[\.\):]|[-–•*]|\([a-z]\))\s+\w', line) and len(line) > 15:
                clean = re.sub(r'^(\d+[\.\):]|[-–•*]|\([a-z]\))\s+', '', line)
                items.append(clean)

        if not items:
            # Fall back: best sentences as list
            scored = self._score_sentences(chunk, query_tokens)
            items = [s for _, _, s in scored[:5]]

        return items[:6]

    # ──────────────────────────────────────────
    # Response formatting
    # ──────────────────────────────────────────

    def _format(self, qtype: str, topic: str, content: list[str],
                meta: dict = None) -> str:
        topic_title = topic.title() if topic else ''
        body = ' '.join(content)

        chapter_tag = ''
        if meta and meta.get('chapter'):
            chapter_tag = f'\n\n_Chapter: {meta["chapter"]}_'

        templates = {
            'definition':  f'**{topic_title}**\n\n{body}{chapter_tag}',
            'explanation': f'**How {topic} works:**\n\n{body}{chapter_tag}',
            'reason':      f'**Why {topic}:**\n\n{body}{chapter_tag}',
            'formula':     f'**Formula / Equation — {topic_title}:**\n\n{body}{chapter_tag}',
            'comparison':  f'**Comparison — {topic_title}:**\n\n{body}{chapter_tag}',
            'process':     f'**Process / Steps — {topic_title}:**\n\n{body}{chapter_tag}',
            'example':     f'**Examples of {topic}:**\n\n{body}{chapter_tag}',
            'list':        f'**{topic_title}:**\n\n' + '\n'.join(f'• {s}' for s in content) + chapter_tag,
            'general':     body + chapter_tag,
        }
        return templates.get(qtype, body + chapter_tag)

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def answer(self, query: str, top_k: int = 5) -> dict:
        """
        Query the knowledge base.
        Returns: { answer, confidence, type, topic }
        """
        query = query.strip()
        if not query:
            return {'answer': 'Please ask a question.', 'confidence': 0, 'type': 'error', 'topic': ''}

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {'answer': 'Please ask a more specific question.', 'confidence': 0,
                    'type': 'error', 'topic': ''}

        # ── BM25 retrieval ──
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[-top_k:][::-1]

        MIN_SCORE = 0.4
        results = [
            (self.chunks[i], float(scores[i]), self.meta[i])
            for i in top_indices
            if scores[i] >= MIN_SCORE
        ]

        topic = self._extract_topic(query)
        qtype = self._classify_question(query)

        if not results:
            return {
                'answer': (
                    f"I don't have information about **{topic}** in my knowledge base yet.\n"
                    "Please check your textbook or ask your teacher."
                ),
                'confidence': 0,
                'type': 'not_found',
                'topic': topic,
            }

        # ── Extract answer ──
        chunk_results = [(c, s) for c, s, _ in results]
        top_meta = results[0][2]

        if qtype == 'list':
            items = self._extract_list_items(results[0][0], query_tokens)
            if len(items) < 2:
                # supplement from 2nd result
                items += self._extract_list_items(results[1][0], query_tokens) if len(results) > 1 else []
            answer_text = self._format(qtype, topic, items, top_meta)

        else:
            n = 5 if qtype in ('explanation', 'process') else 3
            sents = self._get_best_sentences(chunk_results, query, n=n)

            if not sents:
                answer_text = results[0][0][:600]
            else:
                answer_text = self._format(qtype, topic, sents, top_meta)

        # ── Confidence (0–95) ──
        raw = results[0][1]
        # BM25 scores vary by corpus; normalize empirically
        confidence = min(int((raw / 12.0) * 100), 95)

        return {
            'answer':     answer_text,
            'confidence': confidence,
            'type':       qtype,
            'topic':      topic,
        }
