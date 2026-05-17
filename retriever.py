"""
retriever.py — TF-IDF Retrieval Layer with PostgreSQL persistence
All training data is stored in Postgres — survives restarts forever.
Built by UE Developers | Owned by Makaram MS
"""

import re
import math
import os
import json
import logging
from collections import defaultdict
from typing import List, Tuple

import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger("leemai.retriever")

STOP_WORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and","or","but",
    "are","was","were","be","been","being","have","has","had","do","does","did",
    "will","would","could","should","may","might","shall","can","not","with",
    "this","that","these","those","from","by","as","up","out","so","if","no",
    "its","also","than","then","there","their","they","what","when","where",
    "which","who","how","why","all","any","each","more","most","other","into",
    "about","after","before","between","through","during","over","under","again",
    "am","i","you","he","she","we","your","my","our","his","her","us",
}


def get_conn():
    """Get a fresh PostgreSQL connection using DATABASE_URL env var."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable not set.")
    return psycopg2.connect(db_url)


def init_db():
    """Create tables if they don't exist. Called once on startup."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS leemai_chunks (
            id          SERIAL PRIMARY KEY,
            chunk_text  TEXT NOT NULL UNIQUE,
            terms_json  TEXT NOT NULL,
            source_name TEXT NOT NULL DEFAULT 'manual',
            created_at  TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS leemai_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    logger.info("DB tables ready.")


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def ngrams(tokens: List[str], n: int = 2) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def extract_terms(text: str) -> List[str]:
    tokens = tokenize(text)
    return tokens + ngrams(tokens, 2)


def split_into_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for p in parts:
        sub = [s.strip() for s in p.split("\n") if s.strip()]
        result.extend(sub)
    return [s for s in result if len(s) > 25]


def build_chunks(text: str, chunk_size: int = 5, overlap: int = 2) -> List[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i+chunk_size])
        if len(chunk) > 40:
            chunks.append(chunk.strip())
    return chunks


class Retriever:
    def __init__(self):
        self.chunks: List[str] = []
        self.chunk_terms: List[List[str]] = []
        self.df: defaultdict = defaultdict(int)
        self.idf: dict = {}
        self.tfidf_matrix: List[dict] = []
        self.doc_count: int = 0
        self.trained: bool = False
        self.training_sources: List[str] = []

    def load(self) -> bool:
        """Load all chunks from PostgreSQL and rebuild TF-IDF index in RAM."""
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("SELECT chunk_text, terms_json, source_name FROM leemai_chunks ORDER BY id")
            rows = cur.fetchall()
            cur.execute("SELECT value FROM leemai_meta WHERE key = 'training_sources'")
            row = cur.fetchone()
            self.training_sources = json.loads(row[0]) if row else []
            cur.close()
            conn.close()

            if not rows:
                logger.info("No training data in DB yet.")
                return False

            self.chunks = []
            self.chunk_terms = []
            self.df = defaultdict(int)

            for chunk_text, terms_json, _ in rows:
                terms = json.loads(terms_json)
                self.chunks.append(chunk_text)
                self.chunk_terms.append(terms)
                for t in set(terms):
                    self.df[t] += 1

            self.doc_count = len(self.chunks)
            self._compute_idf()
            self._compute_tfidf()
            self.trained = True
            logger.info(f"Loaded {self.doc_count} chunks from DB.")
            return True

        except Exception as e:
            logger.error(f"DB load error: {e}")
            return False

    def add_text(self, text: str, source_name: str = "manual") -> dict:
        new_chunks = build_chunks(text, chunk_size=5, overlap=2)
        if not new_chunks:
            return {"status": "error", "message": "No usable content found."}

        added = 0
        existing_set = set(self.chunks)
        rows_to_insert = []

        for chunk in new_chunks:
            if chunk not in existing_set:
                terms = extract_terms(chunk)
                rows_to_insert.append((chunk, json.dumps(terms), source_name))
                self.chunks.append(chunk)
                self.chunk_terms.append(terms)
                for t in set(terms):
                    self.df[t] += 1
                existing_set.add(chunk)
                added += 1

        if rows_to_insert:
            try:
                conn = get_conn()
                cur = conn.cursor()
                execute_values(cur,
                    "INSERT INTO leemai_chunks (chunk_text, terms_json, source_name) "
                    "VALUES %s ON CONFLICT (chunk_text) DO NOTHING",
                    rows_to_insert
                )
                if source_name not in self.training_sources:
                    self.training_sources.append(source_name)
                cur.execute("""
                    INSERT INTO leemai_meta (key, value) VALUES ('training_sources', %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """, (json.dumps(self.training_sources),))
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                logger.error(f"DB save error: {e}")
                return {"status": "error", "message": f"Database error: {str(e)}"}

        self.doc_count = len(self.chunks)
        self._compute_idf()
        self._compute_tfidf()
        self.trained = True

        return {"status": "ok", "chunks_added": added, "total_chunks": self.doc_count, "source": source_name}

    def _compute_idf(self):
        N = self.doc_count
        self.idf = {term: math.log((N + 1) / (freq + 1)) + 1 for term, freq in self.df.items()}

    def _compute_tfidf(self):
        self.tfidf_matrix = []
        for terms in self.chunk_terms:
            tf_raw = defaultdict(int)
            for t in terms:
                tf_raw[t] += 1
            total = len(terms) or 1
            vec = {t: (count / total) * self.idf.get(t, 1.0) for t, count in tf_raw.items()}
            self.tfidf_matrix.append(vec)

    def _cosine(self, va: dict, vb: dict) -> float:
        common = set(va) & set(vb)
        if not common:
            return 0.0
        dot = sum(va[t] * vb[t] for t in common)
        mag_a = math.sqrt(sum(v**2 for v in va.values()))
        mag_b = math.sqrt(sum(v**2 for v in vb.values()))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

    def retrieve(self, query: str, top_k: int = 4, threshold: float = 0.04) -> List[Tuple[str, float]]:
        if not self.trained:
            return []
        terms = extract_terms(query)
        if not terms:
            return []
        tf_raw = defaultdict(int)
        for t in terms:
            tf_raw[t] += 1
        total = len(terms)
        q_vec = {t: (count / total) * self.idf.get(t, 1.0) for t, count in tf_raw.items()}
        scores = [(self._cosine(q_vec, chunk_vec), i) for i, chunk_vec in enumerate(self.tfidf_matrix)]
        scores.sort(reverse=True)
        results = []
        seen = set()
        for score, idx in scores:
            if score < threshold:
                break
            key = self.chunks[idx][:50]
            if key not in seen:
                seen.add(key)
                results.append((self.chunks[idx], round(score, 4)))
            if len(results) >= top_k:
                break
        return results

    def reset(self):
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("DELETE FROM leemai_chunks")
            cur.execute("DELETE FROM leemai_meta")
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"DB reset error: {e}")
        self.chunks = []
        self.chunk_terms = []
        self.df = defaultdict(int)
        self.idf = {}
        self.tfidf_matrix = []
        self.doc_count = 0
        self.trained = False
        self.training_sources = []

    def stats(self) -> dict:
        return {
            "trained": self.trained,
            "total_chunks": self.doc_count,
            "vocabulary_size": len(self.idf),
            "training_sources": self.training_sources,
        }

    def save(self):
        """No-op — data written to DB immediately in add_text()."""
        pass
