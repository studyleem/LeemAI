"""
Leem AI Brain — TF-IDF Retrieval Engine
No external API. Fully self-contained.
Built by UE Developers | Owned by Makaram MS
"""

import re
import math
import json
import os
import pickle
from collections import defaultdict
from typing import List, Tuple, Optional

# ── Stop words (lightweight, no NLTK needed) ─────────────────────────────────
STOP_WORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and","or","but",
    "are","was","were","be","been","being","have","has","had","do","does","did",
    "will","would","could","should","may","might","shall","can","not","with",
    "this","that","these","those","from","by","as","up","out","so","if","no",
    "its","also","than","then","there","their","they","what","when","where",
    "which","who","how","why","all","any","each","more","most","other","into",
    "about","after","before","between","through","during","over","under","again",
    "further","am","i","you","he","she","we","your","my","our","his","her","us",
}

DATA_FILE = "leemai_data.pkl"

# ── Text utilities ─────────────────────────────────────────────────────────────
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

# ── Sentence splitter ──────────────────────────────────────────────────────────
def split_sentences(text: str) -> List[str]:
    # Split on . ! ? followed by space/newline/end
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    # Also split on newlines for list-style content
    result = []
    for p in parts:
        sub = [s.strip() for s in p.split("\n") if s.strip()]
        result.extend(sub)
    return [s for s in result if len(s) > 20]

# ── Chunk builder: sliding window over sentences ──────────────────────────────
def build_chunks(text: str, chunk_size: int = 3, overlap: int = 1) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i+chunk_size])
        if len(chunk) > 30:
            chunks.append(chunk)
    return chunks

# ── TF-IDF Engine ─────────────────────────────────────────────────────────────
class LeemBrain:
    def __init__(self):
        self.chunks: List[str] = []          # raw text chunks
        self.chunk_terms: List[List[str]] = [] # tokenized terms per chunk
        self.df: defaultdict = defaultdict(int)  # document frequency
        self.idf: dict = {}
        self.tfidf_matrix: List[dict] = []   # TF-IDF vectors
        self.doc_count: int = 0
        self.trained: bool = False
        self.training_sources: List[str] = []

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, text: str, source_name: str = "manual") -> dict:
        new_chunks = build_chunks(text, chunk_size=4, overlap=1)
        if not new_chunks:
            return {"status": "error", "message": "No usable content found in text."}

        added = 0
        for chunk in new_chunks:
            if chunk not in self.chunks:  # avoid duplicates
                self.chunks.append(chunk)
                terms = extract_terms(chunk)
                self.chunk_terms.append(terms)
                for t in set(terms):
                    self.df[t] += 1
                added += 1

        self.doc_count = len(self.chunks)
        self._compute_idf()
        self._compute_tfidf()
        self.trained = True

        if source_name not in self.training_sources:
            self.training_sources.append(source_name)

        self.save()
        return {
            "status": "ok",
            "chunks_added": added,
            "total_chunks": self.doc_count,
            "source": source_name
        }

    def _compute_idf(self):
        N = self.doc_count
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((N + 1) / (freq + 1)) + 1  # smooth IDF

    def _compute_tfidf(self):
        self.tfidf_matrix = []
        for terms in self.chunk_terms:
            tf_raw = defaultdict(int)
            for t in terms:
                tf_raw[t] += 1
            total = len(terms) if terms else 1
            vec = {}
            for t, count in tf_raw.items():
                tf = count / total
                idf = self.idf.get(t, 1.0)
                vec[t] = tf * idf
            self.tfidf_matrix.append(vec)

    # ── Query ─────────────────────────────────────────────────────────────────
    def _vectorize_query(self, query: str) -> dict:
        terms = extract_terms(query)
        tf_raw = defaultdict(int)
        for t in terms:
            tf_raw[t] += 1
        total = len(terms) if terms else 1
        vec = {}
        for t, count in tf_raw.items():
            tf = count / total
            idf = self.idf.get(t, 1.0)
            vec[t] = tf * idf
        return vec

    def _cosine_sim(self, vec_a: dict, vec_b: dict) -> float:
        common = set(vec_a) & set(vec_b)
        if not common:
            return 0.0
        dot = sum(vec_a[t] * vec_b[t] for t in common)
        mag_a = math.sqrt(sum(v**2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v**2 for v in vec_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def answer(self, question: str, top_k: int = 3, threshold: float = 0.05) -> dict:
        if not self.trained or not self.chunks:
            return {
                "answer": "I haven't been trained on any content yet. Please upload some study material first.",
                "confidence": 0.0,
                "sources": []
            }

        q_vec = self._vectorize_query(question)
        if not q_vec:
            return {
                "answer": "I couldn't understand your question. Try rephrasing it.",
                "confidence": 0.0,
                "sources": []
            }

        # Score all chunks
        scores: List[Tuple[float, int]] = []
        for i, chunk_vec in enumerate(self.tfidf_matrix):
            sim = self._cosine_sim(q_vec, chunk_vec)
            scores.append((sim, i))

        scores.sort(reverse=True)
        top = [(s, i) for s, i in scores[:top_k] if s >= threshold]

        if not top:
            return {
                "answer": "I don't have enough information to answer that. Try training me on more related content.",
                "confidence": 0.0,
                "sources": []
            }

        # Build answer from top chunks
        best_score = top[0][0]
        answer_parts = []
        seen = set()
        for score, idx in top:
            chunk = self.chunks[idx]
            # De-duplicate overlapping chunks
            key = chunk[:60]
            if key not in seen:
                seen.add(key)
                answer_parts.append(chunk)

        # Format answer
        if len(answer_parts) == 1:
            final_answer = answer_parts[0]
        else:
            # Join and clean
            combined = " ".join(answer_parts)
            # Remove repeated sentences
            sents = split_sentences(combined)
            unique_sents = []
            seen_s = set()
            for s in sents:
                k = s[:40].lower()
                if k not in seen_s:
                    seen_s.add(k)
                    unique_sents.append(s)
            final_answer = " ".join(unique_sents)

        # Confidence label
        if best_score > 0.4:
            conf_label = "high"
        elif best_score > 0.15:
            conf_label = "medium"
        else:
            conf_label = "low"

        return {
            "answer": final_answer.strip(),
            "confidence": round(best_score, 3),
            "confidence_label": conf_label,
            "sources_count": len(top)
        }

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self):
        with open(DATA_FILE, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "chunk_terms": self.chunk_terms,
                "df": dict(self.df),
                "idf": self.idf,
                "tfidf_matrix": self.tfidf_matrix,
                "doc_count": self.doc_count,
                "trained": self.trained,
                "training_sources": self.training_sources,
            }, f)

    def load(self) -> bool:
        if not os.path.exists(DATA_FILE):
            return False
        try:
            with open(DATA_FILE, "rb") as f:
                d = pickle.load(f)
            self.chunks = d["chunks"]
            self.chunk_terms = d["chunk_terms"]
            self.df = defaultdict(int, d["df"])
            self.idf = d["idf"]
            self.tfidf_matrix = d["tfidf_matrix"]
            self.doc_count = d["doc_count"]
            self.trained = d["trained"]
            self.training_sources = d.get("training_sources", [])
            return True
        except Exception:
            return False

    def stats(self) -> dict:
        return {
            "trained": self.trained,
            "total_chunks": self.doc_count,
            "vocabulary_size": len(self.idf),
            "training_sources": self.training_sources,
        }

    def reset(self):
        self.chunks = []
        self.chunk_terms = []
        self.df = defaultdict(int)
        self.idf = {}
        self.tfidf_matrix = []
        self.doc_count = 0
        self.trained = False
        self.training_sources = []
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
