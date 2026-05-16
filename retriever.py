"""
retriever.py — TF-IDF Retrieval Layer
Finds the most relevant chunks from trained data for a given query.
Built by UE Developers | Owned by Makaram MS
"""

import re
import math
import os
import pickle
from collections import defaultdict
from typing import List, Tuple

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

DATA_FILE = "leemai_data.pkl"


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
    """Sliding window chunker — larger chunks = more context for Flan-T5."""
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

    def add_text(self, text: str, source_name: str = "manual") -> dict:
        new_chunks = build_chunks(text, chunk_size=5, overlap=2)
        if not new_chunks:
            return {"status": "error", "message": "No usable content found."}

        added = 0
        for chunk in new_chunks:
            if chunk not in self.chunks:
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

        return {"status": "ok", "chunks_added": added, "total_chunks": self.doc_count}

    def _compute_idf(self):
        N = self.doc_count
        self.idf = {
            term: math.log((N + 1) / (freq + 1)) + 1
            for term, freq in self.df.items()
        }

    def _compute_tfidf(self):
        self.tfidf_matrix = []
        for terms in self.chunk_terms:
            tf_raw = defaultdict(int)
            for t in terms:
                tf_raw[t] += 1
            total = len(terms) or 1
            vec = {t: (count / total) * self.idf.get(t, 1.0)
                   for t, count in tf_raw.items()}
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
        """Return top_k (chunk, score) pairs for a query."""
        if not self.trained:
            return []

        terms = extract_terms(query)
        if not terms:
            return []

        tf_raw = defaultdict(int)
        for t in terms:
            tf_raw[t] += 1
        total = len(terms)
        q_vec = {t: (count / total) * self.idf.get(t, 1.0)
                 for t, count in tf_raw.items()}

        scores = [
            (self._cosine(q_vec, chunk_vec), i)
            for i, chunk_vec in enumerate(self.tfidf_matrix)
        ]
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

    def reset(self):
        self.__init__()
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)

    def stats(self) -> dict:
        return {
            "trained": self.trained,
            "total_chunks": self.doc_count,
            "vocabulary_size": len(self.idf),
            "training_sources": self.training_sources,
        }
