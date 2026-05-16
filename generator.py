"""
generator.py — Flan-T5-small Generative Layer
Reads retrieved context + question → generates a real answer.
Model is loaded once on startup and kept in RAM.
Built by UE Developers | Owned by Makaram MS
"""

import os
import time
import logging
from typing import Optional

logger = logging.getLogger("leemai.generator")

# Suppress HuggingFace noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

MODEL_NAME = "google/flan-t5-small"   # ~300 MB RAM — fits Railway free tier

_pipeline = None   # loaded once, stays in memory


def load_model() -> bool:
    """Load Flan-T5-small into memory. Called once at startup."""
    global _pipeline
    if _pipeline is not None:
        return True
    try:
        logger.info("Loading Flan-T5-small model...")
        t0 = time.time()

        from transformers import pipeline
        _pipeline = pipeline(
            "text2text-generation",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            max_new_tokens=180,
            do_sample=False,          # deterministic — best for factual Q&A
            device=-1,                # CPU only (no GPU on Railway free)
        )
        logger.info(f"Model loaded in {time.time()-t0:.1f}s")
        return True
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        _pipeline = None
        return False


def is_ready() -> bool:
    return _pipeline is not None


def generate_answer(question: str, context_chunks: list, max_context_chars: int = 900) -> dict:
    """
    Given a question and retrieved context chunks, generate an answer.
    Returns dict with answer, generation_time, model_ready.
    """
    if not is_ready():
        return {
            "answer": "The AI model is still loading. Please wait a moment and try again.",
            "generation_time": 0,
            "model_ready": False,
            "fallback": True,
        }

    if not context_chunks:
        return {
            "answer": "I don't have enough information to answer that. Please train me with relevant content first.",
            "generation_time": 0,
            "model_ready": True,
            "fallback": True,
        }

    # ── Build context string from top retrieved chunks ─────────────────────
    context_parts = []
    total_chars = 0
    for chunk, score in context_chunks:
        if total_chars + len(chunk) > max_context_chars:
            # Add truncated if there's still room
            remaining = max_context_chars - total_chars
            if remaining > 80:
                context_parts.append(chunk[:remaining])
            break
        context_parts.append(chunk)
        total_chars += len(chunk)

    context = " ".join(context_parts)

    # ── Flan-T5 prompt (instruction-style works best) ──────────────────────
    prompt = (
        f"You are a helpful study assistant. "
        f"Use the following context to answer the question clearly and concisely.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # ── Generate ───────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        result = _pipeline(
            prompt,
            max_new_tokens=180,
            num_beams=3,              # beam search = more coherent output
            early_stopping=True,
            no_repeat_ngram_size=3,   # avoid repetitive phrases
        )
        raw_answer = result[0]["generated_text"].strip()
        gen_time = round(time.time() - t0, 2)

        # Clean up any prompt leakage
        raw_answer = raw_answer.replace("Answer:", "").strip()

        if not raw_answer or len(raw_answer) < 5:
            raw_answer = context_parts[0] if context_parts else "No answer found."

        return {
            "answer": raw_answer,
            "generation_time": gen_time,
            "model_ready": True,
            "fallback": False,
        }

    except Exception as e:
        logger.error(f"Generation error: {e}")
        # Graceful fallback — return best retrieved chunk
        fallback = context_parts[0] if context_parts else "Could not generate answer."
        return {
            "answer": fallback,
            "generation_time": round(time.time() - t0, 2),
            "model_ready": True,
            "fallback": True,
            "error": str(e),
        }
