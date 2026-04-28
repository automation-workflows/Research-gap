"""
research_gap/prior_work/embeddings.py – Local embedding-based similarity ranking.

Uses the ``sentence-transformers`` library with ``all-MiniLM-L6-v2`` by default
to compute cosine similarity between an idea/query text and a list of candidate
paper abstracts, then assigns a risk label (low/medium/high).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Cosine-similarity thresholds for risk labelling
_HIGH_RISK_THRESHOLD = 0.75
_MEDIUM_RISK_THRESHOLD = 0.55

# Lazy-loaded model cache
_model_cache: dict[str, Any] = {}


def _get_model(model_name: str = _DEFAULT_MODEL) -> Any:
    """Load (and cache) a SentenceTransformer model."""
    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding-based ranking. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        logger.info("Loading embedding model %s …", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _risk_label(similarity: float) -> str:
    if similarity >= _HIGH_RISK_THRESHOLD:
        return "high"
    if similarity >= _MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def rank_by_similarity(
    idea: str,
    candidates: list[dict[str, Any]],
    model_name: str = _DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Rank candidate papers by cosine similarity to *idea*.

    Each returned dict is a copy of the candidate with two extra fields:
        ``similarity`` (float): cosine similarity score in [0, 1]
        ``risk`` (str): "low", "medium", or "high"

    Parameters
    ----------
    idea:
        The research idea / gap text to compare against.
    candidates:
        List of candidate paper dicts (must have at least an ``abstract`` field).
    model_name:
        Sentence-transformer model to use.

    Returns
    -------
    list[dict]
        Candidates sorted by descending similarity.
    """
    if not candidates:
        return []

    model = _get_model(model_name)

    # Build texts: use abstract, fall back to title
    texts = [
        (c.get("abstract") or c.get("title") or "").strip()
        for c in candidates
    ]

    # Encode all at once for efficiency
    all_texts = [idea] + texts
    embeddings = model.encode(all_texts, show_progress_bar=False, convert_to_numpy=True)

    idea_emb: np.ndarray = embeddings[0]
    candidate_embs: np.ndarray = embeddings[1:]

    ranked: list[dict[str, Any]] = []
    for i, (candidate, emb) in enumerate(zip(candidates, candidate_embs)):
        sim = _cosine_similarity(idea_emb, emb)
        entry = dict(candidate)
        entry["similarity"] = round(sim, 4)
        entry["risk"] = _risk_label(sim)
        ranked.append(entry)

    ranked.sort(key=lambda x: x["similarity"], reverse=True)
    return ranked
