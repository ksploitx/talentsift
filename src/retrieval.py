"""Hybrid retrieval: BM25 + dense vector search with Reciprocal Rank Fusion.

At ranking time, loads cached BM25 index and dense embeddings — never rebuilds
them. Embeds the JD text once, computes per-candidate BM25 and cosine scores,
then fuses rankings with RRF. The fusion constant k comes from config.yaml.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
BM25_INDEX_PATH = ROOT / "precompute" / "bm25_index.pkl"
EMBEDDINGS_PATH = ROOT / "precompute" / "embeddings.npy"
CANDIDATE_IDS_PATH = ROOT / "precompute" / "candidate_ids.json"

# Module-level cache — loaded once per process
_cache: dict[str, Any] = {}


def _load_config() -> dict:
    if "config" not in _cache:
        with open(CONFIG_PATH, "r") as f:
            _cache["config"] = yaml.safe_load(f)
    return _cache["config"]


def _load_bm25():
    """Load cached BM25 index and candidate_ids from pickle."""
    if "bm25" not in _cache:
        with open(BM25_INDEX_PATH, "rb") as f:
            payload = pickle.load(f)
        _cache["bm25"] = payload["bm25"]
        _cache["bm25_candidate_ids"] = payload["candidate_ids"]
    return _cache["bm25"], _cache["bm25_candidate_ids"]


def _load_embeddings():
    """Load cached embedding matrix and candidate ID index."""
    if "embeddings" not in _cache:
        _cache["embeddings"] = np.load(EMBEDDINGS_PATH)
        with open(CANDIDATE_IDS_PATH, "r") as f:
            _cache["emb_candidate_ids"] = json.load(f)
    return _cache["embeddings"], _cache["emb_candidate_ids"]


def _load_model() -> SentenceTransformer:
    """Load the sentence-transformer model (for JD embedding only)."""
    if "model" not in _cache:
        cfg = _load_config()
        model_name = cfg.get("embedding_model_name", "BAAI/bge-small-en-v1.5")
        _cache["model"] = SentenceTransformer(model_name)
    return _cache["model"]


def _tokenize(text: str) -> list[str]:
    """Same tokenizer as build_bm25_index.py — simple whitespace split."""
    return text.lower().split()


def retrieve(jd_text: str, top_k: int = 100) -> list[dict]:
    """Retrieve top-k candidates using hybrid BM25 + dense retrieval with RRF.

    Args:
        jd_text: The full job description text to match against.
        top_k: Number of candidates to return.

    Returns:
        List of dicts sorted by fused RRF score (descending), each containing:
        {"candidate_id", "rrf_score", "bm25_rank", "dense_rank"}
    """
    cfg = _load_config()
    rrf_k = cfg.get("rrf_k", 60)

    # --- BM25 scoring ---
    bm25, bm25_ids = _load_bm25()
    query_tokens = _tokenize(jd_text)
    bm25_scores = bm25.get_scores(query_tokens)

    # Rank by BM25 score (descending) — rank 1 = best
    bm25_rank_order = np.argsort(-bm25_scores)
    bm25_ranks = np.empty_like(bm25_rank_order)
    bm25_ranks[bm25_rank_order] = np.arange(1, len(bm25_rank_order) + 1)

    # --- Dense scoring ---
    embeddings, emb_ids = _load_embeddings()
    model = _load_model()

    jd_embedding = model.encode(
        [jd_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]  # shape: (D,)

    # Cosine similarity = dot product (embeddings are L2-normalized)
    dense_scores = embeddings @ jd_embedding

    # Rank by dense score (descending)
    dense_rank_order = np.argsort(-dense_scores)
    dense_ranks = np.empty_like(dense_rank_order)
    dense_ranks[dense_rank_order] = np.arange(1, len(dense_rank_order) + 1)

    # --- Reciprocal Rank Fusion ---
    # RRF(c) = 1/(k + rank_bm25(c)) + 1/(k + rank_dense(c))
    rrf_scores = 1.0 / (rrf_k + bm25_ranks) + 1.0 / (rrf_k + dense_ranks)

    # Select top-k by RRF score
    top_indices = np.argsort(-rrf_scores)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "candidate_id": bm25_ids[idx],
            "rrf_score": float(rrf_scores[idx]),
            "bm25_rank": int(bm25_ranks[idx]),
            "dense_rank": int(dense_ranks[idx]),
        })

    return results
