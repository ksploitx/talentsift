"""MMR-based diversity re-ranking to avoid redundant shortlists.

Greedy Maximal Marginal Relevance: iteratively selects candidates that
balance composite score (relevance) against embedding similarity to
already-selected candidates (diversity).

    MMR(c) = lambda * score(c) - (1 - lambda) * max_sim(c, selected)

All tunable parameters (mmr_lambda, shortlist_size, diversity_pool_size)
come from config.yaml.  Embedding vectors are loaded from precomputed
artifacts — no embedding generation happens here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
EMBEDDINGS_PATH = ROOT / "precompute" / "embeddings.npy"
CANDIDATE_IDS_PATH = ROOT / "precompute" / "candidate_ids.json"

_cache: dict[str, Any] = {}


def _load_config() -> dict:
    if "config" not in _cache:
        with open(CONFIG_PATH, "r") as f:
            _cache["config"] = yaml.safe_load(f)
    return _cache["config"]


def _load_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load cached L2-normalized embedding matrix and candidate ID list."""
    if "embeddings" not in _cache:
        _cache["embeddings"] = np.load(EMBEDDINGS_PATH)
        with open(CANDIDATE_IDS_PATH, "r") as f:
            _cache["emb_candidate_ids"] = json.load(f)
    return _cache["embeddings"], _cache["emb_candidate_ids"]


def mmr_rerank(
    scored_candidates: list[dict[str, Any]],
    final_k: int | None = None,
    pool_size: int | None = None,
) -> list[dict[str, Any]]:
    """Select a diverse top-K from scored candidates using greedy MMR.

    Args:
        scored_candidates: List of dicts, each must contain "candidate_id"
            and "final_score". Expected to already be sorted by final_score
            descending.
        final_k: Number of candidates to select (default: config shortlist_size).
        pool_size: Pre-diversity pool size to consider (default: config
            diversity_pool_size, fallback 5 * final_k).

    Returns:
        List of final_k candidate dicts in MMR-selected order, with an
        added "mmr_rank" field (1-indexed).
    """
    cfg = _load_config()
    if final_k is None:
        final_k = cfg.get("shortlist_size", 100)
    if pool_size is None:
        pool_size = cfg.get("diversity_pool_size", final_k * 5)

    mmr_lambda = cfg.get("mmr_lambda", 0.5)

    # --- Build the candidate pool (top pool_size by composite score) ---
    pool = scored_candidates[:pool_size]
    if len(pool) <= final_k:
        # Pool too small for meaningful MMR; return as-is
        for i, c in enumerate(pool):
            c["mmr_rank"] = i + 1
        return pool

    # --- Load precomputed embeddings and build a cid → row-index map ---
    emb_matrix, emb_ids = _load_embeddings()
    id_to_idx = {cid: idx for idx, cid in enumerate(emb_ids)}

    # Collect embedding rows for candidates in the pool
    pool_cids = [c["candidate_id"] for c in pool]
    pool_scores = np.array([c["final_score"] for c in pool], dtype=np.float64)
    pool_embs = np.zeros((len(pool), emb_matrix.shape[1]), dtype=np.float32)

    for i, cid in enumerate(pool_cids):
        idx = id_to_idx.get(cid)
        if idx is not None:
            pool_embs[i] = emb_matrix[idx]

    # Min-max normalize pool scores to [0, 1] for fair lambda blending
    s_min, s_max = pool_scores.min(), pool_scores.max()
    if s_max > s_min:
        norm_scores = (pool_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.zeros_like(pool_scores)

    # --- Greedy MMR selection ---
    selected_indices: list[int] = []
    remaining = set(range(len(pool)))

    # Precompute pairwise similarity matrix (cosine; embeddings are L2-normed)
    # Shape: (pool_size, pool_size).  For 500 candidates this is tiny.
    sim_matrix = pool_embs @ pool_embs.T

    for _ in range(min(final_k, len(pool))):
        best_idx = -1
        best_mmr = -float("inf")

        for idx in remaining:
            relevance = norm_scores[idx]

            if selected_indices:
                # Max cosine similarity to any already-selected candidate
                max_sim = float(sim_matrix[idx, selected_indices].max())
            else:
                max_sim = 0.0

            mmr_val = mmr_lambda * relevance - (1.0 - mmr_lambda) * max_sim
            if mmr_val > best_mmr:
                best_mmr = mmr_val
                best_idx = idx

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    # --- Build output ---
    results = []
    for rank, idx in enumerate(selected_indices, start=1):
        entry = dict(pool[idx])  # shallow copy
        entry["mmr_rank"] = rank
        results.append(entry)

    return results
