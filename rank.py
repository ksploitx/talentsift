#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranking
Author: Khushneet Singh (team: ksploitx)

Final orchestrator: loads cached precompute/ artifacts, runs feature
extraction → hybrid retrieval → honeypot detection → composite scoring →
MMR diversity re-ranking → reasoning generation, and writes submission.csv.

Prerequisites (run once before this script):
    python precompute/build_bm25_index.py
    python precompute/build_embeddings.py
    python precompute/build_honeypot_model.py

This script is the FAST PATH — no index building, no embedding generation,
no model training, no network calls.
"""

import json
import csv
import argparse
import sys
import time
from pathlib import Path

from src.features import extract_features
from src.retrieval import retrieve
from src.honeypot import compute_honeypot_score
from src.score import compute_composite_scores
from src.diversity import mmr_rerank
from src.reasoning import generate_reasoning


def _load_jsonl(path: Path) -> list[dict]:
    """Load candidates from a JSON Lines file."""
    candidates = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def _load_json(path: Path) -> list[dict]:
    """Load candidates from a JSON array file."""
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates — fast-path runtime pipeline.",
    )
    parser.add_argument(
        "--candidates", default="data/candidates.jsonl",
        help="Path to candidates file (.jsonl or .json)",
    )
    parser.add_argument(
        "--jd", default="jd_requirements.yaml",
        help="Path to JD text/YAML",
    )
    parser.add_argument(
        "--out", default="submission.csv",
        help="Output submission CSV path",
    )
    args = parser.parse_args()

    jd_path = Path(args.jd)
    candidates_path = Path(args.candidates)

    if not jd_path.exists():
        print(f"Error: JD file {args.jd} not found.")
        sys.exit(1)

    if not candidates_path.exists():
        print(f"Error: Candidates file {args.candidates} not found.")
        sys.exit(1)

    t0 = time.time()

    # --- Load inputs ---
    print(f"[1/7] Loading JD from {args.jd}...")
    with open(args.jd, "r") as f:
        jd_text = f.read()

    print(f"[2/7] Loading candidates from {args.candidates}...")
    if candidates_path.suffix == ".jsonl":
        candidates = _load_jsonl(candidates_path)
    else:
        candidates = _load_json(candidates_path)
    print(f"       Loaded {len(candidates)} candidates.")

    # --- Phase 3: Hybrid Retrieval (loads cached BM25 + embeddings) ---
    print("[3/7] Hybrid retrieval (BM25 + dense, cached artifacts)...")
    t_ret = time.time()
    retrieval_results = retrieve(jd_text, top_k=len(candidates))
    retrieval_scores = {r["candidate_id"]: r["rrf_score"]
                        for r in retrieval_results}
    print(f"       Retrieval done in {time.time() - t_ret:.1f}s.")

    # --- Phase 2 & 4: Feature Extraction + Honeypot Detection ---
    print("[4/7] Feature extraction + honeypot detection...")
    t_feat = time.time()
    candidates_features: list[tuple[str, dict]] = []
    honeypot_scores: dict[str, float] = {}
    candidate_map: dict[str, dict] = {}

    for cand in candidates:
        cid = cand.get("candidate_id", "unknown")
        candidate_map[cid] = cand
        feats = extract_features(cand)
        candidates_features.append((cid, feats))
        hp_res = compute_honeypot_score(cand, feats)
        honeypot_scores[cid] = hp_res.honeypot_score
    print(f"       Features + honeypot done in {time.time() - t_feat:.1f}s.")

    # --- Phase 5: Composite Weighted Scoring ---
    print("[5/7] Composite weighted scoring...")
    scored_population = compute_composite_scores(
        candidates_features=candidates_features,
        retrieval_scores=retrieval_scores,
        honeypot_scores=honeypot_scores,
    )
    scored_population.sort(key=lambda x: x["final_score"], reverse=True)

    # --- Phase 6: MMR Diversity Re-ranking → top 100 ---
    print("[6/7] MMR diversity re-ranking → top 100...")
    t_mmr = time.time()
    top_100 = mmr_rerank(scored_population)
    print(f"       MMR selected {len(top_100)} candidates in {time.time() - t_mmr:.1f}s.")

    # Build lookup: cid → features dict for reasoning
    feat_lookup = {cid: feats for cid, feats in candidates_features}

    # --- Phase 7: Reasoning Generation ---
    print("[7/7] Generating reasoning strings...")
    rows = []
    for entry in top_100:
        cid = entry["candidate_id"]
        rank = entry["mmr_rank"]
        score = entry["final_score"]

        cand = candidate_map.get(cid, {})
        feats = feat_lookup.get(cid, {})
        # Augment features with honeypot_penalty for reasoning concerns
        feats["honeypot_penalty"] = honeypot_scores.get(cid, 0.0)

        reasoning = generate_reasoning(cand, feats, score_row=entry)

        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 4),
            "reasoning": reasoning,
        })

    # --- Write submission CSV ---
    print(f"Writing {len(rows)} candidates to {args.out}...")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s → {args.out}")


if __name__ == "__main__":
    main()