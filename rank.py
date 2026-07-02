#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranking
Author: Khushneet Singh (team: ksploitx)

Runtime pipeline: Loads cached artifacts, computes features, retrieves
dense/BM25 scores, runs honeypot checks, computes composite scores,
applies MMR diversity re-ranking, and outputs the final ranked CSV.
"""

import json
import csv
import argparse
import sys
from pathlib import Path

from src.features import extract_features
from src.retrieval import retrieve
from src.honeypot import compute_honeypot_score
from src.score import compute_composite_scores
from src.diversity import mmr_rerank


def main():
    parser = argparse.ArgumentParser(description="Rank candidates pipeline.")
    parser.add_argument("--candidates", default="data/sample_candidates.json",
                        help="Path to candidates JSON")
    parser.add_argument("--jd", default="jd_requirements.yaml",
                        help="Path to JD text/YAML")
    parser.add_argument("--output", default="ranked_candidates.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    jd_path = Path(args.jd)
    candidates_path = Path(args.candidates)

    if not jd_path.exists():
        print(f"Error: JD file {args.jd} not found.")
        sys.exit(1)

    if not candidates_path.exists():
        print(f"Error: Candidates file {args.candidates} not found.")
        sys.exit(1)

    print(f"Loading job description from {args.jd}...")
    with open(args.jd, "r") as f:
        jd_text = f.read()

    print(f"Loading candidates from {args.candidates}...")
    with open(args.candidates, "r") as f:
        candidates = json.load(f)

    # Phase 3: Hybrid Retrieval
    print("Phase 3: Hybrid Retrieval...")
    retrieval_results = retrieve(jd_text, top_k=len(candidates))
    retrieval_scores = {r["candidate_id"]: r["rrf_score"]
                        for r in retrieval_results}

    # Phase 2 & 4: Feature Extraction + Honeypot Detection
    print("Phase 2 & 4: Feature Extraction and Honeypot Detection...")
    candidates_features = []
    honeypot_scores = {}

    for cand in candidates:
        cid = cand.get("candidate_id", "unknown")
        feats = extract_features(cand)
        candidates_features.append((cid, feats))

        hp_res = compute_honeypot_score(cand, feats)
        honeypot_scores[cid] = hp_res.honeypot_score

    # Phase 5: Composite Weighted Scoring
    print("Phase 5: Composite Weighted Scoring...")
    scored_population = compute_composite_scores(
        candidates_features=candidates_features,
        retrieval_scores=retrieval_scores,
        honeypot_scores=honeypot_scores,
    )
    scored_population.sort(key=lambda x: x["final_score"], reverse=True)

    # Phase 6: MMR Diversity Re-ranking → top 100
    print("Phase 6: MMR Diversity Re-ranking...")
    top_100 = mmr_rerank(scored_population)
    print(f"  Selected {len(top_100)} candidates after MMR.")

    # Write output
    print(f"Writing ranked table to {args.output}...")
    fieldnames = [
        "mmr_rank", "candidate_id", "final_score",
        "composite_score_before_penalty", "norm_retrieval",
        "norm_structured_fit", "norm_behavioral", "honeypot_penalty",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(top_100)

    print("Done!")


if __name__ == "__main__":
    main()