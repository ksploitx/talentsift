#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranking
Author: Khushneet Singh (team: ksploitx)

Phase 5 runtime: Loads cached artifacts, computes features, retrieves dense/BM25 scores,
runs honeypot checks, and computes the final normalized composite score.
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

def main():
    parser = argparse.ArgumentParser(description="Rank candidates pipeline (Phase 5).")
    parser.add_argument("--candidates", default="data/sample_candidates.json", help="Path to candidates JSON")
    parser.add_argument("--jd", default="jd_requirements.yaml", help="Path to JD text/YAML")
    parser.add_argument("--output", default="ranked_candidates.csv", help="Output CSV path")
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
        
    print("Phase 3: Hybrid Retrieval...")
    # Retrieve top_k for all candidates to get everyone's score
    retrieval_results = retrieve(jd_text, top_k=len(candidates))
    retrieval_scores = {r["candidate_id"]: r["rrf_score"] for r in retrieval_results}

    print("Phase 2 & 4: Feature Extraction and Honeypot Detection...")
    candidates_features = []
    honeypot_scores = {}
    
    for cand in candidates:
        cid = cand.get("candidate_id", "unknown")
        
        # Extract structured features
        feats = extract_features(cand)
        candidates_features.append((cid, feats))
        
        # Detect honeypots
        hp_res = compute_honeypot_score(cand, feats)
        honeypot_scores[cid] = hp_res.honeypot_score

    print("Phase 5: Composite Weighted Scoring...")
    scored_population = compute_composite_scores(
        candidates_features=candidates_features,
        retrieval_scores=retrieval_scores,
        honeypot_scores=honeypot_scores
    )

    print("Sorting candidates...")
    scored_population.sort(key=lambda x: x["final_score"], reverse=True)

    print(f"Writing ranked table to {args.output}...")
    with open(args.output, "w", newline="") as f:
        fieldnames = [
            "candidate_id", "final_score", "composite_score_before_penalty",
            "norm_retrieval", "norm_structured_fit", "norm_behavioral", "honeypot_penalty"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_population)
        
    print("Done!")

if __name__ == "__main__":
    main()