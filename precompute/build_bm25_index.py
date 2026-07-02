"""Build a BM25 index from candidate profiles for offline retrieval.

Reads all candidates from data/candidates.jsonl, concatenates each candidate's
summary + skill names + career descriptions into a single document, tokenizes,
builds a BM25Okapi index with k1/b from config.yaml, and serializes the index
plus candidate_id ordering to precompute/bm25_index.pkl.

This script can take as long as it needs — it runs offline.
"""

import json
import pickle
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
CANDIDATES_PATH = ROOT / "data" / "candidates.jsonl"
OUTPUT_PATH = ROOT / "precompute" / "bm25_index.pkl"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _build_candidate_text(candidate: dict) -> str:
    """Concatenate summary + skill names + career descriptions."""
    parts = []

    summary = candidate.get("profile", {}).get("summary", "")
    if summary:
        parts.append(summary)

    skills = candidate.get("skills", [])
    skill_names = " ".join(s.get("name", "") for s in skills)
    if skill_names.strip():
        parts.append(skill_names)

    career = candidate.get("career_history", [])
    for job in career:
        desc = job.get("description", "")
        if desc:
            parts.append(desc)

    return " ".join(parts)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization with lowercasing."""
    return text.lower().split()


def main():
    cfg = _load_config()
    bm25_params = cfg.get("bm25_params", {})
    k1 = bm25_params.get("k1", 1.5)
    b = bm25_params.get("b", 0.75)

    candidate_ids: list[str] = []
    tokenized_corpus: list[list[str]] = []

    print(f"Loading candidates from {CANDIDATES_PATH} ...")
    with open(CANDIDATES_PATH, "r") as f:
        for i, line in enumerate(f):
            candidate = json.loads(line)
            cid = candidate["candidate_id"]
            text = _build_candidate_text(candidate)
            tokens = _tokenize(text)

            candidate_ids.append(cid)
            tokenized_corpus.append(tokens)

            if (i + 1) % 10_000 == 0:
                print(f"  Processed {i + 1} candidates")

    print(f"Total candidates: {len(candidate_ids)}")
    print(f"Building BM25Okapi index (k1={k1}, b={b}) ...")

    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    payload = {
        "bm25": bm25,
        "candidate_ids": candidate_ids,
    }

    print(f"Serializing to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Done. Index size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
