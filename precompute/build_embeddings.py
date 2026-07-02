"""Pre-compute dense embeddings for all candidate profiles.

Loads the embedding model specified in config.yaml, builds the same concatenated
text per candidate as build_bm25_index.py, embeds all candidates in batches,
and saves:
  - precompute/embeddings.npy  (N x D float32 matrix)
  - precompute/candidate_ids.json  (ordered list mapping row index → candidate_id)

This script can take as long as it needs — it runs offline.
"""

import json
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
CANDIDATES_PATH = ROOT / "data" / "candidates.jsonl"
EMBEDDINGS_PATH = ROOT / "precompute" / "embeddings.npy"
IDS_PATH = ROOT / "precompute" / "candidate_ids.json"

BATCH_SIZE = 256


def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _build_candidate_text(candidate: dict) -> str:
    """Concatenate summary + skill names + career descriptions.

    Identical logic to build_bm25_index.py so both indexes align.
    """
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


def main():
    cfg = _load_config()
    model_name = cfg.get("embedding_model_name", "BAAI/bge-small-en-v1.5")

    print(f"Loading model: {model_name} ...")
    model = SentenceTransformer(model_name)

    candidate_ids: list[str] = []
    texts: list[str] = []

    print(f"Loading candidates from {CANDIDATES_PATH} ...")
    with open(CANDIDATES_PATH, "r") as f:
        for i, line in enumerate(f):
            candidate = json.loads(line)
            candidate_ids.append(candidate["candidate_id"])
            texts.append(_build_candidate_text(candidate))

            if (i + 1) % 10_000 == 0:
                print(f"  Loaded {i + 1} candidates")

    print(f"Total candidates: {len(candidate_ids)}")
    print(f"Embedding {len(texts)} texts in batches of {BATCH_SIZE} ...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize so dot product = cosine sim
    )

    print(f"Embeddings shape: {embeddings.shape}")

    print(f"Saving embeddings to {EMBEDDINGS_PATH} ...")
    np.save(EMBEDDINGS_PATH, embeddings.astype(np.float32))

    print(f"Saving candidate IDs to {IDS_PATH} ...")
    with open(IDS_PATH, "w") as f:
        json.dump(candidate_ids, f)

    size_mb = EMBEDDINGS_PATH.stat().st_size / (1024 * 1024)
    print(f"Done. Embeddings file: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
