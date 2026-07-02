# Changelog

## 2026-07-02 — Phase 1, Subphases 1.1–1.4: Repo scaffold + JD rubric

**Files touched:**
- Created: `data/`, `docs/`, `precompute/`, `src/`, `tests/`, `sandbox/`
- Moved to `data/`: `candidate_schema.json`, `candidates.jsonl`, `sample_candidates.json`, `sample_submission.csv`
- Moved to `docs/`: `job_description.docx`, `redrob_signals_doc.docx`, `submission_spec.docx`, `README.docx`
- Created placeholders: `precompute/build_bm25_index.py`, `precompute/build_embeddings.py`, `src/features.py`, `src/retrieval.py`, `src/honeypot.py`, `src/score.py`, `src/diversity.py`, `src/reasoning.py`, `src/rank.py`, `tests/test_features.py`, `tests/test_honeypot.py`, `tests/test_validator_compliance.py`, `sandbox/app.py`
- Created: `jd_requirements.yaml`, `config.yaml`, `requirements.txt`, `submission_metadata.yaml`, `CHANGELOG.md`

**Description:**
Scaffolded the project at repo root, organizing existing hackathon bundle files into `data/` (schemas, candidate data) and `docs/` (reference docx files). Created placeholder modules across `src/`, `precompute/`, `tests/`, and `sandbox/` with one-line docstrings only. Parsed the full job description into `jd_requirements.yaml` as a structured rubric covering must-have skills, nice-to-haves, disqualifiers, experience range, location tiers, company-type preferences, notice-period constraints, and behavioral priorities — each with reasoning quoted or closely paraphrased from the JD. Initialized `config.yaml` with all tunable keys (composite weights at 0, shortlist size, honeypot threshold, BM25 params, embedding model name, MMR lambda) and one comment per key.

**Deviations from spec:**
- Added a `docs/` directory (not in original spec) to house the reference `.docx` files and keep root clean.
- Existing `rank.py` kept at root as the v1 monolithic entry point; the modular replacement lives at `src/rank.py`.
