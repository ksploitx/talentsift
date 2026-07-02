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

## 2026-07-02 — Phase 2, Subphases 2.1–2.4: Feature extraction

**Files touched:**
- Inspected: `data/candidates.jsonl`, `data/candidate_schema.json`, `data/sample_candidates.json`
- Modified: `config.yaml` (added feature extraction thresholds section)
- Rewritten: `src/features.py` (from placeholder to full implementation)
- Rewritten: `tests/test_features.py` (from placeholder to full test suite)
- Modified: `CHANGELOG.md`

**Description:**
Loaded and inspected the first records of `data/candidates.jsonl` alongside `candidate_schema.json` and `sample_candidates.json` — confirmed all field names, nesting, and enum values match the schema exactly with zero discrepancies. Added a comprehensive feature-extraction config block to `config.yaml` covering company size scores, consulting firms list, location tier mapping, notice period thresholds, proficiency weights, education tier scores, experience range, min tenure threshold, and reference date. Implemented `extract_features(candidate) -> dict` in `src/features.py` returning 28 flat numeric/categorical features: `years_of_experience`, `experience_fit_score`, `is_product_company`, `company_size_score`, `consulting_firm_flag`, `consulting_only_career`, `avg_tenure_months`, `short_tenure_ratio`, `career_entry_count`, `location_tier`, `notice_period_days`, `notice_period_bucket`, `recruiter_response_rate`, `days_since_active`, `open_to_work`, `platform_engagement_score`, `verification_score`, `skill_count_raw`, `skill_count_by_proficiency`, `certification_count`, `github_activity_score`, `education_tier_score`, `expected_salary_min_lpa`, `expected_salary_max_lpa`, `preferred_work_mode`, `willing_to_relocate`, `avg_response_time_hours`, and `offer_acceptance_rate`. Wrote `tests/test_features.py` with 30 assertions across 3 real candidates (CAND_0000001, CAND_0000002, CAND_0000003) plus edge-case/structural tests — hand-verified expected outputs against config thresholds. Tests were not executed per project rules.

**Deviations from spec:**
- Added several features beyond the explicitly listed set (e.g., `experience_fit_score`, `consulting_only_career`, `avg_tenure_months`, `short_tenure_ratio`, `platform_engagement_score`, `verification_score`, `education_tier_score`, `offer_acceptance_rate`) because they are directly implied by `jd_requirements.yaml` disqualifiers and behavioral priorities. These enrich the scoring signal available in later phases.
- `skill_count_by_proficiency` uses weighted proficiency scores (config-driven) rather than separate counts per proficiency level — this is more useful as a single scalar feature for scoring.

## 2026-07-02 — Phase 3, Subphases 3.1–3.4: Hybrid retrieval (BM25 + embeddings)

**Files touched:**
- Rewritten: `precompute/build_bm25_index.py` (from placeholder to full implementation)
- Rewritten: `precompute/build_embeddings.py` (from placeholder to full implementation)
- Rewritten: `src/retrieval.py` (from placeholder to full implementation)
- Modified: `config.yaml` (added `rrf_k: 60`)
- Modified: `requirements.txt` (added `rank_bm25`, `sentence-transformers`, `numpy`, `pyyaml`)
- Modified: `CHANGELOG.md`

**Description:**
Implemented the full hybrid retrieval pipeline. `build_bm25_index.py` loads all 100k candidates from `candidates.jsonl`, concatenates each candidate's summary + skill names + career descriptions, tokenizes with whitespace splitting, builds a `BM25Okapi` index with k1/b from config.yaml, and serializes to `bm25_index.pkl`. `build_embeddings.py` loads the `BAAI/bge-small-en-v1.5` model from config, builds the same concatenated text, embeds all candidates in batches of 256 with L2-normalization (so dot product = cosine similarity), and saves to `embeddings.npy` + `candidate_ids.json`. `src/retrieval.py` loads all cached artifacts at runtime (never rebuilds), embeds the JD text once, computes per-candidate BM25 scores and cosine similarities, ranks independently, and fuses with Reciprocal Rank Fusion: `RRF(c) = 1/(k + rank_bm25) + 1/(k + rank_dense)` where k=60 from config. Returns top-k candidates with their fused score and component ranks.

**Deviations from spec:**
- None. All parameters live in config.yaml. No LLM API calls. No index-building in src/.
