"""Tests for the feature extraction module.

Uses candidates from data/sample_candidates.json as known inputs.
Expected outputs are hand-computed from the schema and config.yaml thresholds.
DO NOT RUN THIS FILE — it is a deliverable only.
"""

import json
import os
import sys

import pytest
import yaml

# Ensure the repo root is on sys.path so we can import src.features.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.features import extract_features, reset_config_cache


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Reset the config cache before each test."""
    reset_config_cache()
    yield
    reset_config_cache()


@pytest.fixture
def sample_candidates():
    """Load all candidates from data/sample_candidates.json."""
    path = os.path.join(ROOT_DIR, "data", "sample_candidates.json")
    with open(path, "r") as f:
        return json.load(f)


@pytest.fixture
def config():
    """Load config.yaml."""
    path = os.path.join(ROOT_DIR, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def cand_1(sample_candidates):
    """CAND_0000001 — Ira Vora, Backend Engineer at Mindtree, Toronto, 6.9 yoe."""
    return next(c for c in sample_candidates if c["candidate_id"] == "CAND_0000001")


@pytest.fixture
def cand_2(sample_candidates):
    """CAND_0000002 — Saanvi Sethi, Operations Manager at Wipro, Chennai, 12.5 yoe."""
    return next(c for c in sample_candidates if c["candidate_id"] == "CAND_0000002")


@pytest.fixture
def cand_3(sample_candidates):
    """CAND_0000003 — Yash Agarwal, Customer Support at TCS, Austin, 1.1 yoe."""
    return next(c for c in sample_candidates if c["candidate_id"] == "CAND_0000003")


# ============================================================================
# Tests for CAND_0000001 — Ira Vora
# ============================================================================

class TestCand1Features:
    """CAND_0000001: Backend Engineer, Mindtree, Toronto, Canada, 6.9 yoe.

    Career: Mindtree (consulting) + Dunder Mifflin (not consulting).
    So: is_product_company=True, consulting_firm_flag=True (current), consulting_only_career=False.
    """

    def test_years_of_experience(self, cand_1):
        feats = extract_features(cand_1)
        assert feats["years_of_experience"] == 6.9

    def test_experience_fit_score_within_range(self, cand_1):
        """6.9 is within [5, 9] → 1.0."""
        feats = extract_features(cand_1)
        assert feats["experience_fit_score"] == 1.0

    def test_is_product_company_true(self, cand_1):
        """Has Dunder Mifflin in career — not a consulting firm."""
        feats = extract_features(cand_1)
        assert feats["is_product_company"] is True

    def test_company_size_score(self, cand_1):
        """current_company_size '10001+' → 8."""
        feats = extract_features(cand_1)
        assert feats["company_size_score"] == 8

    def test_consulting_firm_flag(self, cand_1):
        """Current company is Mindtree → True."""
        feats = extract_features(cand_1)
        assert feats["consulting_firm_flag"] is True

    def test_consulting_only_career_false(self, cand_1):
        """Dunder Mifflin is not consulting → not consulting-only."""
        feats = extract_features(cand_1)
        assert feats["consulting_only_career"] is False

    def test_location_tier_international(self, cand_1):
        """Toronto, Canada → tier 4 (international)."""
        feats = extract_features(cand_1)
        assert feats["location_tier"] == 4

    def test_notice_period(self, cand_1):
        """60 days → bucket 'acceptable'."""
        feats = extract_features(cand_1)
        assert feats["notice_period_days"] == 60
        assert feats["notice_period_bucket"] == "acceptable"

    def test_recruiter_response_rate(self, cand_1):
        feats = extract_features(cand_1)
        assert feats["recruiter_response_rate"] == 0.34

    def test_days_since_active(self, cand_1):
        """last_active_date 2026-05-20, reference_date 2026-06-30 → 41 days."""
        feats = extract_features(cand_1)
        assert feats["days_since_active"] == 41

    def test_skill_count_raw(self, cand_1):
        """17 skills in the profile."""
        feats = extract_features(cand_1)
        assert feats["skill_count_raw"] == 17

    def test_skill_count_by_proficiency(self, cand_1):
        """3 beginner (0.75) + 5 intermediate (2.5) + 6 advanced (6.0) + 0 expert → 12.25.
        Wait — let me recount from the data:
        beginner: AWS, Flask, GCP → 3 × 0.25 = 0.75
        intermediate: Tailwind, W&B, LoRA, Apache Beam, BentoML, Statistical Modeling → 6 × 0.5 = 3.0
        advanced: NLP, Image Classification, Fine-tuning LLMs, Speech Recognition, TTS, Milvus, GANs → 7 × 1.0 = 7.0
        expert: 0
        missing: Photoshop is intermediate → already counted
        Total skills listed = 17. Let me recount:
        1. Tailwind - intermediate
        2. NLP - advanced
        3. Image Classification - advanced
        4. Fine-tuning LLMs - advanced
        5. W&B - intermediate
        6. Speech Recognition - advanced
        7. Photoshop - intermediate
        8. TTS - advanced
        9. LoRA - intermediate
        10. Apache Beam - intermediate
        11. AWS - beginner
        12. Flask - beginner
        13. BentoML - intermediate
        14. Milvus - advanced
        15. GANs - advanced
        16. Statistical Modeling - intermediate
        17. GCP - beginner
        beginner: 3 × 0.25 = 0.75
        intermediate: 7 × 0.5 = 3.5
        advanced: 7 × 1.0 = 7.0
        Total = 11.25
        """
        feats = extract_features(cand_1)
        assert feats["skill_count_by_proficiency"] == 11.25

    def test_github_activity_score(self, cand_1):
        """9.2 → 9.2 (positive, passed through)."""
        feats = extract_features(cand_1)
        assert feats["github_activity_score"] == 9.2

    def test_education_tier_score(self, cand_1):
        """LPU, tier_3 → 2."""
        feats = extract_features(cand_1)
        assert feats["education_tier_score"] == 2

    def test_open_to_work(self, cand_1):
        feats = extract_features(cand_1)
        assert feats["open_to_work"] is True

    def test_avg_tenure_months(self, cand_1):
        """Mindtree 27 + Dunder Mifflin 55 → avg 41.0."""
        feats = extract_features(cand_1)
        assert feats["avg_tenure_months"] == 41.0

    def test_short_tenure_ratio(self, cand_1):
        """Neither 27 nor 55 is < 18 → 0.0."""
        feats = extract_features(cand_1)
        assert feats["short_tenure_ratio"] == 0.0

    def test_salary_range(self, cand_1):
        feats = extract_features(cand_1)
        assert feats["expected_salary_min_lpa"] == 18.7
        assert feats["expected_salary_max_lpa"] == 36.1

    def test_verification_score(self, cand_1):
        """verified_email=True, verified_phone=True, linkedin_connected=False → 2."""
        feats = extract_features(cand_1)
        assert feats["verification_score"] == 2


# ============================================================================
# Tests for CAND_0000002 — Saanvi Sethi
# ============================================================================

class TestCand2Features:
    """CAND_0000002: Operations Manager, Wipro, Chennai, India, 12.5 yoe.

    Career: Wipro (consulting) × 2 + Acme Corp (not consulting) + Dunder Mifflin (not consulting).
    consulting_firm_flag=True (current is Wipro), consulting_only_career=False.
    12.5 yoe is above the [5,9] range.
    """

    def test_experience_fit_above_range(self, cand_2):
        """12.5 yoe, max=9 → overshoot=3.5, decay = 1 - 3.5/9 ≈ 0.61."""
        feats = extract_features(cand_2)
        assert feats["experience_fit_score"] == 0.61

    def test_location_tier_india(self, cand_2):
        """Chennai, Tamil Nadu → city 'chennai' → tier 3."""
        feats = extract_features(cand_2)
        assert feats["location_tier"] == 3

    def test_github_no_account(self, cand_2):
        """github_activity_score = -1 → 0.0."""
        feats = extract_features(cand_2)
        assert feats["github_activity_score"] == 0.0

    def test_consulting_only_false(self, cand_2):
        """Has Acme Corp and Dunder Mifflin → not consulting-only."""
        feats = extract_features(cand_2)
        assert feats["consulting_only_career"] is False

    def test_days_since_active_stale(self, cand_2):
        """last_active_date 2025-11-12, reference 2026-06-30 → 230 days."""
        feats = extract_features(cand_2)
        assert feats["days_since_active"] == 230

    def test_offer_acceptance_no_history(self, cand_2):
        """offer_acceptance_rate = -1 → clamped to 0.0."""
        feats = extract_features(cand_2)
        assert feats["offer_acceptance_rate"] == 0.0


# ============================================================================
# Tests for CAND_0000003 — Yash Agarwal
# ============================================================================

class TestCand3Features:
    """CAND_0000003: Customer Support, TCS, Austin, USA, 1.1 yoe.

    Single career entry: TCS (consulting). consulting_only_career=True.
    1.1 yoe is far below [5,9] range.
    """

    def test_experience_fit_below_range(self, cand_3):
        """1.1 yoe, min=5 → 1.1/5 = 0.22."""
        feats = extract_features(cand_3)
        assert feats["experience_fit_score"] == 0.22

    def test_consulting_only_true(self, cand_3):
        """Only TCS in career → consulting-only."""
        feats = extract_features(cand_3)
        assert feats["consulting_only_career"] is True

    def test_is_product_company_false(self, cand_3):
        """Only TCS → no product company."""
        feats = extract_features(cand_3)
        assert feats["is_product_company"] is False

    def test_location_international(self, cand_3):
        """Austin, USA → tier 4."""
        feats = extract_features(cand_3)
        assert feats["location_tier"] == 4

    def test_notice_period_very_long(self, cand_3):
        """150 days → 'very_long'."""
        feats = extract_features(cand_3)
        assert feats["notice_period_days"] == 150
        assert feats["notice_period_bucket"] == "very_long"

    def test_career_single_entry(self, cand_3):
        feats = extract_features(cand_3)
        assert feats["career_entry_count"] == 1

    def test_short_tenure_ratio_single_short(self, cand_3):
        """TCS 13 months < 18 → ratio 1.0."""
        feats = extract_features(cand_3)
        assert feats["short_tenure_ratio"] == 1.0


# ============================================================================
# Edge-case / synthetic tests
# ============================================================================

class TestEdgeCases:
    """Edge cases with minimal or missing data."""

    def test_empty_candidate(self):
        """Candidate with all fields empty/missing shouldn't crash."""
        empty = {
            "candidate_id": "CAND_0000000",
            "profile": {},
            "career_history": [],
            "education": [],
            "skills": [],
            "certifications": [],
            "languages": [],
            "redrob_signals": {},
        }
        feats = extract_features(empty)
        assert feats["years_of_experience"] == 0
        assert feats["skill_count_raw"] == 0
        assert feats["avg_tenure_months"] == 0.0
        assert feats["education_tier_score"] == 0
        assert feats["days_since_active"] == 999  # Sentinel for missing

    def test_feature_dict_is_flat(self, sample_candidates):
        """All values in the output dict should be scalar (no nested dicts/lists)."""
        feats = extract_features(sample_candidates[0])
        for key, val in feats.items():
            assert not isinstance(val, (dict, list, set)), (
                f"Feature '{key}' is not scalar: {type(val)}"
            )

    def test_all_expected_keys_present(self, sample_candidates):
        """Ensure all documented feature keys are present."""
        expected_keys = {
            "years_of_experience",
            "experience_fit_score",
            "is_product_company",
            "company_size_score",
            "consulting_firm_flag",
            "consulting_only_career",
            "avg_tenure_months",
            "short_tenure_ratio",
            "career_entry_count",
            "location_tier",
            "notice_period_days",
            "notice_period_bucket",
            "recruiter_response_rate",
            "days_since_active",
            "open_to_work",
            "platform_engagement_score",
            "verification_score",
            "skill_count_raw",
            "skill_count_by_proficiency",
            "certification_count",
            "github_activity_score",
            "education_tier_score",
            "expected_salary_min_lpa",
            "expected_salary_max_lpa",
            "preferred_work_mode",
            "willing_to_relocate",
            "avg_response_time_hours",
            "offer_acceptance_rate",
        }
        feats = extract_features(sample_candidates[0])
        assert set(feats.keys()) == expected_keys
