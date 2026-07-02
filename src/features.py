"""
Stage 1 of the ranking pipeline: candidate JSON -> flat numeric/categorical feature dict.

Design principle: every feature here is traceable to either
  (a) a specific line in job_description.md (see config/jd_requirements.yaml), or
  (b) a schema field used for honeypot/behavioral reasoning.
No feature is "just because it seemed useful" -- if you can't point to why it's here,
it doesn't belong here, because you'll need to defend every one of these at Stage 5.

Usage:
    from features import FeatureExtractor
    fx = FeatureExtractor("config/jd_requirements.yaml")
    feature_row = fx.extract(candidate_dict)
"""

from __future__ import annotations

import re
import json
import yaml
from datetime import date, datetime
from dataclasses import dataclass, field, asdict
from typing import Any


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _keyword_hits(text: str, keywords: list[str]) -> int:
    """Case-insensitive substring count, word-boundary-aware where possible."""
    text_low = text.lower()
    hits = 0
    for kw in keywords:
        kw_low = kw.lower()
        if re.search(r"(?<![a-z0-9])" + re.escape(kw_low) + r"(?![a-z0-9])", text_low):
            hits += 1
    return hits


def _combined_text(candidate: dict) -> str:
    """All free text on the profile, concatenated, for keyword/role-relevance matching."""
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
        candidate["profile"].get("current_title", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))
    return " \n ".join(p for p in parts if p)


_PROFICIENCY_TO_EXPECTED_SCORE = {
    "beginner": 25.0,
    "intermediate": 50.0,
    "advanced": 70.0,
    "expert": 85.0,
}

TODAY = date(2026, 7, 2)  # frozen "as of" date so features are reproducible run-to-run


# ----------------------------------------------------------------------------
# Feature container
# ----------------------------------------------------------------------------

@dataclass
class CandidateFeatures:
    candidate_id: str

    # --- experience / seniority ---
    years_of_experience: float = 0.0
    years_in_relevant_roles: float = 0.0
    num_jobs: int = 0
    avg_tenure_months: float = 0.0
    is_currently_non_ic_senior: bool = False
    months_since_last_ic_role: float | None = None  # None if never non-IC

    # --- skill match ---
    must_have_hits: dict = field(default_factory=dict)      # category -> bool
    must_have_score: float = 0.0                             # 0-1, fraction of categories hit
    nice_to_have_score: float = 0.0                           # 0-1
    role_relevance_score: float = 0.0                         # 0-1, keyword density capped
    skill_credibility_gap: float = 0.0                        # 0-1, higher = bigger claim/assessment gap

    # --- disqualifiers (JD "things we explicitly do NOT want") ---
    disq_consulting_only: bool = False
    disq_cv_speech_robotics_only: bool = False
    disq_title_chaser: bool = False
    disq_senior_no_recent_code: bool = False
    disq_pure_research_no_prod: bool = False
    disq_recent_langchain_only: bool = False
    disqualifier_count: int = 0

    # --- education ---
    education_tier_score: float = 0.0  # tier_1=1.0 ... unknown=0.0

    # --- logistics ---
    location_in_target_list: bool = False
    location_is_preferred: bool = False
    willing_to_relocate: bool = False
    notice_period_days: int = 999
    notice_period_ok: bool = False

    # --- behavioral / availability (redrob_signals) ---
    recruiter_response_rate: float = 0.0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = 0.0   # -1 sentinel remapped to neutral 0.5 upstream
    months_since_last_active: float = 999.0
    open_to_work_flag: bool = False
    behavioral_availability_score: float = 0.0  # 0-1 composite

    # --- honeypot heuristics (raw signals, final decision made in honeypot.py) ---
    hp_experience_duration_mismatch: bool = False
    hp_overlapping_employment: bool = False
    hp_expert_with_near_zero_duration: bool = False
    hp_duration_vs_dates_mismatch: bool = False
    hp_experience_vs_education_implausible: bool = False
    honeypot_flag_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------
# Extractor
# ----------------------------------------------------------------------------

class FeatureExtractor:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    # -- public API -----------------------------------------------------

    def extract(self, candidate: dict) -> CandidateFeatures:
        cf = CandidateFeatures(candidate_id=candidate["candidate_id"])
        text_blob = _combined_text(candidate)

        self._experience_features(candidate, cf)
        self._skill_match_features(candidate, cf, text_blob)
        self._disqualifier_features(candidate, cf, text_blob)
        self._education_features(candidate, cf)
        self._logistics_features(candidate, cf)
        self._behavioral_features(candidate, cf)
        self._honeypot_heuristics(candidate, cf)

        cf.disqualifier_count = sum([
            cf.disq_consulting_only,
            cf.disq_cv_speech_robotics_only,
            cf.disq_title_chaser,
            cf.disq_senior_no_recent_code,
            cf.disq_pure_research_no_prod,
            cf.disq_recent_langchain_only,
        ])
        cf.honeypot_flag_count = sum([
            cf.hp_experience_duration_mismatch,
            cf.hp_overlapping_employment,
            cf.hp_expert_with_near_zero_duration,
            cf.hp_duration_vs_dates_mismatch,
            cf.hp_experience_vs_education_implausible,
        ])
        return cf

    # -- experience / seniority ------------------------------------------

    def _experience_features(self, c: dict, cf: CandidateFeatures) -> None:
        profile = c["profile"]
        history = c.get("career_history", [])
        cf.years_of_experience = float(profile.get("years_of_experience", 0.0))
        cf.num_jobs = len(history)

        durations = [j.get("duration_months", 0) or 0 for j in history]
        cf.avg_tenure_months = (sum(durations) / len(durations)) if durations else 0.0

        role_kw = [k.lower() for k in self.cfg["role_relevance_keywords"]]
        relevant_months = 0
        for job in history:
            job_text = f"{job.get('title','')} {job.get('description','')}".lower()
            if any(re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", job_text) for k in role_kw):
                relevant_months += job.get("duration_months", 0) or 0
        cf.years_in_relevant_roles = round(relevant_months / 12.0, 2)

        non_ic_titles = [t.lower() for t in self.cfg["non_ic_seniority_titles"]]

        def is_non_ic(title: str) -> bool:
            title_low = title.lower()
            return any(t in title_low for t in non_ic_titles)

        def has_code_signal(job: dict) -> bool:
            desc = job.get("description", "").lower()
            code_markers = ["built", "implemented", "wrote", "shipped", "coded",
                            "developed", "designed the", "owned the"]
            return any(m in desc for m in code_markers)

        current_job = next((j for j in history if j.get("is_current")), None)
        if current_job and is_non_ic(current_job.get("title", "")):
            cf.is_currently_non_ic_senior = True
            # look for the most recent job (by end_date) that had hands-on code signal
            months_since = None
            sorted_hist = sorted(
                (j for j in history if not j.get("is_current")),
                key=lambda j: _parse_date(j.get("end_date")) or date.min,
                reverse=True,
            )
            for job in sorted_hist:
                if has_code_signal(job) and not is_non_ic(job.get("title", "")):
                    end = _parse_date(job.get("end_date"))
                    if end:
                        months_since = (TODAY.year - end.year) * 12 + (TODAY.month - end.month)
                    break
            cf.months_since_last_ic_role = months_since

    # -- skill match -------------------------------------------------------

    def _skill_match_features(self, c: dict, cf: CandidateFeatures, text_blob: str) -> None:
        must_have = self.cfg["must_have_skills"]
        hits = {}
        for category, spec in must_have.items():
            n = _keyword_hits(text_blob, spec["keywords"])
            hits[category] = n > 0
        cf.must_have_hits = hits
        cf.must_have_score = round(sum(hits.values()) / max(len(hits), 1), 3)

        nice_to_have = self.cfg["nice_to_have_skills"]
        nice_hits = sum(1 for spec in nice_to_have.values() if _keyword_hits(text_blob, spec["keywords"]) > 0)
        cf.nice_to_have_score = round(nice_hits / max(len(nice_to_have), 1), 3)

        role_kw = self.cfg["role_relevance_keywords"]
        n_role_hits = _keyword_hits(text_blob, role_kw)
        # density capped at 5 distinct hits -> saturates the score; more than 5 doesn't add signal
        cf.role_relevance_score = round(min(n_role_hits, 5) / 5.0, 3)

        # Skill credibility gap: claimed proficiency vs actual assessment score, where both exist.
        # This is the direct counter to the "keyword stuffer" trap the JD warns about.
        signals = c.get("redrob_signals", {})
        assessments = signals.get("skill_assessment_scores", {}) or {}
        gaps = []
        for skill in c.get("skills", []):
            name = skill.get("name")
            prof = skill.get("proficiency")
            if name in assessments and prof in _PROFICIENCY_TO_EXPECTED_SCORE:
                expected = _PROFICIENCY_TO_EXPECTED_SCORE[prof]
                actual = assessments[name]
                gap = max(0.0, (expected - actual) / 100.0)  # only penalize over-claiming, not under-claiming
                gaps.append(gap)
        cf.skill_credibility_gap = round(sum(gaps) / len(gaps), 3) if gaps else 0.0

    # -- disqualifiers -------------------------------------------------------

    def _disqualifier_features(self, c: dict, cf: CandidateFeatures, text_blob: str) -> None:
        history = c.get("career_history", [])
        companies = [j.get("company", "").lower() for j in history]
        service_list = [s.lower() for s in self.cfg["service_only_companies"]]

        # Consulting-only: every company on record is a known service firm.
        if companies and all(any(sc in comp for sc in service_list) for comp in companies):
            cf.disq_consulting_only = True

        # CV/speech/robotics-only with no NLP/IR exposure.
        non_fit_hits = _keyword_hits(text_blob, self.cfg["non_fit_domain_keywords"])
        nlp_ir_hits = _keyword_hits(text_blob, ["nlp", "natural language processing",
                                                 "information retrieval", "retrieval", "ranking", "search"])
        if non_fit_hits > 0 and nlp_ir_hits == 0:
            cf.disq_cv_speech_robotics_only = True

        # Title-chaser: 3+ jobs, escalating seniority language, average tenure < 18 months.
        escalation_titles = ["senior", "staff", "principal", "lead"]
        has_escalation = any(_keyword_hits(j.get("title", ""), escalation_titles) > 0 for j in history)
        if cf.num_jobs >= 3 and cf.avg_tenure_months < 18 and has_escalation:
            cf.disq_title_chaser = True

        # Senior with no recent hands-on code (18mo+ threshold from JD).
        if cf.is_currently_non_ic_senior and (
            cf.months_since_last_ic_role is None or cf.months_since_last_ic_role >= 18
        ):
            cf.disq_senior_no_recent_code = True

        # Pure research, no production: industry/title signals research with no "production"/"shipped"/"deployed" language.
        research_signal = _keyword_hits(text_blob, ["research scientist", "research lab", "academia", "phd research"])
        prod_signal = _keyword_hits(text_blob, ["production", "deployed", "shipped", "real users", "in prod"])
        if research_signal > 0 and prod_signal == 0:
            cf.disq_pure_research_no_prod = True

        # Recent LangChain-only "AI experience" with no pre-LLM ML production history.
        langchain_hit = _keyword_hits(text_blob, ["langchain"])
        pre_llm_signal = _keyword_hits(text_blob, ["recommendation", "search relevance", "ranking",
                                                     "information retrieval", "nlp", "classification model"])
        # crude recency check: langchain only counts against them if it appears in a job with <12mo duration
        recent_short_ai_job = any(
            _keyword_hits(j.get("description", ""), ["langchain"]) > 0 and (j.get("duration_months", 999) < 12)
            for j in history
        )
        if langchain_hit > 0 and recent_short_ai_job and pre_llm_signal == 0:
            cf.disq_recent_langchain_only = True

    # -- education -------------------------------------------------------

    def _education_features(self, c: dict, cf: CandidateFeatures) -> None:
        tier_map = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.45, "tier_4": 0.25, "unknown": 0.0}
        edu = c.get("education", [])
        if edu:
            best = max(tier_map.get(e.get("tier", "unknown"), 0.0) for e in edu)
            cf.education_tier_score = best

    # -- logistics -------------------------------------------------------

    def _logistics_features(self, c: dict, cf: CandidateFeatures) -> None:
        profile = c["profile"]
        signals = c.get("redrob_signals", {})
        location = profile.get("location", "").lower()

        cf.location_in_target_list = any(loc in location for loc in self.cfg["target_ok_locations"])
        cf.location_is_preferred = any(loc in location for loc in self.cfg["preferred_locations"])
        cf.willing_to_relocate = bool(signals.get("willing_to_relocate", False))
        cf.notice_period_days = signals.get("notice_period_days", 999)
        cf.notice_period_ok = cf.notice_period_days <= self.cfg["notice_period_soft_max_days"]

    # -- behavioral --------------------------------------------------------

    def _behavioral_features(self, c: dict, cf: CandidateFeatures) -> None:
        signals = c.get("redrob_signals", {})
        cf.recruiter_response_rate = float(signals.get("recruiter_response_rate", 0.0))
        cf.interview_completion_rate = float(signals.get("interview_completion_rate", 0.0))
        raw_offer_rate = float(signals.get("offer_acceptance_rate", -1.0))
        cf.offer_acceptance_rate = 0.5 if raw_offer_rate < 0 else raw_offer_rate  # -1 sentinel -> neutral
        cf.open_to_work_flag = bool(signals.get("open_to_work_flag", False))

        last_active = _parse_date(signals.get("last_active_date"))
        if last_active:
            cf.months_since_last_active = round(
                (TODAY.year - last_active.year) * 12 + (TODAY.month - last_active.month), 1
            )

        # Composite availability score: recency decay * responsiveness * completion, gated by open_to_work.
        recency_decay = max(0.0, 1.0 - (cf.months_since_last_active / 12.0))  # linear decay over 12mo
        base = (0.4 * cf.recruiter_response_rate
                + 0.3 * cf.interview_completion_rate
                + 0.3 * cf.offer_acceptance_rate)
        availability_multiplier = 1.0 if cf.open_to_work_flag else 0.6
        cf.behavioral_availability_score = round(base * recency_decay * availability_multiplier, 3)

    # -- honeypot heuristics -------------------------------------------------

    def _honeypot_heuristics(self, c: dict, cf: CandidateFeatures) -> None:
        history = c.get("career_history", [])

        # 1. Total claimed experience vs sum of career_history durations.
        duration_years = sum(j.get("duration_months", 0) or 0 for j in history) / 12.0
        if abs(duration_years - cf.years_of_experience) > 2.0:
            cf.hp_experience_duration_mismatch = True

        # 2. Overlapping employment: two jobs with overlapping date ranges, or 2+ concurrent "is_current".
        current_count = sum(1 for j in history if j.get("is_current"))
        if current_count > 1:
            cf.hp_overlapping_employment = True
        else:
            intervals = []
            for j in history:
                start = _parse_date(j.get("start_date"))
                end = _parse_date(j.get("end_date")) or TODAY
                if start:
                    intervals.append((start, end))
            intervals.sort()
            for i in range(1, len(intervals)):
                if intervals[i][0] < intervals[i - 1][1]:
                    cf.hp_overlapping_employment = True
                    break

        # 3. "Expert" proficiency claimed with near-zero duration_months on that skill.
        for skill in c.get("skills", []):
            if skill.get("proficiency") == "expert" and (skill.get("duration_months") or 0) < 6:
                cf.hp_expert_with_near_zero_duration = True
                break

        # 4. duration_months field inconsistent with start_date/end_date gap (>3 month drift).
        for j in history:
            start = _parse_date(j.get("start_date"))
            end = _parse_date(j.get("end_date")) or TODAY
            if start:
                actual_months = (end.year - start.year) * 12 + (end.month - start.month)
                stated_months = j.get("duration_months", actual_months)
                if abs(actual_months - stated_months) > 3:
                    cf.hp_duration_vs_dates_mismatch = True
                    break

        # 5. Experience implausible given education timeline (started working before finishing degree
        #    by more than ~1 year of overlap, repeatedly, or claims more years than time since first degree end).
        edu = c.get("education", [])
        if edu:
            earliest_start = min(e.get("start_year", 9999) for e in edu)
            years_since_edu_start = TODAY.year - earliest_start
            if cf.years_of_experience > years_since_edu_start + 2:
                cf.hp_experience_vs_education_implausible = True


# ----------------------------------------------------------------------------
# CLI / smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pandas as pd

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/jd_requirements.yaml"
    sample_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample_candidates.json"

    fx = FeatureExtractor(config_path)
    with open(sample_path) as f:
        candidates = json.load(f)

    rows = [fx.extract(c).to_dict() for c in candidates]
    df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df[["candidate_id", "years_of_experience", "must_have_score", "role_relevance_score",
              "skill_credibility_gap", "disqualifier_count", "behavioral_availability_score",
              "honeypot_flag_count"]].to_string(index=False))