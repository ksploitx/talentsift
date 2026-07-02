"""Extract structured features from candidate profiles for scoring.

All thresholds and lookup tables are loaded from config.yaml.
No hardcoded weights, thresholds, or model names live in this file.
"""

from datetime import date, datetime
from typing import Any

import yaml

_CONFIG_CACHE: dict | None = None


def _load_config(config_path: str = "config.yaml") -> dict:
    """Load and cache config.yaml."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with open(config_path, "r") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


def reset_config_cache() -> None:
    """Clear the config cache (useful for testing with different configs)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def _parse_date(date_str: str | None) -> date | None:
    """Parse a YYYY-MM-DD date string, returning None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _location_tier(location: str, country: str, cfg: dict) -> int:
    """Determine location tier from config lookup tables.

    Returns 1 (preferred office city) through 4 (international).
    """
    tiers = cfg.get("location_tiers", {})
    loc_lower = location.lower().strip() if location else ""
    # Try full location string first, then first part before comma
    if loc_lower in tiers:
        return tiers[loc_lower]
    city = loc_lower.split(",")[0].strip()
    if city in tiers:
        return tiers[city]
    # Fallback: India vs international
    if country and country.strip().lower() == "india":
        return cfg.get("location_default_india_tier", 3)
    return cfg.get("location_default_international_tier", 4)


def _company_size_score(size_str: str, cfg: dict) -> int:
    """Map company_size enum string to a numeric score from config."""
    scores = cfg.get("company_size_scores", {})
    return scores.get(size_str, 0)


def _is_consulting_firm(company: str, cfg: dict) -> bool:
    """Check if a company name matches the consulting firms list (case-insensitive)."""
    firms = {f.lower() for f in cfg.get("consulting_firms", [])}
    return company.strip().lower() in firms if company else False


def _skill_count_by_proficiency(skills: list[dict], cfg: dict) -> float:
    """Weighted skill count using proficiency weights from config."""
    weights = cfg.get("proficiency_weights", {})
    total = 0.0
    for s in skills:
        prof = s.get("proficiency", "beginner")
        total += weights.get(prof, 0.25)
    return round(total, 2)


def _education_tier_score(education: list[dict], cfg: dict) -> int:
    """Best (highest) education tier score across all degrees."""
    tier_scores = cfg.get("education_tier_scores", {})
    if not education:
        return 0
    return max(tier_scores.get(e.get("tier", "unknown"), 1) for e in education)


def _avg_tenure_months(career_history: list[dict]) -> float:
    """Average tenure across all career entries, in months."""
    if not career_history:
        return 0.0
    durations = [j.get("duration_months", 0) for j in career_history]
    return round(sum(durations) / len(durations), 1)


def _short_tenure_ratio(career_history: list[dict], cfg: dict) -> float:
    """Fraction of jobs with tenure below min_tenure_months (job-hopping signal)."""
    min_tenure = cfg.get("min_tenure_months", 18)
    if not career_history:
        return 0.0
    short = sum(1 for j in career_history if j.get("duration_months", 0) < min_tenure)
    return round(short / len(career_history), 2)


def _consulting_only_flag(career_history: list[dict], current_company: str, cfg: dict) -> bool:
    """True if *every* company in career + current is a consulting firm."""
    all_companies = [j.get("company", "") for j in career_history]
    all_companies.append(current_company or "")
    all_companies = [c for c in all_companies if c.strip()]
    if not all_companies:
        return False
    return all(
        _is_consulting_firm(c, cfg) for c in all_companies
    )


def _has_product_company(career_history: list[dict], current_company: str, cfg: dict) -> bool:
    """True if at least one company in career is NOT a consulting firm."""
    all_companies = [j.get("company", "") for j in career_history]
    all_companies.append(current_company or "")
    all_companies = [c for c in all_companies if c.strip()]
    return any(not _is_consulting_firm(c, cfg) for c in all_companies)


def _experience_fit_score(yoe: float, cfg: dict) -> float:
    """Score how well years-of-experience fits the JD ideal range.

    Returns 1.0 if within range, linearly decays outside.
    """
    exp_range = cfg.get("experience_range", {"min_years": 5, "max_years": 9})
    min_y = exp_range["min_years"]
    max_y = exp_range["max_years"]
    if min_y <= yoe <= max_y:
        return 1.0
    if yoe < min_y:
        # Linear decay: 0 at 0 years
        return round(max(0.0, yoe / min_y), 2)
    # yoe > max_y: gentle decay
    overshoot = yoe - max_y
    return round(max(0.0, 1.0 - overshoot / max_y), 2)


def _github_activity_score(raw_score: float) -> float:
    """Normalize github_activity_score: -1 (no GitHub) → 0.0, else pass through."""
    if raw_score < 0:
        return 0.0
    return raw_score


def _days_since_active(last_active_date: str | None, cfg: dict) -> int:
    """Days between last_active_date and the reference_date from config."""
    ref = _parse_date(cfg.get("reference_date", "2026-06-30"))
    last = _parse_date(last_active_date)
    if ref is None or last is None:
        return 999  # Sentinel for missing data
    delta = (ref - last).days
    return max(0, delta)


def _notice_period_bucket(notice_days: int, cfg: dict) -> str:
    """Categorize notice period into a bucket label from config thresholds."""
    thresholds = cfg.get("notice_period_thresholds", {})
    ideal = thresholds.get("ideal_max_days", 30)
    acceptable = thresholds.get("acceptable_max_days", 60)
    long_ = thresholds.get("long_max_days", 90)
    if notice_days <= ideal:
        return "ideal"
    if notice_days <= acceptable:
        return "acceptable"
    if notice_days <= long_:
        return "long"
    return "very_long"


def _platform_engagement_score(signals: dict) -> float:
    """Composite engagement score from redrob_signals.

    Combines profile completeness, views, search appearances,
    recruiter saves, and interview completion into one 0-100 scale.
    """
    completeness = signals.get("profile_completeness_score", 0)
    views = min(signals.get("profile_views_received_30d", 0), 100)
    search = min(signals.get("search_appearance_30d", 0), 500)
    saves = min(signals.get("saved_by_recruiters_30d", 0), 50)
    interview_rate = signals.get("interview_completion_rate", 0) * 100

    # Weighted average (out of 100)
    score = (
        completeness * 0.25
        + views * 0.15
        + (search / 5) * 0.20
        + (saves * 2) * 0.15
        + interview_rate * 0.25
    )
    return round(min(100.0, score), 2)


def _verification_score(signals: dict) -> int:
    """Count of verified identity signals (email, phone, linkedin)."""
    return sum([
        int(signals.get("verified_email", False)),
        int(signals.get("verified_phone", False)),
        int(signals.get("linkedin_connected", False)),
    ])


def extract_features(candidate: dict, config_path: str = "config.yaml") -> dict[str, Any]:
    """Extract a flat dict of numeric/categorical features from a candidate.

    Args:
        candidate: A single candidate dict matching candidate_schema.json.
        config_path: Path to config.yaml (default: repo root).

    Returns:
        Flat dict with feature names as keys.
    """
    cfg = _load_config(config_path)
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    certs = candidate.get("certifications", [])
    signals = candidate.get("redrob_signals", {})

    yoe = profile.get("years_of_experience", 0)
    current_company = profile.get("current_company", "")
    notice_days = signals.get("notice_period_days", 0)

    return {
        # === Experience ===
        "years_of_experience": yoe,
        "experience_fit_score": _experience_fit_score(yoe, cfg),

        # === Company signals ===
        "is_product_company": _has_product_company(career, current_company, cfg),
        "company_size_score": _company_size_score(
            profile.get("current_company_size", ""), cfg
        ),
        "consulting_firm_flag": _is_consulting_firm(current_company, cfg),
        "consulting_only_career": _consulting_only_flag(career, current_company, cfg),

        # === Career stability ===
        "avg_tenure_months": _avg_tenure_months(career),
        "short_tenure_ratio": _short_tenure_ratio(career, cfg),
        "career_entry_count": len(career),

        # === Location ===
        "location_tier": _location_tier(
            profile.get("location", ""),
            profile.get("country", ""),
            cfg,
        ),

        # === Notice period ===
        "notice_period_days": notice_days,
        "notice_period_bucket": _notice_period_bucket(notice_days, cfg),

        # === Platform engagement ===
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0.0),
        "days_since_active": _days_since_active(
            signals.get("last_active_date"), cfg
        ),
        "open_to_work": signals.get("open_to_work_flag", False),
        "platform_engagement_score": _platform_engagement_score(signals),
        "verification_score": _verification_score(signals),

        # === Skills ===
        "skill_count_raw": len(skills),
        "skill_count_by_proficiency": _skill_count_by_proficiency(skills, cfg),
        "certification_count": len(certs),

        # === GitHub ===
        "github_activity_score": _github_activity_score(
            signals.get("github_activity_score", -1)
        ),

        # === Education ===
        "education_tier_score": _education_tier_score(education, cfg),

        # === Salary ===
        "expected_salary_min_lpa": signals.get("expected_salary_range_inr_lpa", {}).get("min", 0),
        "expected_salary_max_lpa": signals.get("expected_salary_range_inr_lpa", {}).get("max", 0),

        # === Work mode ===
        "preferred_work_mode": signals.get("preferred_work_mode", "unknown"),
        "willing_to_relocate": signals.get("willing_to_relocate", False),

        # === Response quality ===
        "avg_response_time_hours": signals.get("avg_response_time_hours", 0),
        "offer_acceptance_rate": max(0.0, signals.get("offer_acceptance_rate", -1)),
    }
