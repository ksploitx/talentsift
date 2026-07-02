"""Build human-readable reasoning strings from real feature values.

Each reasoning string is assembled from templates using actual candidate
data — title, company, years of experience, top skills, notice period,
location, and flagged concerns.  No LLM calls, no API calls.
"""

from __future__ import annotations

from typing import Any


def _top_skills(candidate: dict, n: int = 3) -> list[dict]:
    """Return the top-n skills sorted by proficiency tier then endorsements."""
    tier = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    skills = candidate.get("skills", [])
    ranked = sorted(
        skills,
        key=lambda s: (
            tier.get(s.get("proficiency", "beginner"), 0),
            s.get("endorsements", 0),
            s.get("duration_months", 0),
        ),
        reverse=True,
    )
    return ranked[:n]


def _format_skill(s: dict) -> str:
    name = s.get("name", "?")
    prof = s.get("proficiency", "?")
    dur = s.get("duration_months", 0)
    end = s.get("endorsements", 0)
    parts = [f"{name} ({prof}"]
    if dur:
        parts.append(f", {dur}mo")
    if end:
        parts.append(f", {end} endorsements")
    return "".join(parts) + ")"


def _notice_label(days: int) -> str:
    if days <= 0:
        return "immediately available"
    if days <= 30:
        return f"{days}-day notice (ideal)"
    if days <= 60:
        return f"{days}-day notice (acceptable)"
    if days <= 90:
        return f"{days}-day notice (long)"
    return f"{days}-day notice (very long)"


def _concerns(candidate: dict, features: dict[str, Any]) -> list[str]:
    """Collect notable concerns worth mentioning in the reasoning."""
    issues: list[str] = []

    # Location
    loc_tier = features.get("location_tier", 4)
    location = candidate.get("profile", {}).get("location", "Unknown")
    if loc_tier >= 3:
        issues.append(f"non-preferred location ({location}, tier {loc_tier})")

    # Consulting-only career
    if features.get("consulting_only_career"):
        issues.append("entire career at consulting/services firms")

    # Job-hopping
    ratio = features.get("short_tenure_ratio", 0)
    if ratio > 0.5:
        issues.append(f"high job-hopping ratio ({ratio:.0%} short tenures)")

    # Experience out of range
    fit = features.get("experience_fit_score", 1.0)
    yoe = features.get("years_of_experience", 0)
    if fit < 0.7:
        issues.append(f"experience outside ideal range ({yoe} years, fit {fit:.2f})")

    # Honeypot penalty (passed in via features augmentation in rank.py)
    hp = features.get("honeypot_penalty", 0.0)
    if hp > 0.3:
        issues.append(f"elevated honeypot risk ({hp:.2f})")

    return issues


def _strengths(candidate: dict, features: dict[str, Any]) -> list[str]:
    """Collect notable strengths worth mentioning."""
    points: list[str] = []

    if features.get("is_product_company"):
        company = candidate.get("profile", {}).get("current_company", "")
        if company:
            points.append(f"product company experience ({company})")

    gh = features.get("github_activity_score", 0)
    if gh > 0.5:
        points.append(f"active GitHub contributor (score {gh:.2f})")

    eng = features.get("platform_engagement_score", 0)
    if eng > 60:
        points.append(f"high platform engagement ({eng:.0f}/100)")

    resp = features.get("recruiter_response_rate", 0)
    if resp > 0.7:
        points.append(f"strong recruiter response rate ({resp:.0%})")

    loc_tier = features.get("location_tier", 4)
    if loc_tier == 1:
        location = candidate.get("profile", {}).get("location", "")
        points.append(f"preferred office location ({location})")

    edu = features.get("education_tier_score", 0)
    if edu >= 3:
        points.append(f"tier-{5 - edu} education background")

    return points


def generate_reasoning(
    candidate: dict,
    features: dict[str, Any],
    score_row: dict[str, Any] | None = None,
) -> str:
    """Build a reasoning string for why this candidate was ranked here.

    Args:
        candidate: Raw candidate dict.
        features: Extracted feature dict from features.extract_features().
        score_row: Optional dict from score.compute_composite_scores() with
            final_score, norm_retrieval, etc.

    Returns:
        A multi-sentence reasoning string built from real values.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown Role")
    company = profile.get("current_company", "Unknown Company")
    yoe = profile.get("years_of_experience", 0)
    notice = signals.get("notice_period_days", 0)

    # Top skills
    top = _top_skills(candidate)
    skill_strs = [_format_skill(s) for s in top]

    # Header
    lines: list[str] = []
    lines.append(
        f"{title} at {company} with {yoe} years of experience."
    )

    # Skills
    if skill_strs:
        lines.append(f"Top skills: {', '.join(skill_strs)}.")

    # Availability
    lines.append(f"Availability: {_notice_label(notice)}.")

    # Score breakdown
    if score_row:
        final = score_row.get("final_score", 0)
        nr = score_row.get("norm_retrieval", 0)
        nf = score_row.get("norm_structured_fit", 0)
        nb = score_row.get("norm_behavioral", 0)
        lines.append(
            f"Score breakdown — retrieval: {nr:.2f}, "
            f"structured fit: {nf:.2f}, behavioral: {nb:.2f} "
            f"→ final: {final:.4f}."
        )

    # Strengths
    strengths = _strengths(candidate, features)
    if strengths:
        lines.append(f"Strengths: {'; '.join(strengths)}.")

    # Concerns
    concerns = _concerns(candidate, features)
    if concerns:
        lines.append(f"Concerns: {'; '.join(concerns)}.")

    return " ".join(lines)
