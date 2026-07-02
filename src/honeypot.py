"""Detect honeypot candidates via profile-consistency heuristics.

Honeypot pattern description (quoted from docs/submission_spec.docx, Section 7):

    "The dataset contains a small number (~80) of honeypot candidates with
    subtly impossible profiles (e.g., 8 years of experience at a company
    founded 3 years ago; 'expert' proficiency in 10 skills with 0 years
    used). These are forced to relevance tier 0 in the ground truth."

    "If your submission ranks honeypots in the top 10, this is a strong
    signal that your system isn't reading profiles — it's just doing keyword
    embedding. We use the honeypot rate as a Stage 3 filter: submissions
    with honeypot rate > 10% in top 100 are disqualified."

Additional honeypot context (quoted from docs/README.docx):

    "The dataset contains traps. Keyword stuffers, plain-language Tier 5s,
    behavioral twins, and ~80 honeypots with subtly impossible profiles.
    Submissions with honeypot rate > 10% in top 100 are disqualified."

Additional context (quoted from docs/job_description.docx):

    "A candidate who has all the AI keywords listed as skills but whose
    title is 'Marketing Manager' is not a fit, no matter how perfect their
    skill list looks."

Two independent detection layers:
  1. Explicit rule-based checks — each returns a partial penalty in [0, 1].
  2. IsolationForest anomaly detector — trained offline on the full feature
     table, loaded at runtime from a cached pickle.

Both are combined into a single inspectable HoneypotResult per candidate.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CONFIG_CACHE: dict | None = None


def _load_config(config_path: str = "config.yaml") -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with open(config_path, "r") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


def reset_config_cache() -> None:
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


# ---------------------------------------------------------------------------
# AI/ML skill keywords — used to detect non-technical title + skill stuffing.
# Loaded from config.yaml at runtime.
# ---------------------------------------------------------------------------

_NON_TECHNICAL_TITLE_PATTERNS: list[str] | None = None
_AI_ML_KEYWORDS: set[str] | None = None


def _get_non_technical_patterns(cfg: dict) -> list[str]:
    global _NON_TECHNICAL_TITLE_PATTERNS
    if _NON_TECHNICAL_TITLE_PATTERNS is None:
        _NON_TECHNICAL_TITLE_PATTERNS = [
            p.lower() for p in cfg.get("honeypot_non_technical_titles", [])
        ]
    return _NON_TECHNICAL_TITLE_PATTERNS


def _get_ai_ml_keywords(cfg: dict) -> set[str]:
    global _AI_ML_KEYWORDS
    if _AI_ML_KEYWORDS is None:
        _AI_ML_KEYWORDS = {
            k.lower() for k in cfg.get("honeypot_ai_ml_keywords", [])
        }
    return _AI_ML_KEYWORDS


# ---------------------------------------------------------------------------
# Result container — inspectable per-rule breakdown + isolation forest score.
# ---------------------------------------------------------------------------

@dataclass
class HoneypotResult:
    """Inspectable honeypot detection result for a single candidate."""

    candidate_id: str

    # Rule-based partial penalties (each 0.0–1.0, higher = more suspicious).
    skill_duration_mismatch_penalty: float = 0.0
    career_timeline_penalty: float = 0.0
    title_skill_stuffing_penalty: float = 0.0
    completeness_vs_career_penalty: float = 0.0

    # Which specific rules fired (human-readable descriptions).
    rules_fired: list[str] = field(default_factory=list)

    # Isolation-forest anomaly score (higher = more anomalous, range ~0–1).
    isolation_forest_score: float = 0.0

    # Combined honeypot score (weighted blend of rule + IF signals).
    honeypot_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "honeypot_score": round(self.honeypot_score, 4),
            "rule_penalties": {
                "skill_duration_mismatch": round(self.skill_duration_mismatch_penalty, 4),
                "career_timeline": round(self.career_timeline_penalty, 4),
                "title_skill_stuffing": round(self.title_skill_stuffing_penalty, 4),
                "completeness_vs_career": round(self.completeness_vs_career_penalty, 4),
            },
            "rules_fired": self.rules_fired,
            "isolation_forest_score": round(self.isolation_forest_score, 4),
        }


# ---------------------------------------------------------------------------
# Rule 1: Skill proficiency vs duration_months mismatch.
#
# Expert proficiency but very low duration_months is suspicious — you can't
# be "expert" in a skill you've used for 0–3 months.
# ---------------------------------------------------------------------------

def _rule_skill_duration_mismatch(candidate: dict, cfg: dict) -> tuple[float, list[str]]:
    """Return (penalty, reasons) for skills where proficiency doesn't match duration."""
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0, []

    thresholds = cfg.get("honeypot_skill_duration_thresholds", {})
    # Expected minimum months per proficiency level before it's plausible.
    min_months = {
        "expert": thresholds.get("expert_min_months", 12),
        "advanced": thresholds.get("advanced_min_months", 6),
    }

    violations = 0
    reasons: list[str] = []

    for s in skills:
        prof = s.get("proficiency", "beginner")
        duration = s.get("duration_months", 0)
        name = s.get("name", "unknown")

        if prof in min_months and duration < min_months[prof]:
            violations += 1
            reasons.append(
                f"'{name}': {prof} proficiency but only {duration} months"
            )

    if violations == 0:
        return 0.0, []

    # Penalty scales with the fraction of expert/advanced skills that violate.
    eligible = sum(1 for s in skills if s.get("proficiency") in min_months)
    ratio = violations / max(eligible, 1)

    # Also consider absolute count — 10+ mismatches is extremely suspicious.
    abs_factor = min(violations / thresholds.get("extreme_violation_count", 10), 1.0)

    penalty = min(1.0, 0.5 * ratio + 0.5 * abs_factor)
    return round(penalty, 4), reasons


# ---------------------------------------------------------------------------
# Rule 2: Career timeline inconsistencies.
#
# Checks for: overlapping jobs that don't make sense, duration_months that
# don't match start/end dates, total career duration exceeding YoE, etc.
# ---------------------------------------------------------------------------

def _parse_date_loose(d: str | None):
    """Parse YYYY-MM-DD, returning None on any failure."""
    if not d:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(d, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _rule_career_timeline(candidate: dict, cfg: dict) -> tuple[float, list[str]]:
    """Return (penalty, reasons) for timeline impossibilities."""
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)

    if not career:
        return 0.0, []

    tolerance = cfg.get("honeypot_timeline_tolerance_months", 3)
    issues: list[str] = []

    # Check 1: duration_months vs actual date span mismatch.
    for job in career:
        start = _parse_date_loose(job.get("start_date"))
        end = _parse_date_loose(job.get("end_date"))
        stated_dur = job.get("duration_months", 0)

        if start and end:
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            diff = abs(actual_months - stated_dur)
            if diff > tolerance:
                issues.append(
                    f"'{job.get('company','?')}': stated {stated_dur}mo but "
                    f"dates span {actual_months}mo (off by {diff})"
                )

    # Check 2: Total career months vs stated years_of_experience.
    total_months = sum(j.get("duration_months", 0) for j in career)
    yoe_months = yoe * 12
    # Allow some overlap (concurrent jobs), but flag extreme divergence.
    if total_months > 0 and yoe_months > 0:
        ratio = total_months / yoe_months
        divergence_threshold = cfg.get("honeypot_yoe_divergence_ratio", 2.0)
        if ratio > divergence_threshold:
            issues.append(
                f"Total career months ({total_months}) is {ratio:.1f}x "
                f"stated YoE ({yoe} years = {yoe_months:.0f}mo)"
            )

    # Check 3: Job at a company longer than the company could plausibly exist.
    # We can't know founding dates, but a single stint > 25 years is suspect.
    for job in career:
        dur = job.get("duration_months", 0)
        if dur > cfg.get("honeypot_max_single_tenure_months", 300):
            issues.append(
                f"'{job.get('company','?')}': {dur}mo tenure is implausibly long"
            )

    if not issues:
        return 0.0, []

    # Scale penalty by number of issues found.
    max_issues = cfg.get("honeypot_timeline_max_issues", 3)
    penalty = min(1.0, len(issues) / max_issues)
    return round(penalty, 4), issues


# ---------------------------------------------------------------------------
# Rule 3: Non-technical title with heavy AI/ML skill stuffing.
#
# A "Marketing Manager" or "HR Coordinator" claiming expert-level ML skills
# is suspicious (quoted directly from the JD's advice to participants).
# ---------------------------------------------------------------------------

def _rule_title_skill_stuffing(candidate: dict, cfg: dict) -> tuple[float, list[str]]:
    """Return (penalty, reasons) for non-technical titles with AI/ML skill lists."""
    profile = candidate.get("profile", {})
    title = (profile.get("current_title") or "").lower()
    skills = candidate.get("skills", [])

    non_tech_patterns = _get_non_technical_patterns(cfg)
    ai_ml_kw = _get_ai_ml_keywords(cfg)

    # Check if title matches any non-technical pattern.
    is_non_technical = any(pat in title for pat in non_tech_patterns)
    if not is_non_technical:
        return 0.0, []

    # Count AI/ML skills at advanced or expert level.
    ai_ml_skills: list[str] = []
    for s in skills:
        sname = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")
        if prof in ("advanced", "expert") and any(kw in sname for kw in ai_ml_kw):
            ai_ml_skills.append(f"{s.get('name')} ({prof})")

    if not ai_ml_skills:
        return 0.0, []

    reasons = [
        f"Title '{profile.get('current_title')}' is non-technical but has "
        f"{len(ai_ml_skills)} advanced/expert AI/ML skills: {', '.join(ai_ml_skills[:5])}"
    ]

    threshold = cfg.get("honeypot_stuffing_skill_threshold", 3)
    penalty = min(1.0, len(ai_ml_skills) / threshold)
    return round(penalty, 4), reasons


# ---------------------------------------------------------------------------
# Rule 4: Profile completeness vs thin career history mismatch.
#
# High profile_completeness_score (90+) but very few career entries / short
# total duration suggests a fabricated profile that was filled out
# meticulously but has no real substance.
# ---------------------------------------------------------------------------

def _rule_completeness_vs_career(candidate: dict, cfg: dict) -> tuple[float, list[str]]:
    """Return (penalty, reasons) for high completeness but thin career."""
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    completeness = signals.get("profile_completeness_score", 0)

    min_completeness = cfg.get("honeypot_completeness_min", 85)
    if completeness < min_completeness:
        return 0.0, []

    total_months = sum(j.get("duration_months", 0) for j in career)
    entry_count = len(career)

    reasons: list[str] = []

    # High completeness but very few career entries.
    max_thin_entries = cfg.get("honeypot_thin_career_max_entries", 1)
    max_thin_months = cfg.get("honeypot_thin_career_max_months", 12)

    if entry_count <= max_thin_entries and total_months <= max_thin_months:
        reasons.append(
            f"Profile completeness {completeness}% but only {entry_count} "
            f"career entries totaling {total_months}mo"
        )

    # High completeness + many skills but zero endorsements across all skills.
    skills = candidate.get("skills", [])
    total_endorsements = sum(s.get("endorsements", 0) for s in skills)
    if len(skills) >= 8 and total_endorsements == 0 and completeness >= 90:
        reasons.append(
            f"{len(skills)} skills with 0 total endorsements despite "
            f"{completeness}% profile completeness"
        )

    if not reasons:
        return 0.0, []

    penalty = min(1.0, len(reasons) * 0.6)
    return round(penalty, 4), reasons


# ---------------------------------------------------------------------------
# Isolation Forest loader — loads the pre-trained model from a pickle file.
# ---------------------------------------------------------------------------

_IF_MODEL = None
_IF_FEATURE_NAMES: list[str] | None = None


def _load_isolation_forest(cfg: dict):
    """Load the pre-trained IsolationForest from the precompute directory."""
    global _IF_MODEL, _IF_FEATURE_NAMES
    if _IF_MODEL is not None:
        return _IF_MODEL, _IF_FEATURE_NAMES

    model_path = Path(cfg.get(
        "honeypot_if_model_path",
        "precompute/honeypot_iforest.pkl",
    ))

    if not model_path.exists():
        return None, None

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    _IF_MODEL = artifact["model"]
    _IF_FEATURE_NAMES = artifact["feature_names"]
    return _IF_MODEL, _IF_FEATURE_NAMES


def _isolation_forest_score(features: dict[str, Any], cfg: dict) -> float:
    """Score a candidate using the pre-trained IsolationForest.

    Returns a score in [0, 1] where higher = more anomalous.
    Uses sklearn's decision_function (negated and rescaled).
    """
    model, feature_names = _load_isolation_forest(cfg)
    if model is None or feature_names is None:
        return 0.0

    import numpy as np

    # Build the feature vector in the same order used during training.
    row = []
    for fname in feature_names:
        val = features.get(fname, 0)
        # Convert booleans to int.
        if isinstance(val, bool):
            val = int(val)
        # Convert non-numeric to 0 (e.g., string categoricals).
        if not isinstance(val, (int, float)):
            val = 0
        row.append(float(val))

    X = np.array([row])

    # decision_function: negative = anomaly, positive = normal.
    raw = model.decision_function(X)[0]

    # Convert to 0–1 scale: more negative → closer to 1.
    # Typical range is roughly [-0.5, 0.5]; we clamp and rescale.
    score = max(0.0, min(1.0, 0.5 - raw))
    return round(score, 4)


# ---------------------------------------------------------------------------
# Public API — compute_honeypot_score
# ---------------------------------------------------------------------------

def compute_honeypot_score(
    candidate: dict,
    features: dict[str, Any],
    config_path: str = "config.yaml",
) -> HoneypotResult:
    """Compute an inspectable honeypot score for a single candidate.

    Args:
        candidate: Raw candidate dict (full profile data).
        features: Pre-extracted feature dict from features.extract_features().
        config_path: Path to config.yaml.

    Returns:
        HoneypotResult with per-rule penalties, IF score, and combined score.
    """
    cfg = _load_config(config_path)
    cid = candidate.get("candidate_id", "unknown")
    result = HoneypotResult(candidate_id=cid)

    # --- Rule-based checks ---
    penalty1, reasons1 = _rule_skill_duration_mismatch(candidate, cfg)
    result.skill_duration_mismatch_penalty = penalty1
    result.rules_fired.extend(reasons1)

    penalty2, reasons2 = _rule_career_timeline(candidate, cfg)
    result.career_timeline_penalty = penalty2
    result.rules_fired.extend(reasons2)

    penalty3, reasons3 = _rule_title_skill_stuffing(candidate, cfg)
    result.title_skill_stuffing_penalty = penalty3
    result.rules_fired.extend(reasons3)

    penalty4, reasons4 = _rule_completeness_vs_career(candidate, cfg)
    result.completeness_vs_career_penalty = penalty4
    result.rules_fired.extend(reasons4)

    # Weighted combination of rule penalties.
    rule_weights = cfg.get("honeypot_rule_weights", {})
    w1 = rule_weights.get("skill_duration_mismatch", 0.30)
    w2 = rule_weights.get("career_timeline", 0.30)
    w3 = rule_weights.get("title_skill_stuffing", 0.25)
    w4 = rule_weights.get("completeness_vs_career", 0.15)

    rule_score = w1 * penalty1 + w2 * penalty2 + w3 * penalty3 + w4 * penalty4

    # --- Isolation Forest ---
    if_score = _isolation_forest_score(features, cfg)
    result.isolation_forest_score = if_score

    # --- Combine rule + IF into final honeypot_score ---
    blend = cfg.get("honeypot_blend_weights", {})
    rule_weight = blend.get("rules", 0.6)
    if_weight = blend.get("isolation_forest", 0.4)

    result.honeypot_score = round(
        rule_weight * rule_score + if_weight * if_score,
        4,
    )

    return result
