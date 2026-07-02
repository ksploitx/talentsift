"""Composite scoring: combine retrieval, structured-fit, and behavioral signals."""

from typing import Any
import yaml

_CONFIG_CACHE: dict | None = None

def _load_config(config_path: str = "config.yaml") -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with open(config_path, "r") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE

def _min_max_normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list of floats to the range [0.0, 1.0]."""
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v > min_v:
        return [(v - min_v) / (max_v - min_v) for v in values]
    return [0.0 for _ in values]

def compute_composite_scores(
    candidates_features: list[tuple[str, dict[str, Any]]],
    retrieval_scores: dict[str, float],
    honeypot_scores: dict[str, float],
    config_path: str = "config.yaml"
) -> list[dict[str, Any]]:
    """
    Computes normalized composite scores for the population.
    Applies the honeypot score as a multiplicative penalty.
    """
    cfg = _load_config(config_path)
    weights = cfg.get("weights", {})
    w_ret = weights.get("retrieval", 0.5)
    w_fit = weights.get("structured_fit", 0.3)
    w_beh = weights.get("behavioral", 0.2)
    
    n = len(candidates_features)
    if n == 0:
        return []
        
    cids = []
    raw_fit = []
    raw_beh = []
    raw_ret = []
    
    for cid, feats in candidates_features:
        cids.append(cid)
        
        # Structured Fit (higher is better)
        loc_score = max(0.0, 4.0 - feats.get("location_tier", 4))
        prod_bonus = 2.0 if feats.get("is_product_company") else 0.0
        cons_penalty = -2.0 if feats.get("consulting_firm_flag") else 0.0
        
        fit_val = (
            feats.get("experience_fit_score", 0.0) * 5.0 +
            feats.get("skill_count_by_proficiency", 0.0) +
            feats.get("company_size_score", 0.0) +
            loc_score * 2.0 +
            prod_bonus +
            cons_penalty +
            feats.get("education_tier_score", 0.0)
        )
        raw_fit.append(fit_val)
        
        # Behavioral (higher is better)
        notice_days = feats.get("notice_period_days", 90)
        notice_score = max(0.0, 90.0 - notice_days)
        
        days_active = feats.get("days_since_active", 999)
        active_score = max(0.0, 365.0 - days_active)
        
        beh_val = (
            feats.get("platform_engagement_score", 0.0) +
            feats.get("recruiter_response_rate", 0.0) * 100.0 +
            feats.get("verification_score", 0.0) * 10.0 +
            notice_score +
            active_score * 0.1
        )
        raw_beh.append(beh_val)
        
        # Retrieval
        raw_ret.append(retrieval_scores.get(cid, 0.0))
        
    # Normalize across population
    norm_fit = _min_max_normalize(raw_fit)
    norm_beh = _min_max_normalize(raw_beh)
    norm_ret = _min_max_normalize(raw_ret)
    
    results = []
    for i in range(n):
        cid = cids[i]
        hp_penalty = honeypot_scores.get(cid, 0.0)
        
        composite_before_hp = (
            w_ret * norm_ret[i] +
            w_fit * norm_fit[i] +
            w_beh * norm_beh[i]
        )
        
        # Apply multiplicative penalty: (1 - penalty) * score
        final_score = composite_before_hp * max(0.0, 1.0 - hp_penalty)
        
        results.append({
            "candidate_id": cid,
            "final_score": round(final_score, 4),
            "composite_score_before_penalty": round(composite_before_hp, 4),
            "norm_retrieval": round(norm_ret[i], 4),
            "norm_structured_fit": round(norm_fit[i], 4),
            "norm_behavioral": round(norm_beh[i], 4),
            "honeypot_penalty": round(hp_penalty, 4)
        })
        
    return results
