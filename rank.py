#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranking
Author: Khushneet Singh (team: ksploitx)

Strategy:
- Rule-based multi-component scorer (no LLM API calls, no GPU)
- Five scoring dimensions: career_fit, skills, experience, signals, location
- Multiplicative behavioral modifier
- Honeypot detection via profile consistency checks
- Runtime: ~60s on CPU for 100K candidates
"""

import json
import csv
import argparse
import re
from datetime import date, datetime
from pathlib import Path

# ─── JD constants ────────────────────────────────────────────────────────────

MUST_HAVE_SKILLS = {
    # embeddings retrieval
    "sentence-transformers", "sentence transformers", "bge", "e5", "embeddings",
    "openai embeddings", "text embeddings",
    # vector / hybrid search
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "vector search", "hybrid search", "vector database",
    "vector db", "ann", "approximate nearest neighbor",
    # retrieval / ranking
    "retrieval", "information retrieval", "bm25", "dense retrieval", "rag",
    "retrieval augmented", "reranking", "re-ranking", "learning to rank",
    "semantic search",
    # eval frameworks
    "ndcg", "mrr", "map", "a/b test", "ab test", "eval framework",
    "evaluation framework", "ranking evaluation",
    # python & ml
    "python", "pytorch", "tensorflow", "scikit-learn", "sklearn",
    "transformers", "huggingface", "hugging face",
    # LLMs
    "llm", "large language model", "gpt", "fine-tuning", "finetuning",
    "lora", "qlora", "peft", "instruction tuning",
    # xgboost/LTR
    "xgboost", "lightgbm", "learning to rank", "ltr",
}

NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "finetuning",
    "xgboost", "lightgbm", "ltr", "learning to rank",
    "hr-tech", "hrtech", "recruiting", "marketplace",
    "distributed systems", "inference optimization", "kafka", "spark",
    "open-source", "open source", "github", "mlops",
}

# Consulting/pure-services companies to penalize
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "niit",
    "l&t infotech", "ltimindtree", "mindtree", "zensar", "cyient",
    "mastech", "kpit", "persistent systems",
}

# Preferred locations for this JD
PREFERRED_LOCATIONS = {
    "noida", "pune", "delhi", "new delhi", "gurgaon", "gurugram",
    "hyderabad", "mumbai", "bangalore", "bengaluru", "india",
}

# Skills that are red-herring if that's ALL someone has (CV/speech/robotics)
CV_ONLY_SKILLS = {
    "computer vision", "image classification", "object detection",
    "image segmentation", "yolo", "opencv", "speech recognition",
    "text to speech", "tts", "asr", "ocr", "robotics", "ros",
}

# ─── Honeypot detection ───────────────────────────────────────────────────────

def detect_honeypot(candidate: dict) -> float:
    """Returns a penalty 0.0 (clean) to 1.0 (definite honeypot)."""
    signals = 0
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # 1. Career timeline inconsistencies
    for job in career:
        company = job.get("company", "").lower()
        start = job.get("start_date", "")
        duration = job.get("duration_months", 0)
        # Can't have worked somewhere longer than company existed (rough heuristic)
        # If duration > years_of_experience * 12 + 12 — suspicious
        yoe = profile.get("years_of_experience", 0) or 0
        if duration > (yoe * 12 + 24):
            signals += 1

    # 2. Skill proficiency vs duration inconsistency
    for skill in skills:
        prof = skill.get("proficiency", "")
        dur = skill.get("duration_months", 0) or 0
        if prof == "expert" and dur < 6:
            signals += 1
        if prof == "advanced" and dur == 0:
            signals += 0.5

    # 3. Too many "expert" skills with 0 endorsements
    expert_zero = sum(1 for s in skills
                     if s.get("proficiency") == "expert" and s.get("endorsements", 0) == 0)
    if expert_zero >= 5:
        signals += expert_zero * 0.3

    # 4. Profile completeness vs actual content mismatch
    completeness = candidate.get("redrob_signals", {}).get("profile_completeness_score", 0)
    if completeness > 95 and len(career) < 2:
        signals += 0.5

    # 5. Title vs skills mismatch — e.g. "Marketing Manager" with all AI skills
    title = profile.get("current_title", "").lower()
    non_tech_titles = ["marketing", "sales", "finance", "hr ", "human resource",
                       "accountant", "legal", "content writer", "seo "]
    if any(t in title for t in non_tech_titles):
        ai_skills = sum(1 for s in skills
                       if any(k in s.get("name", "").lower() for k in ["llm", "rag", "vector", "embedding", "pytorch"]))
        if ai_skills >= 5:
            signals += 2

    return min(signals / 5.0, 1.0)


# ─── Component scorers ────────────────────────────────────────────────────────

def score_skills(candidate: dict) -> float:
    """0-1: How well skills match the JD requirements."""
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    skill_names = set()
    skill_text = ""
    for s in skills:
        name = s.get("name", "").lower()
        skill_names.add(name)
        skill_text += " " + name

    # Check career descriptions too
    for job in candidate.get("career_history", []):
        skill_text += " " + job.get("description", "").lower()
    profile_text = candidate.get("profile", {}).get("summary", "").lower()
    skill_text += " " + profile_text

    # Must-have: weighted by proficiency and endorsements
    must_score = 0.0
    must_max = 0.0
    proficiency_weights = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.6, "beginner": 0.3}

    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "intermediate")
        endorse = min(s.get("endorsements", 0), 50) / 50.0
        duration = min(s.get("duration_months", 0), 48) / 48.0
        pw = proficiency_weights.get(prof, 0.5)

        # Trust multiplier: endorsed + duration backed
        trust = 0.4 + 0.3 * endorse + 0.3 * duration

        if any(k in name for k in MUST_HAVE_SKILLS):
            must_score += pw * trust
            must_max += 1.0

    # Also check text mentions (lower weight, handles aliases)
    text_matches = sum(1 for k in MUST_HAVE_SKILLS if k in skill_text)
    text_score = min(text_matches / 8.0, 1.0)  # normalize

    nice_score = sum(1 for k in NICE_TO_HAVE_SKILLS if k in skill_text)
    nice_score = min(nice_score / 5.0, 1.0)

    combined = (
        0.5 * (must_score / max(must_max, 1)) +
        0.3 * text_score +
        0.2 * nice_score
    )
    return min(combined, 1.0)


def score_career(candidate: dict) -> float:
    """0-1: Career trajectory fit — product companies, AI/ML titles, relevant roles."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    score = 0.0
    total_weight = 0.0

    # Current role matters most
    current_title = profile.get("current_title", "").lower()
    current_industry = profile.get("current_industry", "").lower()
    current_company = profile.get("current_company", "").lower()
    current_size = profile.get("current_company_size", "")

    # Title signals
    ai_titles = ["ai engineer", "ml engineer", "machine learning", "data scientist",
                 "nlp engineer", "search engineer", "ranking engineer", "applied scientist",
                 "research engineer", "ai researcher", "llm engineer"]
    tech_titles = ["software engineer", "backend engineer", "swe", "data engineer",
                   "platform engineer", "infrastructure"]

    title_score = 0.0
    if any(t in current_title for t in ai_titles):
        title_score = 1.0
    elif any(t in current_title for t in tech_titles):
        title_score = 0.5
    elif "tech lead" in current_title or "principal" in current_title:
        title_score = 0.4
    elif "manager" in current_title and "engineering" in current_title:
        title_score = 0.3

    score += 0.3 * title_score
    total_weight += 0.3

    # Industry — product companies > IT services
    bad_industries = ["it services", "consulting", "staffing", "bpo", "outsourcing"]
    good_industries = ["fintech", "edtech", "healthtech", "saas", "e-commerce",
                       "artificial intelligence", "machine learning", "technology",
                       "internet", "software", "product"]
    if any(b in current_industry for b in bad_industries):
        industry_score = 0.2
    elif any(g in current_industry for g in good_industries):
        industry_score = 1.0
    else:
        industry_score = 0.6

    score += 0.2 * industry_score
    total_weight += 0.2

    # Company size — startup/mid preferred
    size_map = {
        "1-10": 0.7, "11-50": 0.85, "51-200": 1.0, "201-500": 0.95,
        "501-1000": 0.85, "1001-5000": 0.7, "5001-10000": 0.5, "10001+": 0.3
    }
    score += 0.1 * size_map.get(current_size, 0.5)
    total_weight += 0.1

    # Consulting firm penalty
    for firm in CONSULTING_FIRMS:
        if firm in current_company:
            score -= 0.15
            break

    # Career history analysis
    career_score = 0.0
    product_co_months = 0
    consulting_months = 0
    ai_role_months = 0

    for job in career:
        company = job.get("company", "").lower()
        title = job.get("title", "").lower()
        industry = job.get("industry", "").lower()
        duration = job.get("duration_months", 0) or 0
        desc = job.get("description", "").lower()
        size = job.get("company_size", "")

        is_consulting = any(f in company for f in CONSULTING_FIRMS)
        is_product = any(g in industry for g in good_industries) or size in ["11-50", "51-200", "201-500"]
        is_ai_role = any(t in title for t in ai_titles)

        # Retrieval/ranking evidence in descriptions
        has_retrieval = any(k in desc for k in [
            "retrieval", "embedding", "vector", "search", "ranking", "rag",
            "recommendation", "nlp", "llm", "transformer", "bert", "faiss",
            "pinecone", "elasticsearch", "opensearch", "hybrid"
        ])

        if is_consulting:
            consulting_months += duration
        elif is_product:
            product_co_months += duration

        if is_ai_role:
            ai_role_months += duration

        if has_retrieval and not is_consulting:
            career_score += min(duration / 12.0, 3.0) * 0.1  # up to 0.3 per job

    # Ratio of product vs consulting experience
    total_months = product_co_months + consulting_months + 1
    product_ratio = product_co_months / total_months
    score += 0.25 * product_ratio
    total_weight += 0.25

    score += min(career_score, 0.15)
    total_weight += 0.15

    return max(0.0, min(score / total_weight * total_weight, 1.0))


def score_experience(candidate: dict) -> float:
    """0-1: Years of experience in the right range (5-9 ideal)."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0

    if 5 <= yoe <= 9:
        return 1.0
    elif 4 <= yoe < 5:
        return 0.85
    elif 9 < yoe <= 12:
        return 0.75
    elif 3 <= yoe < 4:
        return 0.6
    elif yoe > 12:
        return 0.55  # overqualified / might want staff/principal role
    elif 2 <= yoe < 3:
        return 0.35
    else:
        return 0.1


def score_location(candidate: dict) -> float:
    """0-1: Location fit for Pune/Noida-preferred, India-wide acceptable."""
    profile = candidate.get("profile", {})
    location = (profile.get("location", "") + " " + profile.get("country", "")).lower()
    signals = candidate.get("redrob_signals", {})
    will_relocate = signals.get("willing_to_relocate", False)

    # Tier 1: Preferred cities
    if any(c in location for c in ["noida", "pune", "delhi", "gurgaon", "gurugram"]):
        return 1.0
    # Tier 2: Other major Indian cities
    elif any(c in location for c in ["bangalore", "bengaluru", "hyderabad", "mumbai",
                                      "chennai", "kolkata"]):
        return 0.9
    # Tier 3: Anywhere in India
    elif "india" in location:
        return 0.75
    # Tier 4: Willing to relocate
    elif will_relocate:
        return 0.5
    # Tier 5: International, not willing to relocate
    else:
        return 0.2


def score_behavioral_signals(candidate: dict) -> float:
    """0-1: Behavioral multiplier (availability, engagement, responsiveness)."""
    sig = candidate.get("redrob_signals", {})

    score = 0.0

    # Active recently?
    last_active = sig.get("last_active_date", "")
    try:
        la = datetime.strptime(last_active, "%Y-%m-%d").date()
        days_inactive = (date(2026, 6, 28) - la).days
        if days_inactive <= 14:
            activity_score = 1.0
        elif days_inactive <= 30:
            activity_score = 0.85
        elif days_inactive <= 60:
            activity_score = 0.65
        elif days_inactive <= 90:
            activity_score = 0.45
        elif days_inactive <= 180:
            activity_score = 0.25
        else:
            activity_score = 0.05
    except Exception:
        activity_score = 0.3
    score += 0.25 * activity_score

    # Open to work
    open_to_work = sig.get("open_to_work_flag", False)
    score += 0.15 * (1.0 if open_to_work else 0.2)

    # Recruiter response rate
    rrr = sig.get("recruiter_response_rate", 0.5) or 0
    score += 0.20 * rrr

    # Notice period (lower = better for this JD which wants <30 days)
    notice = sig.get("notice_period_days", 60) or 60
    if notice <= 15:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 0.9
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.5
    else:
        notice_score = 0.3
    score += 0.15 * notice_score

    # Profile completeness
    completeness = sig.get("profile_completeness_score", 50) or 0
    score += 0.10 * (completeness / 100.0)

    # Interview completion rate
    icr = sig.get("interview_completion_rate", 0.5) or 0
    score += 0.10 * icr

    # GitHub activity (good signal for AI engineer)
    github = sig.get("github_activity_score", -1)
    if github == -1:
        github_score = 0.3  # neutral if no github
    else:
        github_score = github / 100.0
    score += 0.05 * github_score

    return min(score, 1.0)


def score_education(candidate: dict) -> float:
    """0-1: Education (minor signal for this JD)."""
    edu = candidate.get("education", [])
    if not edu:
        return 0.4

    best = 0.0
    for e in edu:
        tier = e.get("tier", "unknown")
        degree = (e.get("degree", "") + " " + e.get("field_of_study", "")).lower()
        tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6, "tier_4": 0.4, "unknown": 0.5}
        t = tier_map.get(tier, 0.5)

        # CS/ML/AI field bonus
        if any(f in degree for f in ["computer science", "cs", "machine learning",
                                      "artificial intelligence", "electrical", "information technology"]):
            t = min(t + 0.1, 1.0)

        best = max(best, t)

    return best


def anti_signals(candidate: dict) -> float:
    """Returns a penalty multiplier 0.5-1.0 for disqualifying traits."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills_list = candidate.get("skills", [])

    multiplier = 1.0

    # 1. Entire career at consulting firms
    all_companies = [j.get("company", "").lower() for j in career]
    if all_companies and all(any(f in c for f in CONSULTING_FIRMS) for c in all_companies):
        multiplier *= 0.5

    # 2. CV/speech/robotics only profile
    skill_names = [s.get("name", "").lower() for s in skills_list]
    all_text = " ".join(skill_names) + " " + profile.get("summary", "").lower()
    cv_count = sum(1 for k in CV_ONLY_SKILLS if k in all_text)
    retrieval_count = sum(1 for k in MUST_HAVE_SKILLS if k in all_text)
    if cv_count >= 3 and retrieval_count < 2:
        multiplier *= 0.6

    # 3. Title-chaser: multiple companies with short tenure and rising titles
    if len(career) >= 4:
        tenures = [j.get("duration_months", 0) or 0 for j in career]
        short_stints = sum(1 for t in tenures if t < 18)
        if short_stints >= 3:
            multiplier *= 0.85

    # 4. Non-technical current title
    title = profile.get("current_title", "").lower()
    non_tech = ["marketing", "sales executive", "business development", "content ",
                "seo", "finance", "hr manager", "talent acquisition", "recruiter"]
    if any(t in title for t in non_tech):
        multiplier *= 0.4

    # 5. Non-India, not willing to relocate (JD is India-only, no visa sponsorship)
    country = profile.get("country", "").lower()
    will_relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)
    if country and "india" not in country and not will_relocate:
        multiplier *= 0.35

    return multiplier


def compute_score(candidate: dict) -> tuple[float, str]:
    """Returns (final_score 0-1, reasoning string)."""

    honeypot_penalty = detect_honeypot(candidate)
    if honeypot_penalty > 0.6:
        return 0.01 * (1 - honeypot_penalty), "Profile has inconsistencies suggesting a honeypot (skill proficiency vs duration mismatch, timeline anomalies)."

    # Component scores
    s_skills = score_skills(candidate)
    s_career = score_career(candidate)
    s_exp = score_experience(candidate)
    s_loc = score_location(candidate)
    s_behav = score_behavioral_signals(candidate)
    s_edu = score_education(candidate)

    # Weighted composite
    raw = (
        0.30 * s_skills +
        0.28 * s_career +
        0.15 * s_exp +
        0.12 * s_behav +
        0.10 * s_loc +
        0.05 * s_edu
    )

    # Anti-signal multiplier
    penalty = anti_signals(candidate)
    final = raw * penalty * (1 - 0.5 * honeypot_penalty)

    # Build reasoning
    profile = candidate.get("profile", {})
    signals_d = candidate.get("redrob_signals", {})
    cid = candidate.get("candidate_id", "")
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", "?")
    location = profile.get("location", "")
    notice = signals_d.get("notice_period_days", "?")
    rrr = signals_d.get("recruiter_response_rate", "?")

    # Top skills for reasoning
    top_skills = [s["name"] for s in sorted(
        candidate.get("skills", []),
        key=lambda x: ({"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(x.get("proficiency", ""), 0), x.get("endorsements", 0)),
        reverse=True
    )[:3]]

    concerns = []
    if s_loc < 0.5:
        concerns.append("non-India location")
    if notice and isinstance(notice, int) and notice > 60:
        concerns.append(f"{notice}d notice")
    if isinstance(rrr, float) and rrr < 0.3:
        concerns.append("low recruiter response rate")
    if s_career < 0.4:
        concerns.append("limited product-company AI/ML experience")

    concern_str = "; concerns: " + ", ".join(concerns) if concerns else ""

    skills_str = ", ".join(top_skills) if top_skills else "unclear skills"
    reasoning = (
        f"{yoe}yr {title} at {company} ({location}); top skills: {skills_str}; "
        f"skill/career fit {s_skills:.2f}/{s_career:.2f}, notice {notice}d{concern_str}."
    )

    return min(final, 1.0), reasoning[:300]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="./candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    print(f"Loaded {len(candidates)} candidates. Scoring...")

    scored = []
    for i, cand in enumerate(candidates):
        score, reasoning = compute_score(cand)
        scored.append((cand["candidate_id"], score, reasoning))
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(candidates)}...")

    # Sort descending by score
    scored.sort(key=lambda x: (-x[1], x[0]))

    # Take top 100
    top100 = scored[:100]

    # Write CSV
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, reasoning) in enumerate(top100, start=1):
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    print(f"Written {out_path}")

    # Quick sanity
    prev_score = float("inf")
    for rank, (cid, score, _) in enumerate(top100, 1):
        assert score <= prev_score + 1e-9, f"Score not monotone at rank {rank}"
        prev_score = score
    print("Monotone check passed. Run validate_submission.py to confirm format.")


if __name__ == "__main__":
    main()