# Redrob Hackathon — Ranker

## Quick Start

```bash
pip install -r requirements.txt   # no extra deps beyond stdlib
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
```

Runtime: ~50s on CPU (100K candidates). No GPU, no network calls.

## Architecture

Rule-based multi-component scorer — no LLM API calls, no GPU required.

### Scoring Components (weighted)

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Skills match | 30% | Proficiency × trust (endorsements + duration) across must-have and nice-to-have skills |
| Career fit | 28% | Product company history, AI/ML titles, retrieval/ranking evidence in job descriptions |
| Experience | 15% | Years of experience (5-9 ideal per JD) |
| Behavioral signals | 12% | Recency, open_to_work, recruiter response rate, notice period |
| Location | 10% | Noida/Pune > major Indian cities > India > willing to relocate |
| Education | 5% | Institution tier + CS/ML field bonus |

### Anti-signals (multiplicative penalties)
- Entire career at consulting firms (TCS/Infosys/Wipro etc.): ×0.5
- CV/speech/robotics-only profile with no retrieval exposure: ×0.6
- Non-technical current title (marketing, sales, etc.): ×0.4
- International location, not willing to relocate: ×0.35
- Excessive job-hopping (3+ stints <18 months): ×0.85

### Honeypot Detection
Profiles with inconsistencies (skill proficiency vs duration mismatch, expert skills with 0 endorsements, impossible career timelines, non-tech titles with AI keyword stuffing) are scored near 0 and excluded from top 100.

## Key Design Decisions

1. **Career descriptions over keyword lists** — skills listed are checked against actual job descriptions for trust/endorsement backing. A "expert" skill with 0 endorsements and 0 duration months is penalized.

2. **Product company preference** — the JD explicitly wants non-consulting backgrounds. Company names and industries are checked against a consulting firm list.

3. **Behavioral signals as multiplier** — a perfect-on-paper candidate who hasn't logged in for 6 months or has 5% response rate ranks lower than a slightly-less-perfect but engaged candidate.

4. **No API calls** — fully offline, CPU-only, runs in <60 seconds.

## Interactive Demo (Sandbox)

```bash
pip install streamlit pandas
streamlit run sandbox/app.py
```

Upload a small JSONL sample (≤200 candidates), paste a JD, and explore
ranked results interactively.  The demo runs features → honeypot → scoring →
reasoning (retrieval and MMR diversity are skipped for uploaded samples since
precomputed artifacts are built for the full 100K set).

### Deployment

**Target platform:** [Streamlit Community Cloud](https://streamlit.io/cloud)

The app is deployable as-is — point Streamlit Cloud at this repo with
`sandbox/app.py` as the entrypoint and `requirements.txt` for dependencies.
No secrets or environment variables required.

## Security

Please refer to our [Security Policy](SECURITY.md) for information on reporting vulnerabilities and supported versions.