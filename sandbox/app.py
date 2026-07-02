"""Interactive sandbox app for exploring candidate rankings.

Streamlit UI: upload a small JSONL sample (≤200 candidates), paste JD
text, and run the ranking pipeline.  Displays results as a sortable table
with a CSV download button.

Launch from repo root:
    streamlit run sandbox/app.py

The app imports scoring logic directly from src/ — no duplication.
Retrieval and MMR diversity are skipped because their precomputed artifacts
(BM25 index, embeddings) are built for the full 100K candidate set and
candidate IDs in a small upload won't match.  Instead the pipeline runs:
  features → honeypot → composite scoring → rank by score → reasoning.
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure repo root is on the path so `from src.X import ...` works
# regardless of where Streamlit launches from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import extract_features  # noqa: E402
from src.honeypot import compute_honeypot_score  # noqa: E402
from src.score import compute_composite_scores  # noqa: E402
from src.reasoning import generate_reasoning  # noqa: E402

MAX_CANDIDATES = 200


# ---------------------------------------------------------------------------
# Pipeline — mirrors rank.py phases 2,4,5,7 (skips 3 retrieval + 6 MMR)
# ---------------------------------------------------------------------------

def run_pipeline(candidates: list[dict]) -> pd.DataFrame:
    """Score, rank, and annotate a small set of candidates."""

    candidates_features: list[tuple[str, dict]] = []
    honeypot_scores: dict[str, float] = {}
    candidate_map: dict[str, dict] = {}

    for cand in candidates:
        cid = cand.get("candidate_id", "unknown")
        candidate_map[cid] = cand
        feats = extract_features(cand)
        candidates_features.append((cid, feats))
        hp_res = compute_honeypot_score(cand, feats)
        honeypot_scores[cid] = hp_res.honeypot_score

    # No retrieval scores for uploaded samples — pass zeros
    retrieval_scores: dict[str, float] = {
        cid: 0.0 for cid, _ in candidates_features
    }

    scored = compute_composite_scores(
        candidates_features=candidates_features,
        retrieval_scores=retrieval_scores,
        honeypot_scores=honeypot_scores,
    )
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    feat_lookup = {cid: feats for cid, feats in candidates_features}

    rows = []
    for rank, entry in enumerate(scored, start=1):
        cid = entry["candidate_id"]
        score = entry["final_score"]
        cand = candidate_map.get(cid, {})
        feats = feat_lookup.get(cid, {})
        feats["honeypot_penalty"] = honeypot_scores.get(cid, 0.0)
        # Add mmr_rank for reasoning template compatibility
        entry["mmr_rank"] = rank
        reasoning = generate_reasoning(cand, feats, score_row=entry)

        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 4),
            "reasoning": reasoning,
            "honeypot_penalty": round(honeypot_scores.get(cid, 0.0), 4),
            "structured_fit": entry.get("norm_structured_fit", 0.0),
            "behavioral": entry.get("norm_behavioral", 0.0),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(raw_bytes: bytes) -> list[dict]:
    """Parse uploaded JSONL bytes into a list of dicts."""
    text = raw_bytes.decode("utf-8")
    candidates = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            candidates.append(json.loads(line))
    return candidates


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert dataframe to submission-format CSV bytes."""
    buf = io.BytesIO()
    df[["candidate_id", "rank", "score", "reasoning"]].to_csv(
        buf, index=False
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="TalentSift — Candidate Ranker",
        page_icon="🎯",
        layout="wide",
    )

    # --- Custom styling ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .block-container { padding-top: 2rem; }

        .stDataFrame { border-radius: 8px; overflow: hidden; }
        .stDataFrame th { background: #1a1a2e !important; color: #e0e0ff !important; }

        div[data-testid="stFileUploader"] {
            border: 2px dashed #4a4aff;
            border-radius: 12px;
            padding: 1rem;
        }

        .metric-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #2a2a4a;
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        }
        .metric-card h3 { color: #7b7bff; margin: 0 0 0.3rem 0; font-size: 0.85rem; }
        .metric-card p  { color: #ffffff; margin: 0; font-size: 1.6rem; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Header ---
    st.markdown("# 🎯 TalentSift — Candidate Ranker")
    st.caption(
        "Upload a JSONL sample (≤200 candidates), paste the job description, "
        "and rank candidates in seconds.  "
        "Uses the same scoring pipeline as `rank.py` — features, honeypot detection, "
        "composite scoring, and reasoning generation."
    )

    st.divider()

    # --- Inputs (two columns) ---
    col_upload, col_jd = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("📄 Candidate Data")
        uploaded = st.file_uploader(
            "Upload a `.jsonl` file with candidate profiles",
            type=["jsonl"],
            help=f"Max {MAX_CANDIDATES} candidates. Larger files are truncated.",
        )

    with col_jd:
        st.subheader("📋 Job Description")
        jd_text = st.text_area(
            "Paste the job description text (used for context only — retrieval "
            "is skipped for uploaded samples)",
            height=180,
            placeholder="e.g. Looking for a Senior ML Engineer with 5-9 years "
            "of experience in NLP, Information Retrieval, and Ranking ...",
        )

    st.divider()

    # --- Run button ---
    run_disabled = uploaded is None
    if st.button(
        "🚀 Rank Candidates",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    ):
        raw = uploaded.getvalue()
        try:
            candidates = load_jsonl(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            st.error(f"Failed to parse JSONL: {exc}")
            return

        if not candidates:
            st.warning("Uploaded file contains no candidate records.")
            return

        if len(candidates) > MAX_CANDIDATES:
            st.info(
                f"File contains {len(candidates)} candidates — "
                f"truncating to {MAX_CANDIDATES}."
            )
            candidates = candidates[:MAX_CANDIDATES]

        with st.spinner("Running ranking pipeline …"):
            df = run_pipeline(candidates)

        st.session_state["results"] = df

    # --- Results ---
    if "results" in st.session_state:
        df = st.session_state["results"]

        st.markdown("## 📊 Results")

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-card"><h3>Candidates</h3>'
                f'<p>{len(df)}</p></div>',
                unsafe_allow_html=True,
            )
        with m2:
            top_score = df["score"].max() if len(df) else 0
            st.markdown(
                f'<div class="metric-card"><h3>Top Score</h3>'
                f'<p>{top_score:.4f}</p></div>',
                unsafe_allow_html=True,
            )
        with m3:
            flagged = (df["honeypot_penalty"] > 0.3).sum()
            st.markdown(
                f'<div class="metric-card"><h3>Honeypot Flags</h3>'
                f'<p>{flagged}</p></div>',
                unsafe_allow_html=True,
            )
        with m4:
            median = df["score"].median() if len(df) else 0
            st.markdown(
                f'<div class="metric-card"><h3>Median Score</h3>'
                f'<p>{median:.4f}</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Table
        st.dataframe(
            df,
            use_container_width=True,
            height=480,
            column_config={
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "score": st.column_config.NumberColumn(
                    "Score", format="%.4f"
                ),
                "honeypot_penalty": st.column_config.ProgressColumn(
                    "Honeypot", min_value=0, max_value=1, format="%.2f"
                ),
                "structured_fit": st.column_config.ProgressColumn(
                    "Fit", min_value=0, max_value=1, format="%.2f"
                ),
                "behavioral": st.column_config.ProgressColumn(
                    "Behavioral", min_value=0, max_value=1, format="%.2f"
                ),
                "reasoning": st.column_config.TextColumn(
                    "Reasoning", width="large"
                ),
            },
        )

        # Download button
        csv_bytes = to_csv_bytes(df)
        st.download_button(
            label="⬇️  Download submission CSV",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
            type="primary",
        )


if __name__ == "__main__":
    main()
