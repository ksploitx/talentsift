"""Pre-compute the IsolationForest honeypot detection model.

Loads all 100k candidates from candidates.jsonl, extracts the numeric
feature table via src/features.extract_features(), trains an IsolationForest
(scikit-learn) on the full table, and serializes the model + feature name
list to precompute/honeypot_iforest.pkl.

This script can take as long as it needs — it runs offline.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import IsolationForest

# Allow imports from repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import extract_features, reset_config_cache

CONFIG_PATH = ROOT / "config.yaml"
CANDIDATES_PATH = ROOT / "data" / "candidates.jsonl"
OUTPUT_PATH = ROOT / "precompute" / "honeypot_iforest.pkl"


def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# Numeric features to feed into IsolationForest.
# Excludes string categoricals (notice_period_bucket, preferred_work_mode)
# which can't be used directly.
NUMERIC_FEATURE_NAMES = [
    "years_of_experience",
    "experience_fit_score",
    "company_size_score",
    "avg_tenure_months",
    "short_tenure_ratio",
    "career_entry_count",
    "location_tier",
    "notice_period_days",
    "recruiter_response_rate",
    "days_since_active",
    "platform_engagement_score",
    "verification_score",
    "skill_count_raw",
    "skill_count_by_proficiency",
    "certification_count",
    "github_activity_score",
    "education_tier_score",
    "expected_salary_min_lpa",
    "expected_salary_max_lpa",
    "avg_response_time_hours",
    "offer_acceptance_rate",
    # Boolean features (converted to 0/1).
    "is_product_company",
    "consulting_firm_flag",
    "consulting_only_career",
    "open_to_work",
    "willing_to_relocate",
]


def main():
    cfg = _load_config()

    # Reset feature config cache so it loads fresh.
    reset_config_cache()

    print(f"Loading candidates from {CANDIDATES_PATH} ...")
    rows: list[list[float]] = []
    count = 0

    with open(CANDIDATES_PATH, "r") as f:
        for i, line in enumerate(f):
            candidate = json.loads(line)
            features = extract_features(candidate, str(CONFIG_PATH))

            row = []
            for fname in NUMERIC_FEATURE_NAMES:
                val = features.get(fname, 0)
                if isinstance(val, bool):
                    val = int(val)
                if not isinstance(val, (int, float)):
                    val = 0
                row.append(float(val))

            rows.append(row)
            count += 1

            if count % 10_000 == 0:
                print(f"  Extracted features for {count} candidates")

    print(f"Total candidates: {count}")

    X = np.array(rows, dtype=np.float32)
    print(f"Feature matrix shape: {X.shape}")

    # IsolationForest params from config.
    if_params = cfg.get("honeypot_isolation_forest", {})
    n_estimators = if_params.get("n_estimators", 200)
    contamination = if_params.get("contamination", 0.005)
    random_state = if_params.get("random_state", 42)

    print(
        f"Training IsolationForest (n_estimators={n_estimators}, "
        f"contamination={contamination}) ..."
    )

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    # Serialize model + feature names.
    artifact = {
        "model": model,
        "feature_names": NUMERIC_FEATURE_NAMES,
        "n_candidates_trained": count,
        "contamination": contamination,
    }

    print(f"Saving model to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Done. Model file: {size_mb:.1f} MB")

    # Quick summary: how many flagged as anomalous.
    predictions = model.predict(X)
    n_anomalies = int((predictions == -1).sum())
    print(f"Anomalies detected: {n_anomalies} / {count} ({100*n_anomalies/count:.2f}%)")


if __name__ == "__main__":
    main()
