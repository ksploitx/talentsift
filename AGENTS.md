You're building a candidate ranking system for a hackathon submission. Follow these rules for every phase of this build, no exceptions:

1. Before writing code, restate your understanding of the phase's scope in 2-3 sentences and list the exact files you'll create or modify.
2. After finishing each subphase, append an entry to CHANGELOG.md at the repo root: date, phase and subphase number, files touched, a 2-4 sentence description of what you actually implemented, and any deviations from spec with your reasoning.
3. Do not run pytest, unittest, or any test file. Do not execute validate_submission.py. Do not run rank.py end-to-end and report that it "works." You may write test files as deliverables but must never execute them or report results from running them. I verify everything myself.
4. Stop once all subphases in the current phase are done. Summarize what you built, list the files, and wait for me to say "continue" before starting the next phase.
5. No hardcoded weights, thresholds, or model names in src/ — everything tunable lives in config.yaml.
6. No LLM API calls anywhere in src/, precompute/, or sandbox/. reasoning.py builds strings from real feature values with templates, nothing else.
7. Everything in precompute/ can take as long as it needs. Everything rank.py touches at runtime must be fast — no index-building, no embedding generation, no model training inside rank.py. It only loads cached artifacts.
8. Dont create too much boiler plate and check for early existing boilerplate and remove it.
