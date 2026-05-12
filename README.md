# Telecom Churn Reduction Agent

Streamlit demo of a 3-agent churn reduction pipeline for telecom retention teams.

> **Demo data only. Not for production retention decisions.**

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add GROQ_API_KEY to .env (optional — Ollama fallback works without it)
python data/generate.py
streamlit run app.py
```

## Architecture

See `docs/superpowers/specs/2026-05-12-telecom-churn-agent-design.md`.

## Ethics

- All data is synthetic. No real customer information.
- Customer name and ID are never used as scoring inputs.
- No sensitive attributes (caste, religion, gender, age) in the dataset.
- Scripts must not use pressure, urgency, or guilt tactics.
- All agent outputs include a `rationale` / `justification` field for explainability.
- Bucket cutoffs and methodology are visible in the app sidebar.
