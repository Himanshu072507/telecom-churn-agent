# Telecom Churn Reduction Agent

A Streamlit demo of a 3-agent churn reduction pipeline for telecom retention teams.

> **Demo data only. Not for production retention decisions.**

## What it does

1. **Agent 1 — Data Analyst** scores all 200 synthetic customers and assigns each to a churn bucket (Safe / Watch / At-Risk / Critical).
2. **Agent 3 — Executor** generates a personalized, guardrail-compliant retention offer for at-risk customers.
3. **Agent 2 — Voice Agent** writes a warm, ethical retention call script that references the offer, and plays it back via text-to-speech.

A rule-based orchestrator gates which agents run based on the customer's bucket.

## Quick start

```bash
git clone <repo>
cd telecom-churn-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Optional: add GROQ_API_KEY to .env. Without it, the app uses local Ollama.

python data/generate.py
streamlit run app.py
```

## LLM provider

- **Groq** (primary): set `GROQ_API_KEY` in `.env`. Default model: `llama-3.3-70b-versatile`.
- **Ollama** (fallback): requires `llama3.1:8b` pulled locally (`ollama pull llama3.1:8b`).

If Groq fails or no key is set, the app automatically uses Ollama.

## Architecture

See `docs/superpowers/specs/2026-05-12-telecom-churn-agent-design.md`.

## Guardrails

**Telecom-specific:**
- Offer value capped at 3× monthly ARPU.
- Offer validity capped at 90 days.
- Premium customers cannot receive plan downgrades.
- Prepaid ↔ postpaid benefit integrity enforced.
- Offer types restricted to 5 telecom categories.

**Ethical:**
- Customer name and ID are never used as scoring inputs.
- No sensitive attributes (caste, religion, gender, age) in the dataset.
- Scripts must not name competitor telcos.
- Scripts must not use pressure, urgency, or guilt.
- All agent outputs include a `rationale` / `justification` field.
- Bucket cutoffs and methodology are visible in the app sidebar.

## Testing

```bash
pytest -v
```

Tests are unit tests with mocked LLM calls. One end-to-end smoke test hits real Groq if `GROQ_API_KEY` is set (otherwise skipped).

## Project layout

```
telecom-churn-agent/
├── app.py                    Streamlit UI
├── orchestrator.py           Rule-based gating
├── llm.py                    Groq + Ollama client
├── guardrails.py             Validation rules
├── schemas.py                Pydantic models
├── agents/
│   ├── analyst.py            Agent 1
│   ├── voice.py              Agent 2
│   └── executor.py           Agent 3
├── data/
│   ├── customers.csv         200 synthetic rows
│   └── generate.py
├── prompts/
└── tests/
```
