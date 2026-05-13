# Evals

Three on-demand evaluation scripts that hit real Groq. Not part of `pytest` (they cost real LLM calls).

## Setup

```bash
cd ~/telecom-churn-agent
source .venv/bin/activate
export GROQ_API_KEY=gsk_...   # required for all evals
```

## 1. Anchor accuracy — `run_anchors.py`

Validates Agent 1's core scoring claim: each of the 8 hand-tuned anchor customers (`C0001`–`C0008`) should land in its designed bucket.

```bash
python -m evals.run_anchors --n 5
```

Pass criterion: ≥80% of runs per anchor land in the designed bucket. C0007/C0008 are forced to CRITICAL via the port-out override, so they should be 100%.

Exit code: non-zero if any non-forced anchor falls below threshold.

## 2. Guardrail catch-rate — `run_offers.py`

Adversarial scenarios designed to tempt Agent 3 into violating telecom guardrails (oversized offers, premium downgrades, prepaid/postpaid wording slips, long-validity offers). Calls the LLM directly (bypassing the production retry+fallback) and runs each output through `validate_offer`.

```bash
python -m evals.run_offers --n 3
```

Reports:
- **Self-compliance rate** — % of raw LLM outputs that pass guardrails (higher = better prompt quality)
- **Caught violations** — % of outputs the guardrail layer rejected (defense-in-depth proof)
- **Parse errors** — malformed JSON or schema failures

## 3. Voice tone sanity — `run_scripts.py`

Generates retention scripts for 3 representative customers (At-Risk prepaid, Critical postpaid premium, At-Risk postpaid with bill issues) and regex-checks each `full_script` for competitor mentions or pressure phrases.

```bash
python -m evals.run_scripts --n 3
```

## Notes

- Evals use the same prompts and guardrails as the production agents — they validate the system, not a parallel implementation.
- Total cost per full run (`--n 3`) is roughly 9 + 15 + 9 = ~30 LLM calls. On Groq's free tier this is negligible.
- Run before any change to prompts, guardrails, or anchor data.
