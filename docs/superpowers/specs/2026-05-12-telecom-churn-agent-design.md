# Telecom Churn Reduction Agent — Design Spec

**Date:** 2026-05-12
**Status:** Approved (pending user review of this document)
**Owner:** Himanshu Rawat
**Project path:** `~/telecom-churn-agent/`

---

## 1. Purpose

A Streamlit-based internal demo platform for a telecom retention team to identify customers at risk of churn and trigger a coordinated response. Three specialized LLM agents work behind a lightweight orchestrator:

1. **Data Analyst Agent** — scores every customer and assigns a churn bucket.
2. **Voice Agent** — generates a personalized retention call script (with browser TTS playback).
3. **Executor Agent** — generates a personalized, guardrail-compliant retention offer.

Build fidelity: **demo / showcase prototype**. Synthetic data, no telephony, no persistent storage.

---

## 2. Non-goals

- Real outbound telephony (no Twilio).
- Inbound voice / STT.
- Persistent storage (Supabase, Postgres). All state is in-memory + session.
- Auth or role-based access.
- Production deployment (runs locally; Streamlit Cloud deploy is a future option).
- Real customer PII. All data is synthetic.

---

## 3. Architecture

```
data/customers.csv (200 synthetic rows)
        │
        ▼
┌──────────────────────┐
│   Streamlit App      │
│       (app.py)       │
└──────────┬───────────┘
           │
           ▼
   Orchestrator (rule-based; orchestrator.py)
           │
           ├─► Agent 1: Data Analyst (Groq → Ollama fallback)
           │      ▸ runs on ALL customers at app load
           │      ▸ output: bucket + risk_score + rationale per customer
           │
           ▼
   Dashboard renders table + alert banner
           │
           ▼
   User clicks "Run Retention Flow" on a row
           │
           ▼
   Orchestrator gates by bucket:
       SAFE     → no further agents
       WATCH    → no further agents (analysis already shown)
       AT_RISK  → Agent 3 then Agent 2 (sequential; Agent 2 references the offer)
       CRITICAL → Agent 3 then Agent 2 + escalate flag
           │
           ▼
   Agent 3: Executor   → personalized offer
           │
           ▼
   Agent 2: Voice Agent → retention script (references offer) + TTS audio
```

**Orchestrator is deliberately rule-based, not LLM-driven.** This keeps the demo deterministic and avoids an extra LLM hop. The "3 agents" in the brief remain the three specialized LLM-powered units.

---

## 4. Tech stack

- **UI:** Streamlit
- **LLM:** Groq (primary, `llama-3.1-8b-instant` or `llama-3.3-70b-versatile`) → Ollama (`llama3.1:8b`) fallback
- **TTS:** `gtts` (Google Text-to-Speech, offline-friendly mp3 output) → `st.audio()`
- **Validation:** Pydantic for agent output schemas
- **Data:** Pandas + bundled synthetic CSV (200 rows, seeded generator)
- **Testing:** Pytest

---

## 5. Agents

### 5.1 Agent 1 — Data Analyst

**Purpose:** Score every customer, assign a bucket, explain why.

**Input:** Customer row (dict).

**Output (Pydantic-validated JSON):**
```json
{
  "customer_id": "C0042",
  "risk_score": 73,
  "bucket": "AT_RISK",
  "top_3_drivers": ["3 complaints in 90d", "data usage falling", "ARPU dropped 40%"],
  "rationale": "Tenure customer showing escalating complaints and declining engagement. Watch for port-out next."
}
```

**Bucket cutoffs (hard-coded for reproducibility):**
- SAFE: 0–30
- WATCH: 31–55
- AT_RISK: 56–80
- CRITICAL: 81–100
- **Override:** `port_out_request_flag == true` forces `CRITICAL` regardless of LLM score.

**Behavior:** Runs once on all 200 customers at app load, with progress bar. Cached via `@st.cache_data` so reruns are instant. "Regenerate analysis" sidebar button clears cache.

**Prompt principles:**
- Reasons only over signals (complaints, usage, ARPU, etc.), never over name or customer_id.
- Returns JSON only; no prose.
- Cites specific drivers in `top_3_drivers`.

---

### 5.2 Agent 2 — Voice Agent

**Purpose:** Generate a personalized retention call script for the retention rep, plus TTS playback.

**Input:** Agent 1's output + customer row + Agent 3's offer output (so script can reference the offer naturally).

**Output:**
```json
{
  "opening_line": "Hi {name}, this is {rep} from {telco}. Do you have a moment?",
  "key_talking_points": ["acknowledge complaints", "thank for tenure", "introduce offer"],
  "full_script": "200-300 word natural script",
  "do_not_say": ["never mention competitor by name", "no aggressive urgency"],
  "estimated_call_duration_sec": 90
}
```

**TTS:** `gtts` generates an mp3 from `full_script`, served via `st.audio()`. User clicks ▶ to play.

**Prompt principles:**
- Warm, non-coercive tone.
- Acknowledges customer's specific pain points (from `top_3_drivers`).
- Naturally introduces the offer from Agent 3.
- No pressure tactics, no urgency manipulation, no guilt.

---

### 5.3 Agent 3 — Executor

**Purpose:** Generate a personalized, guardrail-compliant retention offer.

**Input:** Agent 1's output + customer row.

**Output:**
```json
{
  "offer_type": "DATA_BOOST",
  "offer_details": "Extra 20GB/month free for 3 months",
  "monetary_value_inr": 600,
  "validity_days": 30,
  "justification": "Customer's data usage rising 25% but on basic plan; offer addresses pain without changing commitment.",
  "expected_retention_lift": "moderate"
}
```

**Allowed offer types (closed enum):**
- `DATA_BOOST` — extra data quota
- `BILL_DISCOUNT` — % or flat reduction on bill
- `LOYALTY_UPGRADE` — bonus loyalty points or tier upgrade
- `DEVICE_OFFER` — handset / accessory discount
- `PLAN_UPGRADE` — free plan upgrade for a fixed term

**Constraints (validated post-LLM):**
- `monetary_value_inr ≤ 3 × avg_monthly_arpu_inr`
- `validity_days ≤ 90`
- Premium customers cannot receive plan downgrades.
- Prepaid customers cannot receive postpaid-only benefits, and vice versa.

---

## 6. Orchestrator

`orchestrator.py` exposes:

```python
def gate(bucket: Bucket) -> set[AgentName]:
    if bucket in (Bucket.SAFE, Bucket.WATCH):
        return set()
    return {AgentName.VOICE, AgentName.EXECUTOR}

def should_escalate(bucket: Bucket) -> bool:
    return bucket == Bucket.CRITICAL
```

When the user clicks "Run Retention Flow":
1. Read bucket from cached Agent 1 output.
2. Call `gate(bucket)`.
3. If non-empty: run Agent 3 (offer) first, then Agent 2 (script — needs offer as input). Sequential by design — the spinner copy reflects this ("Generating offer… Generating script…").
4. If `should_escalate`: render the "🚨 Escalate to retention manager" button.

---

## 7. Data model

`data/customers.csv` — 200 rows generated by `data/generate.py` (seeded, reproducible).

| Column | Type | Range / Notes |
|---|---|---|
| `customer_id` | str | C0001–C0200 |
| `name` | str | Indian names (Faker, `en_IN`) |
| `plan_type` | str | prepaid / postpaid |
| `is_premium` | bool | ~15% true |
| `tenure_months` | int | 1–120 |
| `avg_monthly_arpu_inr` | float | 150–2500 |
| `complaints_last_90d` | int | 0–8 |
| `offers_availed_last_180d` | int | 0–5 |
| `data_usage_gb_trend` | str | rising / flat / falling |
| `last_recharge_days_ago` | int | 0–60 (prepaid only; null for postpaid) |
| `bill_payment_delays_count` | int | 0–6 (postpaid only; null for prepaid) |
| `network_issue_tickets` | int | 0–5 |
| `call_drop_rate_pct` | float | 0–15 |
| `last_outage_days_ago` | int | 0–365 |
| `app_logins_last_30d` | int | 0–60 |
| `loyalty_points_balance` | int | 0–10000 |
| `family_plan_members` | int | 0–6 |
| `port_out_request_flag` | bool | ~5% true (forces CRITICAL bucket) |

**Distribution target:** ~50% Safe, 25% Watch, 15% At-Risk, 10% Critical.

**Excluded (intentional):** caste, religion, gender, age, location — no sensitive attributes used for scoring.

---

## 8. UI

Single-page Streamlit app, three zones:

### Zone 1 — Alert banner (top)
- Red strip when At-Risk/Critical counts > 0: `⚠ {n_critical} customers flagged Critical · {n_at_risk} At-Risk · Action recommended`
- Quiet gray when none.

### Zone 2 — Dashboard table
- Sortable/filterable dataframe of all 200 customers.
- Columns: customer_id, name, bucket (color chip), risk_score, tenure_months, avg_monthly_arpu_inr, top driver.
- Filter chips above table: `All | Safe | Watch | At-Risk | Critical | Premium only`.
- Row selection opens Zone 3.

### Zone 3 — Drill-down panel
- **Customer card:** all 15 attributes formatted.
- **Agent 1 output:** bucket chip, risk-score gauge, top 3 drivers, rationale.
- **"Run Retention Flow" button** — enabled only for At-Risk/Critical; tooltip explains why disabled for Safe/Watch.
- On click → spinner → Agents 3 then 2 → results render:
  - **Offer card** (Agent 3): offer_type chip, details, value, validity, justification.
  - **Retention script card** (Agent 2): script text + ▶ Play TTS button + do-not-say checklist.
- For Critical customers: extra "🚨 Escalate to retention manager" button (shows toast in demo).

### Sidebar
- LLM status indicators: "Groq: connected" / "Ollama fallback ready".
- "Regenerate analysis" button.
- "About / methodology" expander explaining bucket cutoffs and guardrails.

### Footer
- Disclaimer: "Demo data only. Not for production retention decisions."

---

## 9. Guardrails

### 9.1 Telecom-specific logic
| Rule | Enforcement |
|---|---|
| Offer types restricted to 5 enum values | Pydantic enum + post-LLM validation |
| Offer value ≤ 3× monthly ARPU | `guardrails.py` post-LLM check |
| Offer validity ≤ 90 days | `guardrails.py` post-LLM check |
| Premium customers cannot get downgrade offers | `guardrails.py` |
| Prepaid ↔ postpaid benefit integrity | `guardrails.py` |
| Scripts must not name competitor telcos | Agent 2 prompt + regex check |
| Scripts must not promise out-of-catalog services | Agent 2 prompt + phrase allowlist |

### 9.2 Ethical
| Rule | Enforcement |
|---|---|
| Name / customer_id never used as scoring input | Agent 1 prompt + schema |
| No sensitive attributes in dataset | Schema design |
| Scripts must not use pressure / urgency / guilt | Agent 2 prompt + `do_not_say` field |
| All outputs include `rationale` / `justification` | Pydantic schema requires it |
| Bucket cutoffs and methodology visible to user | UI sidebar expander |
| Synthetic data clearly labeled | README banner + UI footer disclaimer |

### 9.3 Validation flow
```
LLM raw output
  → JSON parse (with retry-once on parse error)
  → Pydantic schema validation
  → guardrails.py business rules
  → if violation: log + safe fallback (generic offer / generic script)
  → if pass: render in UI
```

---

## 10. File structure

```
telecom-churn-agent/
├── app.py                       # Streamlit entrypoint (~150 lines)
├── orchestrator.py              # rule-based gating (~50 lines)
├── llm.py                       # Groq → Ollama fallback (~80 lines)
├── guardrails.py                # validation rules (~120 lines)
├── agents/
│   ├── __init__.py
│   ├── analyst.py               # Agent 1 (~100 lines)
│   ├── voice.py                 # Agent 2 (~80 lines)
│   └── executor.py              # Agent 3 (~80 lines)
├── data/
│   ├── customers.csv            # 200 synthetic rows
│   └── generate.py              # seeded CSV generator
├── prompts/
│   ├── analyst.txt
│   ├── voice.txt
│   └── executor.txt
├── tests/
│   ├── test_guardrails.py
│   ├── test_orchestrator.py
│   └── test_agents.py
├── .env.example                 # GROQ_API_KEY=
├── requirements.txt
├── README.md                    # setup, ethics statement, methodology
└── .gitignore
```

---

## 11. Testing strategy

- **Unit tests, no LLM calls (~20):**
  - `guardrails.py`: offer-value cap, validity cap, plan-type integrity, premium-downgrade rejection, competitor-mention regex, out-of-catalog phrase detection.
  - `orchestrator.py`: gating per bucket, escalation flag.
  - Bucket cutoff math + port-out override.
- **Schema tests with mocked LLM (~5):**
  - Each agent's output parses cleanly into its Pydantic model on a fixture response.
  - Fallback path triggers when LLM returns malformed JSON twice.
- **One end-to-end smoke test:** all 3 agents on a single fixture customer with real Groq call. Skipped (not failed) if `GROQ_API_KEY` is unset.

Target: ~25 tests, all green before declaring done.

---

## 12. Success criteria

The demo is "done" when:
1. App loads, Agent 1 runs on 200 customers, dashboard renders with realistic bucket distribution.
2. Alert banner shows correct counts.
3. Clicking an At-Risk row → "Run Retention Flow" → offer + script appear within 5–10s.
4. Clicking ▶ plays the script audibly via browser.
5. Critical customers show the escalate button; Safe/Watch customers cannot trigger the retention flow.
6. All guardrails fire correctly on malformed LLM output (verified by injecting a bad fixture).
7. Ollama fallback engages when Groq env var is unset (verified manually).
8. All 25 tests pass.
9. README contains setup steps, ethics statement, and methodology link.

---

## 13. Open questions

None at spec time. Surface during planning.
