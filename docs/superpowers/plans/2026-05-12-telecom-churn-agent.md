# Telecom Churn Reduction Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit demo with 3 LLM agents (Data Analyst, Voice, Executor) coordinated by a rule-based orchestrator, identifying telecom customers at risk of churn and producing retention scripts + offers, enforced by telecom-specific and ethical guardrails.

**Architecture:** Single-page Streamlit app reads a bundled synthetic CSV (200 rows). Agent 1 scores all rows at load. User drills into a row and clicks "Run Retention Flow" — for At-Risk/Critical buckets the rule-based orchestrator calls Agent 3 (offer), then Agent 2 (script that references the offer). All agent outputs are Pydantic-validated and pass through `guardrails.py` checks. LLM uses Groq primary with Ollama (`llama3.1:8b`) fallback.

**Tech Stack:** Streamlit, Groq, Ollama, Pydantic, Pandas, gTTS, Faker, Pytest.

**Spec:** `docs/superpowers/specs/2026-05-12-telecom-churn-agent-design.md`

---

## Pre-flight

All paths in this plan are relative to `~/telecom-churn-agent/`. The directory already exists. Tasks assume `cd ~/telecom-churn-agent && git init` has been run before Task 1 (or do it in Task 1).

Python 3.10+ required. Verify with `python3 --version`.

Ollama is expected to be installed with `llama3.1:8b` pulled (per user confirmation). Verify with `ollama list | grep llama3.1`.

---

## Task 1: Project scaffolding

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `README.md` (skeleton)
- Create: `agents/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`

- [ ] **Step 1: Create directory structure**

```bash
cd ~/telecom-churn-agent
mkdir -p agents data prompts tests
git init
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.pytest_cache/
.env
.venv/
venv/
*.mp3
.streamlit/secrets.toml
.DS_Store
```

- [ ] **Step 3: Write `requirements.txt`**

```
streamlit>=1.32
groq>=0.11
ollama>=0.3
pydantic>=2.6
pandas>=2.2
gtts>=2.5
faker>=24.0
pytest>=8.0
python-dotenv>=1.0
```

- [ ] **Step 4: Write `.env.example`**

```
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
OLLAMA_MODEL=llama3.1:8b
```

- [ ] **Step 5: Write skeleton `README.md`**

```markdown
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
```

- [ ] **Step 6: Write `tests/conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 7: Create empty `__init__.py` files**

```bash
touch agents/__init__.py tests/__init__.py
```

- [ ] **Step 8: Install dependencies and verify**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest --version
```

Expected: pytest version printed, no errors.

- [ ] **Step 9: Commit**

```bash
git add .gitignore requirements.txt .env.example README.md agents/ tests/ docs/
git commit -m "feat: scaffold telecom-churn-agent project"
```

---

## Task 2: Schemas and enums

**Files:**
- Create: `schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError

from schemas import (
    Bucket, OfferType, PlanType, DataTrend,
    Customer, AnalystOutput, OfferOutput, VoiceOutput,
)


def test_bucket_enum_has_four_values():
    assert {b.value for b in Bucket} == {"SAFE", "WATCH", "AT_RISK", "CRITICAL"}


def test_offer_type_enum_has_five_values():
    assert {o.value for o in OfferType} == {
        "DATA_BOOST", "BILL_DISCOUNT", "LOYALTY_UPGRADE",
        "DEVICE_OFFER", "PLAN_UPGRADE",
    }


def test_customer_validates_a_realistic_row():
    row = {
        "customer_id": "C0001",
        "name": "Asha Iyer",
        "plan_type": "prepaid",
        "is_premium": False,
        "tenure_months": 24,
        "avg_monthly_arpu_inr": 350.0,
        "complaints_last_90d": 1,
        "offers_availed_last_180d": 0,
        "data_usage_gb_trend": "rising",
        "last_recharge_days_ago": 3,
        "bill_payment_delays_count": None,
        "network_issue_tickets": 0,
        "call_drop_rate_pct": 1.2,
        "last_outage_days_ago": 90,
        "app_logins_last_30d": 12,
        "loyalty_points_balance": 1200,
        "family_plan_members": 0,
        "port_out_request_flag": False,
    }
    customer = Customer(**row)
    assert customer.customer_id == "C0001"
    assert customer.plan_type == PlanType.PREPAID


def test_analyst_output_rejects_score_above_100():
    with pytest.raises(ValidationError):
        AnalystOutput(
            customer_id="C0001",
            risk_score=150,
            bucket=Bucket.AT_RISK,
            top_3_drivers=["a", "b", "c"],
            rationale="x",
        )


def test_offer_output_rejects_negative_value():
    with pytest.raises(ValidationError):
        OfferOutput(
            offer_type=OfferType.DATA_BOOST,
            offer_details="x",
            monetary_value_inr=-100,
            validity_days=30,
            justification="x",
            expected_retention_lift="moderate",
        )


def test_voice_output_requires_do_not_say_list():
    vo = VoiceOutput(
        opening_line="Hi",
        key_talking_points=["a", "b"],
        full_script="...",
        do_not_say=["never name competitors"],
        estimated_call_duration_sec=90,
    )
    assert "never name competitors" in vo.do_not_say
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_schemas.py -v
```

Expected: ImportError — module `schemas` not found.

- [ ] **Step 3: Implement `schemas.py`**

```python
"""Pydantic models and enums shared across agents."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, conint, confloat


class Bucket(str, Enum):
    SAFE = "SAFE"
    WATCH = "WATCH"
    AT_RISK = "AT_RISK"
    CRITICAL = "CRITICAL"


class OfferType(str, Enum):
    DATA_BOOST = "DATA_BOOST"
    BILL_DISCOUNT = "BILL_DISCOUNT"
    LOYALTY_UPGRADE = "LOYALTY_UPGRADE"
    DEVICE_OFFER = "DEVICE_OFFER"
    PLAN_UPGRADE = "PLAN_UPGRADE"


class PlanType(str, Enum):
    PREPAID = "prepaid"
    POSTPAID = "postpaid"


class DataTrend(str, Enum):
    RISING = "rising"
    FLAT = "flat"
    FALLING = "falling"


class RetentionLift(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class Customer(BaseModel):
    customer_id: str
    name: str
    plan_type: PlanType
    is_premium: bool
    tenure_months: conint(ge=0, le=240)
    avg_monthly_arpu_inr: confloat(ge=0)
    complaints_last_90d: conint(ge=0)
    offers_availed_last_180d: conint(ge=0)
    data_usage_gb_trend: DataTrend
    last_recharge_days_ago: Optional[conint(ge=0)] = None
    bill_payment_delays_count: Optional[conint(ge=0)] = None
    network_issue_tickets: conint(ge=0)
    call_drop_rate_pct: confloat(ge=0, le=100)
    last_outage_days_ago: conint(ge=0)
    app_logins_last_30d: conint(ge=0)
    loyalty_points_balance: conint(ge=0)
    family_plan_members: conint(ge=0)
    port_out_request_flag: bool


class AnalystOutput(BaseModel):
    customer_id: str
    risk_score: conint(ge=0, le=100)
    bucket: Bucket
    top_3_drivers: list[str] = Field(..., min_length=1, max_length=5)
    rationale: str = Field(..., min_length=10)


class OfferOutput(BaseModel):
    offer_type: OfferType
    offer_details: str = Field(..., min_length=5)
    monetary_value_inr: conint(ge=0)
    validity_days: conint(ge=1, le=365)
    justification: str = Field(..., min_length=10)
    expected_retention_lift: RetentionLift


class VoiceOutput(BaseModel):
    opening_line: str = Field(..., min_length=5)
    key_talking_points: list[str] = Field(..., min_length=1)
    full_script: str = Field(..., min_length=50)
    do_not_say: list[str] = Field(..., min_length=1)
    estimated_call_duration_sec: conint(ge=15, le=600)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_schemas.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add schemas.py tests/test_schemas.py
git commit -m "feat: add Pydantic schemas for agents and customer data"
```

---

## Task 3: Synthetic data generator

**Files:**
- Create: `data/generate.py`
- Create: `data/customers.csv` (generated output, committed)
- Create: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data.py`:

```python
import pandas as pd
import pytest
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "customers.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)


def test_csv_has_200_rows(df):
    assert len(df) == 200


def test_csv_has_all_expected_columns(df):
    expected = {
        "customer_id", "name", "plan_type", "is_premium",
        "tenure_months", "avg_monthly_arpu_inr",
        "complaints_last_90d", "offers_availed_last_180d",
        "data_usage_gb_trend",
        "last_recharge_days_ago", "bill_payment_delays_count",
        "network_issue_tickets", "call_drop_rate_pct",
        "last_outage_days_ago", "app_logins_last_30d",
        "loyalty_points_balance", "family_plan_members",
        "port_out_request_flag",
    }
    assert set(df.columns) == expected


def test_prepaid_customers_have_no_bill_delays(df):
    prepaid = df[df["plan_type"] == "prepaid"]
    assert prepaid["bill_payment_delays_count"].isna().all()


def test_postpaid_customers_have_no_last_recharge(df):
    postpaid = df[df["plan_type"] == "postpaid"]
    assert postpaid["last_recharge_days_ago"].isna().all()


def test_port_out_flag_in_5_to_10_percent_range(df):
    pct = df["port_out_request_flag"].mean() * 100
    assert 3 <= pct <= 12


def test_premium_flag_in_10_to_25_percent_range(df):
    pct = df["is_premium"].mean() * 100
    assert 10 <= pct <= 25
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py -v
```

Expected: FileNotFoundError on `customers.csv`.

- [ ] **Step 3: Implement `data/generate.py`**

```python
"""Generate seeded synthetic telecom customer data."""
import random
from pathlib import Path

import pandas as pd
from faker import Faker

SEED = 42
N_ROWS = 200
OUT_PATH = Path(__file__).parent / "customers.csv"


def generate() -> pd.DataFrame:
    random.seed(SEED)
    fake = Faker("en_IN")
    Faker.seed(SEED)

    rows = []
    for i in range(1, N_ROWS + 1):
        plan_type = random.choices(["prepaid", "postpaid"], weights=[0.6, 0.4])[0]
        is_premium = random.random() < 0.15
        tenure = random.randint(1, 120)
        arpu = round(random.uniform(150, 2500), 2)
        port_out = random.random() < 0.05

        complaints = random.choices(
            range(0, 9), weights=[40, 20, 15, 10, 6, 4, 2, 2, 1]
        )[0]
        offers_availed = random.choices(
            range(0, 6), weights=[30, 25, 20, 12, 8, 5]
        )[0]
        trend = random.choices(
            ["rising", "flat", "falling"], weights=[0.35, 0.4, 0.25]
        )[0]
        network_issues = random.choices(
            range(0, 6), weights=[50, 20, 12, 8, 6, 4]
        )[0]
        call_drops = round(random.uniform(0, 15), 1)
        last_outage = random.randint(0, 365)
        app_logins = random.randint(0, 60)
        loyalty = random.randint(0, 10000)
        family = random.choices(range(0, 7), weights=[60, 10, 10, 8, 6, 4, 2])[0]

        row = {
            "customer_id": f"C{i:04d}",
            "name": fake.name(),
            "plan_type": plan_type,
            "is_premium": is_premium,
            "tenure_months": tenure,
            "avg_monthly_arpu_inr": arpu,
            "complaints_last_90d": complaints,
            "offers_availed_last_180d": offers_availed,
            "data_usage_gb_trend": trend,
            "last_recharge_days_ago": random.randint(0, 60) if plan_type == "prepaid" else None,
            "bill_payment_delays_count": random.choices(range(0, 7), weights=[40, 25, 15, 10, 5, 3, 2])[0] if plan_type == "postpaid" else None,
            "network_issue_tickets": network_issues,
            "call_drop_rate_pct": call_drops,
            "last_outage_days_ago": last_outage,
            "app_logins_last_30d": app_logins,
            "loyalty_points_balance": loyalty,
            "family_plan_members": family,
            "port_out_request_flag": port_out,
        }
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
```

- [ ] **Step 4: Run generator and tests**

```bash
python data/generate.py
pytest tests/test_data.py -v
```

Expected: "Wrote 200 rows to .../customers.csv" followed by 6 passed.

- [ ] **Step 5: Commit**

```bash
git add data/generate.py data/customers.csv tests/test_data.py
git commit -m "feat: synthetic telecom customer data (200 rows, seeded)"
```

---

## Task 4: Guardrails — telecom + ethical rules

**Files:**
- Create: `guardrails.py`
- Create: `tests/test_guardrails.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_guardrails.py`:

```python
import pytest

from guardrails import (
    GuardrailViolation,
    validate_offer,
    validate_script,
)
from schemas import OfferOutput, OfferType, VoiceOutput


def base_offer(**overrides):
    defaults = dict(
        offer_type=OfferType.DATA_BOOST,
        offer_details="Extra 10GB for 3 months",
        monetary_value_inr=300,
        validity_days=30,
        justification="Customer data usage trending up.",
        expected_retention_lift="moderate",
    )
    defaults.update(overrides)
    return OfferOutput(**defaults)


def base_script(**overrides):
    defaults = dict(
        opening_line="Hi Asha, thanks for being with us.",
        key_talking_points=["acknowledge complaint", "introduce offer"],
        full_script=(
            "Hi Asha, thanks for being with us for years. I wanted to "
            "check in about the network issues you reported, and share "
            "an offer we have for you."
        ),
        do_not_say=["never name competitors"],
        estimated_call_duration_sec=90,
    )
    defaults.update(overrides)
    return VoiceOutput(**defaults)


CUSTOMER_PREPAID_FREE = {"plan_type": "prepaid", "is_premium": False, "avg_monthly_arpu_inr": 300.0}
CUSTOMER_POSTPAID_PREMIUM = {"plan_type": "postpaid", "is_premium": True, "avg_monthly_arpu_inr": 1500.0}


def test_offer_value_within_3x_arpu_passes():
    offer = base_offer(monetary_value_inr=900)  # exactly 3x 300
    validate_offer(offer, CUSTOMER_PREPAID_FREE)


def test_offer_value_above_3x_arpu_fails():
    offer = base_offer(monetary_value_inr=1000)
    with pytest.raises(GuardrailViolation, match="3x"):
        validate_offer(offer, CUSTOMER_PREPAID_FREE)


def test_offer_validity_above_90_days_fails():
    offer = base_offer(validity_days=120)
    with pytest.raises(GuardrailViolation, match="validity"):
        validate_offer(offer, CUSTOMER_PREPAID_FREE)


def test_premium_customer_plan_downgrade_offer_fails():
    offer = base_offer(offer_type=OfferType.PLAN_UPGRADE, offer_details="downgrade to basic plan for cheaper rates")
    with pytest.raises(GuardrailViolation, match="downgrade"):
        validate_offer(offer, CUSTOMER_POSTPAID_PREMIUM)


def test_prepaid_customer_cannot_receive_postpaid_benefit():
    offer = base_offer(offer_details="Free postpaid bill credit for next month")
    with pytest.raises(GuardrailViolation, match="postpaid"):
        validate_offer(offer, CUSTOMER_PREPAID_FREE)


def test_script_mentioning_competitor_fails():
    script = base_script(full_script="Hi, switching to Jio is a mistake, our network is better.")
    with pytest.raises(GuardrailViolation, match="competitor"):
        validate_script(script)


def test_script_with_urgency_pressure_fails():
    script = base_script(full_script="Hi, you MUST act now or lose this forever, this is your last chance.")
    with pytest.raises(GuardrailViolation, match="pressure"):
        validate_script(script)


def test_clean_script_passes():
    validate_script(base_script())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_guardrails.py -v
```

Expected: ImportError — module `guardrails` not found.

- [ ] **Step 3: Implement `guardrails.py`**

```python
"""Telecom-specific and ethical guardrail checks for agent outputs."""
import re

from schemas import OfferOutput, OfferType, VoiceOutput


class GuardrailViolation(Exception):
    pass


COMPETITOR_PATTERNS = [
    r"\bjio\b", r"\bairtel\b", r"\bvi\b", r"\bvodafone\b", r"\bidea\b",
    r"\bbsnl\b", r"\bmtnl\b",
]

PRESSURE_PATTERNS = [
    r"\bmust act now\b", r"\blast chance\b", r"\blose this forever\b",
    r"\bonly today\b", r"\bif you don't\b.*\bnow\b",
    r"\byou'll regret\b", r"\bdon't be foolish\b",
]


def validate_offer(offer: OfferOutput, customer: dict) -> None:
    """Raise GuardrailViolation if offer breaks any rule."""
    arpu = float(customer["avg_monthly_arpu_inr"])
    if offer.monetary_value_inr > 3 * arpu:
        raise GuardrailViolation(
            f"Offer value {offer.monetary_value_inr} exceeds 3x ARPU ({3 * arpu:.0f})"
        )

    if offer.validity_days > 90:
        raise GuardrailViolation(
            f"Offer validity {offer.validity_days} exceeds 90 days"
        )

    details_lower = offer.offer_details.lower()

    if customer.get("is_premium") and "downgrade" in details_lower:
        raise GuardrailViolation(
            "Premium customers cannot be offered plan downgrades"
        )

    if customer["plan_type"] == "prepaid" and "postpaid" in details_lower:
        raise GuardrailViolation(
            "Prepaid customers cannot receive postpaid-only benefits"
        )

    if customer["plan_type"] == "postpaid" and "recharge" in details_lower:
        raise GuardrailViolation(
            "Postpaid customers do not recharge; offer mentions prepaid-only term"
        )


def validate_script(script: VoiceOutput) -> None:
    """Raise GuardrailViolation if script breaks tone or competitor rules."""
    text = script.full_script.lower()

    for pattern in COMPETITOR_PATTERNS:
        if re.search(pattern, text):
            raise GuardrailViolation(
                f"Script names competitor (matched /{pattern}/)"
            )

    for pattern in PRESSURE_PATTERNS:
        if re.search(pattern, text):
            raise GuardrailViolation(
                f"Script uses pressure tactic (matched /{pattern}/)"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_guardrails.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add guardrails.py tests/test_guardrails.py
git commit -m "feat: telecom + ethical guardrails with 8 test cases"
```

---

## Task 5: Orchestrator — rule-based gating

**Files:**
- Create: `orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_orchestrator.py`:

```python
from orchestrator import gate, should_escalate, AgentName
from schemas import Bucket


def test_safe_bucket_runs_no_agents():
    assert gate(Bucket.SAFE) == set()


def test_watch_bucket_runs_no_agents():
    assert gate(Bucket.WATCH) == set()


def test_at_risk_bucket_runs_voice_and_executor():
    assert gate(Bucket.AT_RISK) == {AgentName.VOICE, AgentName.EXECUTOR}


def test_critical_bucket_runs_voice_and_executor():
    assert gate(Bucket.CRITICAL) == {AgentName.VOICE, AgentName.EXECUTOR}


def test_only_critical_triggers_escalation():
    assert should_escalate(Bucket.CRITICAL) is True
    assert should_escalate(Bucket.AT_RISK) is False
    assert should_escalate(Bucket.WATCH) is False
    assert should_escalate(Bucket.SAFE) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_orchestrator.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `orchestrator.py`**

```python
"""Rule-based gating: decides which agents run for which bucket."""
from enum import Enum

from schemas import Bucket


class AgentName(str, Enum):
    ANALYST = "ANALYST"
    VOICE = "VOICE"
    EXECUTOR = "EXECUTOR"


def gate(bucket: Bucket) -> set[AgentName]:
    """Return the set of follow-up agents to run for a given bucket."""
    if bucket in (Bucket.SAFE, Bucket.WATCH):
        return set()
    return {AgentName.VOICE, AgentName.EXECUTOR}


def should_escalate(bucket: Bucket) -> bool:
    """Critical-bucket customers get the escalation banner."""
    return bucket == Bucket.CRITICAL
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_orchestrator.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat: orchestrator gating by churn bucket"
```

---

## Task 6: LLM client — Groq primary, Ollama fallback

**Files:**
- Create: `llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm.py`:

```python
from unittest.mock import patch, MagicMock

import pytest

from llm import call_llm, LLMError


def test_groq_success_returns_content():
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]
    with patch("llm._groq_client") as mock_groq:
        mock_groq.chat.completions.create.return_value = fake_resp
        out = call_llm("system", "user", expect_json=True)
        assert out == '{"ok": true}'


def test_groq_failure_falls_back_to_ollama():
    with patch("llm._groq_client") as mock_groq, \
         patch("llm._ollama_chat") as mock_ollama:
        mock_groq.chat.completions.create.side_effect = RuntimeError("groq down")
        mock_ollama.return_value = '{"from": "ollama"}'
        out = call_llm("system", "user", expect_json=True)
        assert out == '{"from": "ollama"}'
        mock_ollama.assert_called_once()


def test_both_providers_failing_raises_llm_error():
    with patch("llm._groq_client") as mock_groq, \
         patch("llm._ollama_chat") as mock_ollama:
        mock_groq.chat.completions.create.side_effect = RuntimeError("groq down")
        mock_ollama.side_effect = RuntimeError("ollama down")
        with pytest.raises(LLMError):
            call_llm("system", "user")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `llm.py`**

```python
"""Unified LLM client: Groq primary, Ollama fallback."""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


class LLMError(Exception):
    pass


def _build_groq_client():
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


_groq_client = _build_groq_client()


def _ollama_chat(system: str, user: str, expect_json: bool) -> str:
    import ollama
    fmt = "json" if expect_json else ""
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format=fmt,
        options={"temperature": 0.3},
    )
    return resp["message"]["content"]


def call_llm(system: str, user: str, expect_json: bool = True) -> str:
    """Try Groq first, fall back to Ollama. Raises LLMError if both fail."""
    if _groq_client is not None:
        try:
            kwargs = {"response_format": {"type": "json_object"}} if expect_json else {}
            resp = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                **kwargs,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning("Groq call failed, falling back to Ollama: %s", e)

    try:
        return _ollama_chat(system, user, expect_json)
    except Exception as e:
        raise LLMError(f"Both Groq and Ollama failed: {e}") from e


def provider_status() -> dict:
    """For the sidebar status indicators."""
    return {
        "groq_configured": _groq_client is not None,
        "groq_model": GROQ_MODEL,
        "ollama_model": OLLAMA_MODEL,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_llm.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "feat: LLM client with Groq primary, Ollama fallback"
```

---

## Task 7: Prompts

**Files:**
- Create: `prompts/analyst.txt`
- Create: `prompts/executor.txt`
- Create: `prompts/voice.txt`

- [ ] **Step 1: Write `prompts/analyst.txt`**

```
You are a telecom churn analyst. Analyze the customer data below and return a JSON object.

Rules:
- DO NOT use the customer's name or customer_id in your reasoning. Reason only over behavioral and account signals.
- Score the customer 0-100 (higher = more likely to churn).
- Assign one bucket based on the score:
  SAFE     = 0-30
  WATCH    = 31-55
  AT_RISK  = 56-80
  CRITICAL = 81-100
- If port_out_request_flag is true, return CRITICAL regardless of the score (and score should be >= 85).
- Identify exactly 3 concrete drivers contributing to the score (cite specific signals from the data, with numbers).
- Provide a 1-2 sentence plain-English rationale.

Output JSON only, no prose, matching this shape:
{
  "customer_id": "<string>",
  "risk_score": <int 0-100>,
  "bucket": "SAFE" | "WATCH" | "AT_RISK" | "CRITICAL",
  "top_3_drivers": ["...", "...", "..."],
  "rationale": "..."
}

Customer data:
{customer_json}
```

- [ ] **Step 2: Write `prompts/executor.txt`**

```
You are a telecom retention specialist. Generate ONE retention offer for the customer below.

Inputs:
- Customer analysis: {analyst_json}
- Customer profile: {customer_json}

Hard constraints:
- offer_type MUST be one of: DATA_BOOST, BILL_DISCOUNT, LOYALTY_UPGRADE, DEVICE_OFFER, PLAN_UPGRADE
- monetary_value_inr MUST be <= 3 * avg_monthly_arpu_inr
- validity_days MUST be <= 90
- If the customer is_premium = true, do NOT offer any plan downgrade.
- If the customer is prepaid, do NOT use words like "postpaid bill" or "monthly bill credit". If postpaid, do NOT mention "recharge".

Soft requirements:
- Tie the offer directly to the top_3_drivers from the analysis.
- Pick the offer_type whose mechanism best addresses the strongest driver.

Output JSON only, matching:
{
  "offer_type": "...",
  "offer_details": "human-readable description, 1 sentence",
  "monetary_value_inr": <int>,
  "validity_days": <int 1-90>,
  "justification": "1-2 sentences explaining the choice in terms of drivers",
  "expected_retention_lift": "low" | "moderate" | "high"
}
```

- [ ] **Step 3: Write `prompts/voice.txt`**

```
You are a telecom retention call script writer. Write a warm, ethical retention script for a rep to use on a phone call.

Inputs:
- Customer analysis: {analyst_json}
- Customer profile: {customer_json}
- Offer to present: {offer_json}

Tone & content rules (mandatory):
- Warm, empathetic, conversational. Like a helpful human rep, not a marketing pitch.
- Acknowledge the customer's specific pain points from top_3_drivers.
- Introduce the offer naturally, near the middle/end of the script.
- NO pressure tactics. NO urgency manipulation ("act now", "last chance", "only today").
- NO guilt or shaming.
- NO mention of competitor telco names (Jio, Airtel, Vi, Vodafone, BSNL, etc.) — never compare.
- NO promising services that aren't in the offer.

Output JSON only, matching:
{
  "opening_line": "<short opener, 1 sentence>",
  "key_talking_points": ["...", "...", "..."],
  "full_script": "<200-300 word natural call script the rep can read>",
  "do_not_say": ["<at least one explicit pitfall to avoid for this customer>"],
  "estimated_call_duration_sec": <int 30-300>
}
```

- [ ] **Step 4: Commit**

```bash
git add prompts/
git commit -m "feat: agent prompts (analyst, executor, voice)"
```

---

## Task 8: Agent 1 — Data Analyst

**Files:**
- Create: `agents/analyst.py`
- Create: `tests/test_analyst.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analyst.py`:

```python
import json
from unittest.mock import patch

from agents.analyst import analyze_customer, force_critical_on_port_out
from schemas import Bucket


SAMPLE_CUSTOMER = {
    "customer_id": "C0001",
    "name": "Asha Iyer",
    "plan_type": "prepaid",
    "is_premium": False,
    "tenure_months": 24,
    "avg_monthly_arpu_inr": 350.0,
    "complaints_last_90d": 3,
    "offers_availed_last_180d": 0,
    "data_usage_gb_trend": "falling",
    "last_recharge_days_ago": 30,
    "bill_payment_delays_count": None,
    "network_issue_tickets": 2,
    "call_drop_rate_pct": 4.5,
    "last_outage_days_ago": 10,
    "app_logins_last_30d": 2,
    "loyalty_points_balance": 500,
    "family_plan_members": 0,
    "port_out_request_flag": False,
}


def test_analyzer_parses_valid_llm_response():
    fake_json = json.dumps({
        "customer_id": "C0001",
        "risk_score": 65,
        "bucket": "AT_RISK",
        "top_3_drivers": ["3 complaints in 90d", "data usage falling", "low app engagement"],
        "rationale": "Engaged tenure customer signaling dissatisfaction across multiple axes.",
    })
    with patch("agents.analyst.call_llm", return_value=fake_json):
        out = analyze_customer(SAMPLE_CUSTOMER)
    assert out.bucket == Bucket.AT_RISK
    assert out.risk_score == 65


def test_port_out_flag_forces_critical_bucket():
    customer = {**SAMPLE_CUSTOMER, "port_out_request_flag": True}
    fake_json = json.dumps({
        "customer_id": "C0001",
        "risk_score": 50,
        "bucket": "WATCH",
        "top_3_drivers": ["a", "b", "c"],
        "rationale": "Mild risk per LLM but port-out is filed.",
    })
    with patch("agents.analyst.call_llm", return_value=fake_json):
        out = analyze_customer(customer)
    assert out.bucket == Bucket.CRITICAL
    assert out.risk_score >= 85


def test_invalid_llm_response_retries_then_falls_back_to_safe():
    with patch("agents.analyst.call_llm", side_effect=["not json", "still not json"]):
        out = analyze_customer(SAMPLE_CUSTOMER)
    assert out.bucket == Bucket.SAFE
    assert "fallback" in out.rationale.lower()


def test_force_critical_helper_is_idempotent():
    fake = {
        "customer_id": "C0001",
        "risk_score": 90,
        "bucket": "CRITICAL",
        "top_3_drivers": ["a", "b", "c"],
        "rationale": "already critical",
    }
    out = force_critical_on_port_out(fake)
    assert out["bucket"] == "CRITICAL"
    assert out["risk_score"] == 90
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_analyst.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `agents/analyst.py`**

```python
"""Agent 1: Data Analyst — scores customers and assigns a churn bucket."""
import json
import logging
from pathlib import Path
from typing import Any

from llm import call_llm
from schemas import AnalystOutput, Bucket

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "analyst.txt"
SYSTEM_PROMPT = "You output strict JSON only."


def _safe_customer(customer: dict) -> dict:
    """Strip name and customer_id from the LLM-facing payload."""
    return {k: v for k, v in customer.items() if k not in {"name"}}


def force_critical_on_port_out(parsed: dict) -> dict:
    """If LLM didn't catch the port-out flag, force the bucket."""
    parsed["bucket"] = "CRITICAL"
    parsed["risk_score"] = max(85, int(parsed.get("risk_score", 0)))
    return parsed


def _fallback_safe_output(customer_id: str) -> AnalystOutput:
    return AnalystOutput(
        customer_id=customer_id,
        risk_score=0,
        bucket=Bucket.SAFE,
        top_3_drivers=["fallback: LLM unavailable"],
        rationale="Fallback output. LLM analysis was unavailable; treating as Safe by default.",
    )


def analyze_customer(customer: dict) -> AnalystOutput:
    """Run Agent 1 on a single customer row. Always returns AnalystOutput."""
    user_msg = PROMPT_PATH.read_text().replace(
        "{customer_json}", json.dumps(_safe_customer(customer))
    )

    for attempt in range(2):
        try:
            raw = call_llm(SYSTEM_PROMPT, user_msg, expect_json=True)
            parsed = json.loads(raw)
            parsed["customer_id"] = customer["customer_id"]
            if customer.get("port_out_request_flag"):
                parsed = force_critical_on_port_out(parsed)
            return AnalystOutput(**parsed)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Analyst attempt %d failed: %s", attempt + 1, e)

    return _fallback_safe_output(customer["customer_id"])


def analyze_all(customers: list[dict]) -> list[AnalystOutput]:
    return [analyze_customer(c) for c in customers]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_analyst.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/analyst.py tests/test_analyst.py
git commit -m "feat: Agent 1 — Data Analyst with port-out override + fallback"
```

---

## Task 9: Agent 3 — Executor (offer generator)

**Files:**
- Create: `agents/executor.py`
- Create: `tests/test_executor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_executor.py`:

```python
import json
from unittest.mock import patch

import pytest

from agents.executor import generate_offer
from schemas import AnalystOutput, Bucket, OfferType


CUSTOMER = {
    "customer_id": "C0001",
    "name": "Asha Iyer",
    "plan_type": "prepaid",
    "is_premium": False,
    "avg_monthly_arpu_inr": 300.0,
    "data_usage_gb_trend": "rising",
    "complaints_last_90d": 1,
    "tenure_months": 24,
    "port_out_request_flag": False,
}

ANALYSIS = AnalystOutput(
    customer_id="C0001",
    risk_score=70,
    bucket=Bucket.AT_RISK,
    top_3_drivers=["rising data usage on basic plan", "1 complaint", "low offer engagement"],
    rationale="Customer outgrowing their plan and minor friction reported.",
)


def test_valid_offer_passes_through():
    fake = json.dumps({
        "offer_type": "DATA_BOOST",
        "offer_details": "Extra 15GB/month free for 2 months",
        "monetary_value_inr": 450,
        "validity_days": 60,
        "justification": "Customer's data usage is rising and plan is hitting cap.",
        "expected_retention_lift": "moderate",
    })
    with patch("agents.executor.call_llm", return_value=fake):
        offer = generate_offer(CUSTOMER, ANALYSIS)
    assert offer.offer_type == OfferType.DATA_BOOST
    assert offer.monetary_value_inr == 450


def test_offer_violating_arpu_cap_falls_back():
    fake = json.dumps({
        "offer_type": "BILL_DISCOUNT",
        "offer_details": "Massive 5000 INR credit",
        "monetary_value_inr": 5000,  # > 3 * 300 = 900
        "validity_days": 30,
        "justification": "Big retention push.",
        "expected_retention_lift": "high",
    })
    with patch("agents.executor.call_llm", return_value=fake):
        offer = generate_offer(CUSTOMER, ANALYSIS)
    assert offer.monetary_value_inr <= 3 * CUSTOMER["avg_monthly_arpu_inr"]
    assert "fallback" in offer.justification.lower()


def test_offer_for_prepaid_with_postpaid_language_falls_back():
    fake = json.dumps({
        "offer_type": "BILL_DISCOUNT",
        "offer_details": "Postpaid bill credit of INR 200",
        "monetary_value_inr": 200,
        "validity_days": 30,
        "justification": "Helps with bill.",
        "expected_retention_lift": "low",
    })
    with patch("agents.executor.call_llm", return_value=fake):
        offer = generate_offer(CUSTOMER, ANALYSIS)
    assert "fallback" in offer.justification.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_executor.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `agents/executor.py`**

```python
"""Agent 3: Executor — generates a personalized retention offer."""
import json
import logging
from pathlib import Path

from guardrails import GuardrailViolation, validate_offer
from llm import call_llm
from schemas import AnalystOutput, OfferOutput, OfferType, RetentionLift

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "executor.txt"
SYSTEM_PROMPT = "You output strict JSON only."


def _fallback_offer(customer: dict) -> OfferOutput:
    """A safe, generic offer that always passes guardrails."""
    arpu = float(customer["avg_monthly_arpu_inr"])
    value = min(int(arpu * 0.5), int(3 * arpu))
    if customer["plan_type"] == "prepaid":
        details = "10% bonus on next recharge"
    else:
        details = "10% bill discount for one cycle"
    return OfferOutput(
        offer_type=OfferType.BILL_DISCOUNT,
        offer_details=details,
        monetary_value_inr=value,
        validity_days=30,
        justification="Fallback offer used because the generated offer failed validation.",
        expected_retention_lift=RetentionLift.LOW,
    )


def generate_offer(customer: dict, analysis: AnalystOutput) -> OfferOutput:
    user_msg = PROMPT_PATH.read_text() \
        .replace("{analyst_json}", analysis.model_dump_json()) \
        .replace("{customer_json}", json.dumps(customer))

    for attempt in range(2):
        try:
            raw = call_llm(SYSTEM_PROMPT, user_msg, expect_json=True)
            parsed = json.loads(raw)
            offer = OfferOutput(**parsed)
            validate_offer(offer, customer)
            return offer
        except (json.JSONDecodeError, ValueError, GuardrailViolation) as e:
            logger.warning("Executor attempt %d failed: %s", attempt + 1, e)

    return _fallback_offer(customer)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_executor.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/executor.py tests/test_executor.py
git commit -m "feat: Agent 3 — Executor with guardrail-aware fallback"
```

---

## Task 10: Agent 2 — Voice Agent (script + TTS)

**Files:**
- Create: `agents/voice.py`
- Create: `tests/test_voice.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_voice.py`:

```python
import json
from unittest.mock import patch

from agents.voice import generate_script
from schemas import AnalystOutput, Bucket, OfferOutput, OfferType, RetentionLift


CUSTOMER = {
    "customer_id": "C0001",
    "name": "Asha Iyer",
    "plan_type": "prepaid",
    "is_premium": False,
    "tenure_months": 24,
    "avg_monthly_arpu_inr": 300.0,
}

ANALYSIS = AnalystOutput(
    customer_id="C0001", risk_score=70, bucket=Bucket.AT_RISK,
    top_3_drivers=["rising data usage", "1 complaint", "low engagement"],
    rationale="Outgrowing plan.",
)

OFFER = OfferOutput(
    offer_type=OfferType.DATA_BOOST, offer_details="Extra 15GB free for 2 months",
    monetary_value_inr=450, validity_days=60,
    justification="Addresses data usage.", expected_retention_lift=RetentionLift.MODERATE,
)


def test_valid_script_passes():
    fake = json.dumps({
        "opening_line": "Hi Asha, thanks for being with us for two years.",
        "key_talking_points": ["acknowledge data usage", "introduce offer"],
        "full_script": (
            "Hi Asha, thanks for being with us for two years. I noticed your data "
            "usage has been growing recently, and I wanted to share an offer that "
            "might help — extra 15GB free for the next two months, on us. "
            "Would that be useful for you?"
        ),
        "do_not_say": ["do not name competitors"],
        "estimated_call_duration_sec": 75,
    })
    with patch("agents.voice.call_llm", return_value=fake):
        script = generate_script(CUSTOMER, ANALYSIS, OFFER)
    assert "Asha" in script.opening_line
    assert script.estimated_call_duration_sec == 75


def test_script_with_competitor_mention_falls_back():
    fake = json.dumps({
        "opening_line": "Hi Asha.",
        "key_talking_points": ["a", "b"],
        "full_script": (
            "Hi Asha, I know Jio has been pushing offers but ours is better. "
            "We have an offer of extra 15GB free for the next two months."
        ),
        "do_not_say": ["x"],
        "estimated_call_duration_sec": 60,
    })
    with patch("agents.voice.call_llm", return_value=fake):
        script = generate_script(CUSTOMER, ANALYSIS, OFFER)
    assert "fallback" in script.do_not_say[0].lower() or "fallback" in script.opening_line.lower()


def test_script_with_pressure_tactic_falls_back():
    fake = json.dumps({
        "opening_line": "Hi Asha.",
        "key_talking_points": ["a", "b"],
        "full_script": (
            "Hi Asha. You must act now or lose this forever. This is your last "
            "chance to claim a 15GB data boost from us."
        ),
        "do_not_say": ["x"],
        "estimated_call_duration_sec": 60,
    })
    with patch("agents.voice.call_llm", return_value=fake):
        script = generate_script(CUSTOMER, ANALYSIS, OFFER)
    assert "fallback" in script.do_not_say[0].lower() or "fallback" in script.opening_line.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_voice.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `agents/voice.py`**

```python
"""Agent 2: Voice Agent — generates a retention call script + TTS audio."""
import io
import json
import logging
from pathlib import Path

from gtts import gTTS

from guardrails import GuardrailViolation, validate_script
from llm import call_llm
from schemas import AnalystOutput, OfferOutput, VoiceOutput

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "voice.txt"
SYSTEM_PROMPT = "You output strict JSON only."


def _fallback_script(customer: dict, offer: OfferOutput) -> VoiceOutput:
    name = customer.get("name", "there")
    return VoiceOutput(
        opening_line=f"Hi {name}, thank you for being a valued customer (fallback).",
        key_talking_points=["thank for tenure", "introduce offer", "invite questions"],
        full_script=(
            f"Hi {name}, thank you for being with us. We appreciate your continued "
            f"trust. I wanted to share an offer with you today: {offer.offer_details}. "
            f"Would this be helpful for you? I'm happy to answer any questions."
        ),
        do_not_say=["fallback script used due to validation failure"],
        estimated_call_duration_sec=60,
    )


def generate_script(
    customer: dict, analysis: AnalystOutput, offer: OfferOutput
) -> VoiceOutput:
    user_msg = PROMPT_PATH.read_text() \
        .replace("{analyst_json}", analysis.model_dump_json()) \
        .replace("{customer_json}", json.dumps(customer)) \
        .replace("{offer_json}", offer.model_dump_json())

    for attempt in range(2):
        try:
            raw = call_llm(SYSTEM_PROMPT, user_msg, expect_json=True)
            parsed = json.loads(raw)
            script = VoiceOutput(**parsed)
            validate_script(script)
            return script
        except (json.JSONDecodeError, ValueError, GuardrailViolation) as e:
            logger.warning("Voice attempt %d failed: %s", attempt + 1, e)

    return _fallback_script(customer, offer)


def script_to_audio_bytes(script: VoiceOutput) -> bytes:
    """Convert script.full_script to MP3 bytes via gTTS."""
    tts = gTTS(text=script.full_script, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_voice.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/voice.py tests/test_voice.py
git commit -m "feat: Agent 2 — Voice Agent with gTTS audio output"
```

---

## Task 11: Streamlit app — load + dashboard

**Files:**
- Create: `app.py`

This task is UI-heavy. No unit tests for UI; visual verification by running the app.

- [ ] **Step 1: Write `app.py` (load + dashboard zones)**

```python
"""Telecom Churn Reduction Agent — Streamlit UI."""
import pandas as pd
import streamlit as st

from agents.analyst import analyze_all
from agents.executor import generate_offer
from agents.voice import generate_script, script_to_audio_bytes
from llm import provider_status
from orchestrator import gate, should_escalate, AgentName
from schemas import Bucket

BUCKET_COLOR = {
    Bucket.SAFE: "#10b981",
    Bucket.WATCH: "#f59e0b",
    Bucket.AT_RISK: "#f97316",
    Bucket.CRITICAL: "#ef4444",
}

st.set_page_config(page_title="Telecom Churn Agent", layout="wide")


@st.cache_data(show_spinner=False)
def load_customers() -> pd.DataFrame:
    return pd.read_csv("data/customers.csv")


@st.cache_data(show_spinner="Running Agent 1 on all customers…")
def run_analysis(rows: list[dict]) -> dict:
    results = analyze_all(rows)
    return {r.customer_id: r.model_dump() for r in results}


def render_sidebar():
    st.sidebar.title("Status")
    status = provider_status()
    if status["groq_configured"]:
        st.sidebar.success(f"Groq: {status['groq_model']}")
    else:
        st.sidebar.info("Groq: not configured (Ollama fallback)")
    st.sidebar.caption(f"Ollama fallback: {status['ollama_model']}")
    if st.sidebar.button("Regenerate analysis"):
        st.cache_data.clear()
        st.rerun()
    with st.sidebar.expander("Methodology & ethics"):
        st.markdown(
            "**Bucket cutoffs**\n\n"
            "- SAFE: 0–30\n- WATCH: 31–55\n- AT_RISK: 56–80\n- CRITICAL: 81–100\n\n"
            "Port-out request forces CRITICAL.\n\n"
            "**Ethical guardrails**\n\n"
            "- Name/ID never used as scoring input\n"
            "- No sensitive attributes (caste, religion, gender, age)\n"
            "- Scripts blocked from competitor mentions and pressure tactics\n"
            "- All outputs include a rationale field"
        )


def render_banner(analysis_by_id: dict):
    counts = {b: 0 for b in Bucket}
    for r in analysis_by_id.values():
        counts[Bucket(r["bucket"])] += 1
    critical = counts[Bucket.CRITICAL]
    at_risk = counts[Bucket.AT_RISK]
    if critical or at_risk:
        st.markdown(
            f"<div style='padding:12px;border-radius:8px;background:#fef2f2;"
            f"border:1px solid #ef4444;color:#7f1d1d;'>"
            f"<b>⚠ {critical} customers flagged Critical · {at_risk} At-Risk · "
            f"Action recommended</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.success("All customers in Safe/Watch buckets.")


def render_dashboard(df: pd.DataFrame, analysis_by_id: dict) -> str | None:
    st.subheader("Customer dashboard")

    bucket_filter = st.radio(
        "Filter",
        options=["All", "Safe", "Watch", "At-Risk", "Critical", "Premium only"],
        horizontal=True,
    )

    df = df.copy()
    df["bucket"] = df["customer_id"].map(lambda cid: analysis_by_id[cid]["bucket"])
    df["risk_score"] = df["customer_id"].map(lambda cid: analysis_by_id[cid]["risk_score"])
    df["top_driver"] = df["customer_id"].map(
        lambda cid: analysis_by_id[cid]["top_3_drivers"][0]
    )

    if bucket_filter == "Premium only":
        df = df[df["is_premium"]]
    elif bucket_filter != "All":
        mapping = {"Safe": "SAFE", "Watch": "WATCH", "At-Risk": "AT_RISK", "Critical": "CRITICAL"}
        df = df[df["bucket"] == mapping[bucket_filter]]

    display = df[[
        "customer_id", "name", "bucket", "risk_score",
        "tenure_months", "avg_monthly_arpu_inr", "top_driver",
    ]].sort_values("risk_score", ascending=False)

    event = st.dataframe(
        display,
        use_container_width=True,
        height=420,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = event.selection.rows if event and hasattr(event, "selection") else []
    if selected_rows:
        return display.iloc[selected_rows[0]]["customer_id"]
    return None


def main():
    df = load_customers()
    analysis_by_id = run_analysis(df.to_dict(orient="records"))

    render_sidebar()
    st.title("Telecom Churn Reduction Agent")
    st.caption("Demo data only. Not for production retention decisions.")

    render_banner(analysis_by_id)
    selected_id = render_dashboard(df, analysis_by_id)

    if selected_id:
        st.session_state["selected_id"] = selected_id

    # Drill-down panel implemented in Task 12.
    st.divider()
    st.caption("Select a customer above to view details (drill-down panel — Task 12).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests so far to verify nothing regressed**

```bash
pytest -v
```

Expected: all prior tests still passing.

- [ ] **Step 3: Manual smoke — run the app**

```bash
streamlit run app.py
```

Expected: App loads, banner shows, dashboard renders 200 rows with bucket chips/risk scores, filters work. Note: this will make 200 LLM calls — set `GROQ_API_KEY` for speed or expect ~10 min on Ollama.

For quick UI iteration before agents work, you can also subset the data: edit `load_customers()` temporarily to `return pd.read_csv("data/customers.csv").head(10)`.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit dashboard with banner, filters, row selection"
```

---

## Task 12: Streamlit app — drill-down panel + retention flow

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add a `render_drilldown` function above `main()` in `app.py`**

```python
def render_drilldown(
    df: pd.DataFrame, analysis_by_id: dict, customer_id: str
):
    row = df[df["customer_id"] == customer_id].iloc[0].to_dict()
    analysis = analysis_by_id[customer_id]
    bucket = Bucket(analysis["bucket"])

    st.divider()
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader(f"Customer {row['customer_id']} — {row['name']}")
        st.markdown(f"**Plan:** {row['plan_type']}  ·  **Premium:** {row['is_premium']}  ·  **Tenure:** {row['tenure_months']} months")
        st.markdown(f"**ARPU:** ₹{row['avg_monthly_arpu_inr']:.0f}/mo  ·  **Loyalty points:** {row['loyalty_points_balance']}")
        st.markdown(
            f"**Complaints (90d):** {row['complaints_last_90d']}  ·  "
            f"**Offers availed (180d):** {row['offers_availed_last_180d']}  ·  "
            f"**Data trend:** {row['data_usage_gb_trend']}"
        )
        st.markdown(
            f"**Network tickets:** {row['network_issue_tickets']}  ·  "
            f"**Call drops:** {row['call_drop_rate_pct']}%  ·  "
            f"**App logins (30d):** {row['app_logins_last_30d']}"
        )
        if row["port_out_request_flag"]:
            st.error("🚨 Port-out request filed")

    with col_b:
        color = BUCKET_COLOR[bucket]
        st.markdown(
            f"<div style='padding:8px 12px;border-radius:6px;"
            f"background:{color};color:white;display:inline-block;font-weight:600;'>"
            f"{bucket.value}</div>",
            unsafe_allow_html=True,
        )
        st.metric("Risk score", analysis["risk_score"], help="0–100, higher = more likely to churn")
        st.markdown("**Top drivers:**")
        for d in analysis["top_3_drivers"]:
            st.markdown(f"- {d}")
        st.markdown(f"_{analysis['rationale']}_")

    follow_ups = gate(bucket)
    disabled = AgentName.EXECUTOR not in follow_ups
    tooltip = "No retention action needed for Safe/Watch buckets" if disabled else None

    if st.button(
        "Run Retention Flow",
        type="primary",
        disabled=disabled,
        help=tooltip,
        key=f"flow_{customer_id}",
    ):
        run_retention_flow(row, analysis_by_id[customer_id])

    if customer_id in st.session_state.get("flow_results", {}):
        _render_flow_results(st.session_state["flow_results"][customer_id], bucket)


def run_retention_flow(customer: dict, analysis_dict: dict):
    from schemas import AnalystOutput
    analysis = AnalystOutput(**analysis_dict)
    with st.spinner("Generating offer…"):
        offer = generate_offer(customer, analysis)
    with st.spinner("Generating retention script…"):
        script = generate_script(customer, analysis, offer)
    audio = script_to_audio_bytes(script)
    st.session_state.setdefault("flow_results", {})[customer["customer_id"]] = {
        "offer": offer.model_dump(),
        "script": script.model_dump(),
        "audio": audio,
    }
    st.rerun()


def _render_flow_results(result: dict, bucket: Bucket):
    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Retention Offer")
        offer = result["offer"]
        st.markdown(
            f"<div style='padding:6px 10px;border-radius:4px;background:#dbeafe;"
            f"color:#1e3a8a;display:inline-block;font-weight:600;'>{offer['offer_type']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{offer['offer_details']}**")
        st.caption(
            f"Value: ₹{offer['monetary_value_inr']}  ·  "
            f"Valid: {offer['validity_days']} days  ·  "
            f"Expected lift: {offer['expected_retention_lift']}"
        )
        st.markdown(f"_{offer['justification']}_")

    with col2:
        st.subheader("Retention Script")
        script = result["script"]
        st.markdown(f"**Opening:** {script['opening_line']}")
        st.text_area("Full script", script["full_script"], height=200, label_visibility="collapsed")
        st.audio(result["audio"], format="audio/mp3")
        with st.expander("Do not say"):
            for item in script["do_not_say"]:
                st.markdown(f"- {item}")
        with st.expander("Talking points"):
            for tp in script["key_talking_points"]:
                st.markdown(f"- {tp}")

    if should_escalate(bucket):
        if st.button("🚨 Escalate to retention manager"):
            st.toast("Escalation logged (demo). In production, this would notify the retention manager.")
```

- [ ] **Step 2: Replace the placeholder in `main()` so it calls the drill-down**

Replace this block at the bottom of `main()`:

```python
    # Drill-down panel implemented in Task 12.
    st.divider()
    st.caption("Select a customer above to view details (drill-down panel — Task 12).")
```

with:

```python
    if selected_id:
        render_drilldown(df, analysis_by_id, selected_id)
```

- [ ] **Step 3: Manual smoke**

```bash
streamlit run app.py
```

Test golden path:
- Select an At-Risk row → click "Run Retention Flow" → offer + script + audio appear.
- Select a Safe row → button is disabled with tooltip.
- Select a Critical row → escalate button appears under results.
- Click ▶ on audio → script plays.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: drill-down panel with retention flow (Agents 2+3 + TTS)"
```

---

## Task 13: End-to-end smoke test

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
"""End-to-end smoke test — hits real Groq if GROQ_API_KEY set, else skips."""
import os

import pytest

from agents.analyst import analyze_customer
from agents.executor import generate_offer
from agents.voice import generate_script
from schemas import Bucket


CUSTOMER = {
    "customer_id": "C9999",
    "name": "Smoke Test User",
    "plan_type": "prepaid",
    "is_premium": False,
    "tenure_months": 36,
    "avg_monthly_arpu_inr": 400.0,
    "complaints_last_90d": 4,
    "offers_availed_last_180d": 0,
    "data_usage_gb_trend": "falling",
    "last_recharge_days_ago": 45,
    "bill_payment_delays_count": None,
    "network_issue_tickets": 3,
    "call_drop_rate_pct": 6.5,
    "last_outage_days_ago": 5,
    "app_logins_last_30d": 1,
    "loyalty_points_balance": 250,
    "family_plan_members": 0,
    "port_out_request_flag": False,
}


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live smoke test",
)
def test_full_pipeline_with_real_llm():
    analysis = analyze_customer(CUSTOMER)
    assert analysis.bucket in set(Bucket)

    offer = generate_offer(CUSTOMER, analysis)
    assert offer.monetary_value_inr <= 3 * CUSTOMER["avg_monthly_arpu_inr"]
    assert offer.validity_days <= 90

    script = generate_script(CUSTOMER, analysis, offer)
    assert len(script.full_script) > 50
    assert script.opening_line
```

- [ ] **Step 2: Run all tests**

```bash
pytest -v
```

Expected: all unit tests pass; smoke test either passes (if Groq key set) or skips.

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: end-to-end smoke test (skipped without Groq key)"
```

---

## Task 14: README polish

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md` with the full version**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: full README with setup, guardrails, testing"
```

---

## Final verification

- [ ] **Step 1: Run full test suite**

```bash
pytest -v
```

Expected: ~39 tests pass (1 skipped if no Groq key — the smoke test).

- [ ] **Step 2: Manual end-to-end demo**

```bash
streamlit run app.py
```

Verify per spec §12 success criteria:
1. App loads, banner shows, dashboard renders ~200 rows with bucket distribution roughly 50/25/15/10.
2. Filters work (All / Safe / Watch / At-Risk / Critical / Premium only).
3. Selecting an At-Risk row → "Run Retention Flow" → offer + script + audio within ~10s.
4. ▶ button plays script audibly.
5. Critical customers show escalate button.
6. Safe/Watch rows have disabled flow button with tooltip.
7. Sidebar shows correct provider status.
8. Methodology expander shows bucket cutoffs.

- [ ] **Step 3: Final commit (if anything was tweaked during verification)**

```bash
git status
# only commit if there's something
```

---

## Out of scope (do NOT add)

- Real telephony / Twilio.
- Inbound voice / STT.
- Persistent storage (Supabase, Postgres).
- Auth or role-based access.
- Production deployment (deferred).
- Additional offer types beyond the 5 enum values.
- Sensitive attributes in dataset.
