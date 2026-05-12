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
