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
