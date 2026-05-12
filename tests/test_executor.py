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
