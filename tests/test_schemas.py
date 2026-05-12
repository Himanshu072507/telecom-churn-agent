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
            rationale="Test rationale longer than ten chars.",
        )


def test_offer_output_rejects_negative_value():
    with pytest.raises(ValidationError):
        OfferOutput(
            offer_type=OfferType.DATA_BOOST,
            offer_details="Test offer details",
            monetary_value_inr=-100,
            validity_days=30,
            justification="Test justification text.",
            expected_retention_lift="moderate",
        )


def test_voice_output_requires_do_not_say_list():
    vo = VoiceOutput(
        opening_line="Hello there, how are you today?",
        key_talking_points=["a", "b"],
        full_script="This is a full script with enough characters to pass validation requirements.",
        do_not_say=["never name competitors"],
        estimated_call_duration_sec=90,
    )
    assert "never name competitors" in vo.do_not_say
