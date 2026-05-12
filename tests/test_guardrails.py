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


def test_postpaid_customer_cannot_receive_recharge_offer():
    customer = {"plan_type": "postpaid", "is_premium": False, "avg_monthly_arpu_inr": 800.0}
    offer = base_offer(offer_details="Bonus 10% on your next recharge")
    with pytest.raises(GuardrailViolation, match="recharge"):
        validate_offer(offer, customer)


def test_script_mentioning_competitor_fails():
    script = base_script(full_script="Hi, switching to Jio is a mistake, our network is better. We offer reliable service.")
    with pytest.raises(GuardrailViolation, match="competitor"):
        validate_script(script)


def test_script_with_urgency_pressure_fails():
    script = base_script(full_script="Hi, you MUST act now or lose this forever, this is your last chance. Reliable network.")
    with pytest.raises(GuardrailViolation, match="pressure"):
        validate_script(script)


def test_clean_script_passes():
    validate_script(base_script())
