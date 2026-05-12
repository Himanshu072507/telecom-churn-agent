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
        "opening_line": "Hi Asha, glad to chat with you today.",
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
    # Fallback returns a known marker in do_not_say
    assert any("fallback" in d.lower() for d in script.do_not_say)


def test_script_with_pressure_tactic_falls_back():
    fake = json.dumps({
        "opening_line": "Hi Asha, hope you have a moment.",
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
    assert any("fallback" in d.lower() for d in script.do_not_say)
