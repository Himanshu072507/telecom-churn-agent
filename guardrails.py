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
    """Raise GuardrailViolation if offer breaks any telecom or business rule.

    `customer` is a dict (typically a row from the CSV converted via Pandas).
    Required keys: ``avg_monthly_arpu_inr`` (float), ``plan_type`` (str),
    ``is_premium`` (bool).
    """
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
