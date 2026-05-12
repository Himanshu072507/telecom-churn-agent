"""Agent 3: Executor — generates a personalized retention offer."""
import json
import logging
from pathlib import Path

from guardrails import GuardrailViolation, validate_offer
from llm import call_llm, LLMError
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
        except Exception as e:  # broad: LLM failures, parse errors, guardrail violations
            logger.warning("Executor attempt %d failed: %s", attempt + 1, e)

    return _fallback_offer(customer)
