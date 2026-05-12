"""Agent 1: Data Analyst — scores customers and assigns a churn bucket."""
import json
import logging
from pathlib import Path

from llm import call_llm
from schemas import AnalystOutput, Bucket

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "analyst.txt"
SYSTEM_PROMPT = "You output strict JSON only."


def _safe_customer(customer: dict) -> dict:
    """Strip name from the LLM-facing payload (customer_id stays for echoing)."""
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
