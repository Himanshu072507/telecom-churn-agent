"""Voice tone sanity eval for Agent 2: do generated scripts ever leak competitor
mentions or pressure phrases past the guardrail layer?

Run from project root:
    GROQ_API_KEY=gsk_... python -m evals.run_scripts --n 3
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from guardrails import COMPETITOR_PATTERNS, PRESSURE_PATTERNS, GuardrailViolation, validate_script
from llm import call_llm, set_groq_key
from schemas import AnalystOutput, Bucket, OfferOutput, OfferType, RetentionLift, VoiceOutput


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "voice.txt"
SYSTEM_PROMPT = "You output strict JSON only."


# One representative customer per bucket. Scripts only generate for At-Risk and
# Critical in the real app, but for the eval we sample across all 4.
SAMPLES = [
    {
        "label": "AT_RISK / prepaid / data falling",
        "customer": {
            "customer_id": "V0001", "name": "Asha", "plan_type": "prepaid",
            "is_premium": False, "tenure_months": 24, "avg_monthly_arpu_inr": 350.0,
        },
        "analysis": AnalystOutput(
            customer_id="V0001", risk_score=70, bucket=Bucket.AT_RISK,
            top_3_drivers=["4 complaints in 90d", "data usage falling", "low app engagement"],
            rationale="Customer dissatisfied across multiple axes.",
        ),
        "offer": OfferOutput(
            offer_type=OfferType.DATA_BOOST, offer_details="Extra 15GB free for 2 months",
            monetary_value_inr=400, validity_days=60,
            justification="Addresses data usage drop.", expected_retention_lift=RetentionLift.MODERATE,
        ),
    },
    {
        "label": "CRITICAL / postpaid / port-out filed",
        "customer": {
            "customer_id": "V0002", "name": "Vikram", "plan_type": "postpaid",
            "is_premium": True, "tenure_months": 48, "avg_monthly_arpu_inr": 1500.0,
        },
        "analysis": AnalystOutput(
            customer_id="V0002", risk_score=92, bucket=Bucket.CRITICAL,
            top_3_drivers=["port-out request filed", "7 complaints", "5 network tickets"],
            rationale="Premium customer poised to leave.",
        ),
        "offer": OfferOutput(
            offer_type=OfferType.LOYALTY_UPGRADE, offer_details="Tier upgrade + 2000 bonus loyalty points",
            monetary_value_inr=2000, validity_days=90,
            justification="Recognize tenure and address dissatisfaction.", expected_retention_lift=RetentionLift.HIGH,
        ),
    },
    {
        "label": "AT_RISK / postpaid / bill delays",
        "customer": {
            "customer_id": "V0003", "name": "Priya", "plan_type": "postpaid",
            "is_premium": False, "tenure_months": 18, "avg_monthly_arpu_inr": 800.0,
        },
        "analysis": AnalystOutput(
            customer_id="V0003", risk_score=65, bucket=Bucket.AT_RISK,
            top_3_drivers=["3 bill payment delays", "2 complaints", "network issues"],
            rationale="Customer facing payment friction.",
        ),
        "offer": OfferOutput(
            offer_type=OfferType.BILL_DISCOUNT, offer_details="15% bill discount for 2 months",
            monetary_value_inr=240, validity_days=60,
            justification="Eases payment burden.", expected_retention_lift=RetentionLift.MODERATE,
        ),
    },
]


def _call_voice_raw(customer, analysis, offer):
    user_msg = PROMPT_PATH.read_text() \
        .replace("{analyst_json}", analysis.model_dump_json()) \
        .replace("{customer_json}", json.dumps(customer)) \
        .replace("{offer_json}", offer.model_dump_json())
    raw = call_llm(SYSTEM_PROMPT, user_msg, expect_json=True)
    parsed = json.loads(raw)
    return VoiceOutput(**parsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="runs per sample")
    args = parser.parse_args()

    key = os.getenv("GROQ_API_KEY")
    if key:
        set_groq_key(key)
        print("Provider: Groq")
    else:
        print("Provider: Ollama (GROQ_API_KEY not set)")

    print(f"Agent 2 voice tone eval — N={args.n} runs per sample\n")

    total = 0
    compliant = 0
    competitor_hits = 0
    pressure_hits = 0
    parse_errors = 0

    for sample in SAMPLES:
        print(f"• {sample['label']}")
        for i in range(args.n):
            total += 1
            try:
                script = _call_voice_raw(sample["customer"], sample["analysis"], sample["offer"])
            except Exception as e:
                parse_errors += 1
                print(f"    run {i+1}: ✗ parse error — {e}")
                continue

            text = script.full_script.lower()
            comp = [p for p in COMPETITOR_PATTERNS if re.search(p, text)]
            press = [p for p in PRESSURE_PATTERNS if re.search(p, text)]

            if comp or press:
                if comp:
                    competitor_hits += 1
                if press:
                    pressure_hits += 1
                print(f"    run {i+1}: ⚠ leak — competitor={comp} pressure={press}")
            else:
                # Confirm guardrail also passes (sanity)
                try:
                    validate_script(script)
                    compliant += 1
                    preview = script.opening_line[:70]
                    print(f"    run {i+1}: ✓ clean — \"{preview}…\"")
                except GuardrailViolation as gv:
                    print(f"    run {i+1}: ⚠ guardrail still caught — {gv}")

    print("\n" + "=" * 70)
    print("Summary")
    print(f"  Total runs:           {total}")
    print(f"  Clean (no leaks):     {compliant} ({compliant/total:.0%})")
    print(f"  Competitor mentions:  {competitor_hits}")
    print(f"  Pressure phrases:     {pressure_hits}")
    print(f"  Parse errors:         {parse_errors}")


if __name__ == "__main__":
    main()
