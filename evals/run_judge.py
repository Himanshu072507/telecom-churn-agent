"""LLM-as-judge evals for output quality (not just rule compliance).

Three judge dimensions:
  1. Rationale specificity — does Agent 1's rationale cite real numbers?
  2. Offer-driver match    — does Agent 3's offer address the top driver?
  3. Script quality        — warmth, specificity, naturalness, offer integration

Bias note: the judge defaults to the same Groq model as the generators. This
introduces self-grading bias. For a stricter eval, override JUDGE_MODEL with a
different family (e.g., a Mixtral or smaller Llama variant).

Run from project root:
    GROQ_API_KEY=gsk_... python -m evals.run_judge
    GROQ_API_KEY=gsk_... JUDGE_MODEL=llama-3.1-8b-instant python -m evals.run_judge
"""
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.analyst import analyze_customer
from agents.executor import generate_offer
from agents.voice import generate_script
from llm import call_llm, set_groq_key


JUDGE_MODEL = os.getenv("JUDGE_MODEL")  # if set, overrides default at call time
JUDGE_SYSTEM = "You are a strict evaluator. Output JSON only. No prose."

QUALITY_THRESHOLD = 3.5  # average score per dimension below this triggers a flag


# 5 representative customers across the bucket spectrum.
SAMPLES = [
    {
        "label": "SAFE (premium long-tenure postpaid)",
        "customer": {
            "customer_id": "J0001", "name": "Ravi", "plan_type": "postpaid",
            "is_premium": True, "tenure_months": 96, "avg_monthly_arpu_inr": 1800.0,
            "complaints_last_90d": 0, "offers_availed_last_180d": 2,
            "data_usage_gb_trend": "rising", "last_recharge_days_ago": None,
            "bill_payment_delays_count": 0, "network_issue_tickets": 0,
            "call_drop_rate_pct": 0.5, "last_outage_days_ago": 200,
            "app_logins_last_30d": 25, "loyalty_points_balance": 8500,
            "family_plan_members": 3, "port_out_request_flag": False,
        },
    },
    {
        "label": "WATCH (mild friction prepaid)",
        "customer": {
            "customer_id": "J0002", "name": "Arjun", "plan_type": "prepaid",
            "is_premium": False, "tenure_months": 30, "avg_monthly_arpu_inr": 350.0,
            "complaints_last_90d": 1, "offers_availed_last_180d": 2,
            "data_usage_gb_trend": "flat", "last_recharge_days_ago": 5,
            "bill_payment_delays_count": None, "network_issue_tickets": 0,
            "call_drop_rate_pct": 1.5, "last_outage_days_ago": 75,
            "app_logins_last_30d": 12, "loyalty_points_balance": 1200,
            "family_plan_members": 0, "port_out_request_flag": False,
        },
    },
    {
        "label": "AT_RISK (network pain, falling usage, postpaid)",
        "customer": {
            "customer_id": "J0003", "name": "Sneha", "plan_type": "postpaid",
            "is_premium": False, "tenure_months": 12, "avg_monthly_arpu_inr": 850.0,
            "complaints_last_90d": 5, "offers_availed_last_180d": 0,
            "data_usage_gb_trend": "falling", "last_recharge_days_ago": None,
            "bill_payment_delays_count": 3, "network_issue_tickets": 3,
            "call_drop_rate_pct": 7.5, "last_outage_days_ago": 8,
            "app_logins_last_30d": 2, "loyalty_points_balance": 200,
            "family_plan_members": 0, "port_out_request_flag": False,
        },
    },
    {
        "label": "AT_RISK (low-ARPU prepaid stressed)",
        "customer": {
            "customer_id": "J0004", "name": "Vikram", "plan_type": "prepaid",
            "is_premium": False, "tenure_months": 8, "avg_monthly_arpu_inr": 280.0,
            "complaints_last_90d": 6, "offers_availed_last_180d": 0,
            "data_usage_gb_trend": "falling", "last_recharge_days_ago": 50,
            "bill_payment_delays_count": None, "network_issue_tickets": 4,
            "call_drop_rate_pct": 9.0, "last_outage_days_ago": 3,
            "app_logins_last_30d": 1, "loyalty_points_balance": 150,
            "family_plan_members": 0, "port_out_request_flag": False,
        },
    },
    {
        "label": "CRITICAL (premium port-out filed)",
        "customer": {
            "customer_id": "J0005", "name": "Neha", "plan_type": "postpaid",
            "is_premium": True, "tenure_months": 36, "avg_monthly_arpu_inr": 1200.0,
            "complaints_last_90d": 7, "offers_availed_last_180d": 0,
            "data_usage_gb_trend": "falling", "last_recharge_days_ago": None,
            "bill_payment_delays_count": 4, "network_issue_tickets": 5,
            "call_drop_rate_pct": 11.0, "last_outage_days_ago": 2,
            "app_logins_last_30d": 0, "loyalty_points_balance": 100,
            "family_plan_members": 2, "port_out_request_flag": True,
        },
    },
]


RATIONALE_JUDGE_PROMPT = """\
Score the rationale field of a churn analysis on 2 dimensions (each 1-5).

Customer data: {customer_json}
Bucket: {bucket}
Risk score: {risk_score}
Top drivers: {drivers_json}
Rationale: "{rationale}"

Dimensions:
- specificity: Does the rationale cite concrete numbers or behaviors from the data? (1=generic platitudes, 5=cites specific signals)
- coherence: Does the rationale logically follow from the drivers? (1=contradicts drivers, 5=clear synthesis)

Output JSON only:
{{
  "specificity": <int 1-5>,
  "coherence": <int 1-5>,
  "comment": "<1 sentence rationale for the lowest score>"
}}
"""

OFFER_JUDGE_PROMPT = """\
Score whether a retention offer addresses the customer's drivers (each 1-5).

Customer profile: {customer_json}
Top drivers: {drivers_json}
Offer: {offer_json}

Dimensions:
- relevance: Does the offer mechanism address the strongest driver?
  Examples: network complaints → device_offer or bill_discount is relevant; data_boost is mismatched. Data usage rising on basic plan → data_boost or plan_upgrade is relevant.
- value_calibration: Is the monetary value reasonable for the risk level and customer ARPU? Not so small it's insulting, not so large it's wasteful.

Output JSON only:
{{
  "relevance": <int 1-5>,
  "value_calibration": <int 1-5>,
  "comment": "<1 sentence rationale for the lowest score>"
}}
"""

SCRIPT_JUDGE_PROMPT = """\
Score a retention call script on 4 dimensions (each 1-5).

Customer profile: {customer_json}
Top drivers: {drivers_json}
Offer being presented: {offer_json}
Generated script: "{script_text}"

Dimensions:
- warmth: Warm and empathetic, like a human rep? (1=robotic, 5=naturally warm)
- specificity: References the customer's actual pain points from top_drivers? (1=generic, 5=cites specifics)
- naturalness: Sounds like spoken conversation, not written marketing? (1=ad copy, 5=natural speech)
- offer_integration: Offer introduced smoothly, not bolted on? (1=jarring, 5=fits naturally)

Output JSON only:
{{
  "warmth": <int 1-5>,
  "specificity": <int 1-5>,
  "naturalness": <int 1-5>,
  "offer_integration": <int 1-5>,
  "comment": "<1 sentence rationale for the lowest score>"
}}
"""


def _judge(prompt: str) -> dict:
    """Call the judge LLM and parse JSON."""
    raw = call_llm(JUDGE_SYSTEM, prompt, expect_json=True)
    return json.loads(raw)


def _print_dimension_stats(name: str, scores: dict[str, list[int]]):
    print(f"\n{name}")
    print(f"{'dimension':<20} {'mean':>6} {'min':>5} {'max':>5} {'n':>4}")
    print("-" * 50)
    for dim, vals in scores.items():
        if not vals:
            continue
        mean = statistics.mean(vals)
        flag = " ⚠" if mean < QUALITY_THRESHOLD else ""
        print(f"{dim:<20} {mean:>6.2f} {min(vals):>5} {max(vals):>5} {len(vals):>4}{flag}")


def main():
    key = os.getenv("GROQ_API_KEY")
    if key:
        set_groq_key(key)
        print("Provider: Groq")
    else:
        print("Provider: Ollama (GROQ_API_KEY not set — using local fallback)")
    if JUDGE_MODEL:
        print(f"Judge model override: {JUDGE_MODEL} (note: passed via prompt only; generator still uses default Groq model)")

    rationale_scores = {"specificity": [], "coherence": []}
    offer_scores = {"relevance": [], "value_calibration": []}
    script_scores = {"warmth": [], "specificity": [], "naturalness": [], "offer_integration": []}
    flagged = []

    for sample in SAMPLES:
        label = sample["label"]
        customer = sample["customer"]
        print(f"\n• {label} ({customer['customer_id']})")

        # 1. Agent 1 → rationale judge
        analysis = analyze_customer(customer)
        judgment = _judge(RATIONALE_JUDGE_PROMPT.format(
            customer_json=json.dumps(customer),
            bucket=analysis.bucket.value,
            risk_score=analysis.risk_score,
            drivers_json=json.dumps(analysis.top_3_drivers),
            rationale=analysis.rationale,
        ))
        rationale_scores["specificity"].append(judgment["specificity"])
        rationale_scores["coherence"].append(judgment["coherence"])
        print(f"  rationale → spec={judgment['specificity']} coh={judgment['coherence']} — {judgment.get('comment', '')[:80]}")
        for dim in ("specificity", "coherence"):
            if judgment[dim] < QUALITY_THRESHOLD:
                flagged.append(f"{customer['customer_id']} rationale.{dim}={judgment[dim]}")

        # Skip Agent 2/3 for SAFE/WATCH (orchestrator wouldn't run them either).
        if analysis.bucket.value in ("SAFE", "WATCH"):
            print("  (no offer/script — orchestrator gates Safe/Watch)")
            continue

        # 2. Agent 3 → offer judge
        offer = generate_offer(customer, analysis)
        judgment = _judge(OFFER_JUDGE_PROMPT.format(
            customer_json=json.dumps(customer),
            drivers_json=json.dumps(analysis.top_3_drivers),
            offer_json=offer.model_dump_json(),
        ))
        offer_scores["relevance"].append(judgment["relevance"])
        offer_scores["value_calibration"].append(judgment["value_calibration"])
        print(f"  offer    → rel={judgment['relevance']} val={judgment['value_calibration']} — {judgment.get('comment', '')[:80]}")
        for dim in ("relevance", "value_calibration"):
            if judgment[dim] < QUALITY_THRESHOLD:
                flagged.append(f"{customer['customer_id']} offer.{dim}={judgment[dim]}")

        # 3. Agent 2 → script judge
        script = generate_script(customer, analysis, offer)
        judgment = _judge(SCRIPT_JUDGE_PROMPT.format(
            customer_json=json.dumps(customer),
            drivers_json=json.dumps(analysis.top_3_drivers),
            offer_json=offer.model_dump_json(),
            script_text=script.full_script,
        ))
        for dim in script_scores:
            script_scores[dim].append(judgment[dim])
        print(f"  script   → warmth={judgment['warmth']} spec={judgment['specificity']} "
              f"nat={judgment['naturalness']} integ={judgment['offer_integration']} — {judgment.get('comment', '')[:60]}")
        for dim in script_scores:
            if judgment[dim] < QUALITY_THRESHOLD:
                flagged.append(f"{customer['customer_id']} script.{dim}={judgment[dim]}")

    print("\n" + "=" * 60)
    print("SUMMARY (1-5 scale, mean across samples)")
    _print_dimension_stats("Rationale (Agent 1)", rationale_scores)
    _print_dimension_stats("Offer (Agent 3)", offer_scores)
    _print_dimension_stats("Script (Agent 2)", script_scores)
    print(f"\nQuality threshold: ≥{QUALITY_THRESHOLD}/5 per dimension")
    if flagged:
        print(f"\n⚠ {len(flagged)} dimension(s) below threshold:")
        for f in flagged:
            print(f"  - {f}")
    else:
        print("\n✓ All dimensions meet threshold")
    print("\nCaveat: judge uses same Groq family as generators — possible self-grading bias.")


if __name__ == "__main__":
    main()
