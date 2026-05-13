"""Anchor accuracy eval: do the 8 hand-tuned customers land in their designed buckets?

Run from project root:
    GROQ_API_KEY=gsk_... python -m evals.run_anchors          # default N=5 per anchor
    GROQ_API_KEY=gsk_... python -m evals.run_anchors --n 10
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.analyst import analyze_customer
from data.generate import ANCHOR_CUSTOMERS
from llm import set_groq_key
from schemas import Bucket


DESIGNED_BUCKET = {
    "C0001": Bucket.SAFE,
    "C0002": Bucket.SAFE,
    "C0003": Bucket.WATCH,
    "C0004": Bucket.WATCH,
    "C0005": Bucket.AT_RISK,
    "C0006": Bucket.AT_RISK,
    "C0007": Bucket.CRITICAL,  # port-out forces this
    "C0008": Bucket.CRITICAL,  # port-out forces this
}

THRESHOLD = 0.80


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="runs per anchor")
    args = parser.parse_args()

    key = os.getenv("GROQ_API_KEY")
    if key:
        set_groq_key(key)
        print("Provider: Groq")
    else:
        print("Provider: Ollama (GROQ_API_KEY not set)")

    print(f"Anchor accuracy eval — N={args.n} runs per anchor\n")
    print(f"{'ID':<6} {'Designed':<10} {'Hits':<6} {'Buckets seen':<40} {'Accuracy'}")
    print("-" * 80)

    overall_hits = 0
    overall_runs = 0
    failures = []

    for anchor in ANCHOR_CUSTOMERS:
        cid = anchor["customer_id"]
        designed = DESIGNED_BUCKET[cid]
        forced = bool(anchor["port_out_request_flag"])

        results = [analyze_customer(anchor) for _ in range(args.n)]
        buckets = Counter(r.bucket.value for r in results)
        hits = buckets[designed.value]
        accuracy = hits / args.n

        overall_hits += hits
        overall_runs += args.n

        note = " (port-out forced)" if forced else ""
        seen = ", ".join(f"{b}:{c}" for b, c in buckets.most_common())
        mark = "✓" if accuracy >= THRESHOLD else "✗"
        print(f"{cid:<6} {designed.value:<10} {hits}/{args.n:<4} {seen:<40} {accuracy:.0%} {mark}{note}")

        if accuracy < THRESHOLD and not forced:
            failures.append((cid, designed.value, seen))

    print("-" * 80)
    overall = overall_hits / overall_runs
    print(f"\nOverall: {overall_hits}/{overall_runs} = {overall:.1%}")
    print(f"Threshold: ≥{THRESHOLD:.0%} per anchor")

    if failures:
        print(f"\n✗ {len(failures)} anchor(s) below threshold:")
        for cid, designed, seen in failures:
            print(f"  - {cid} (designed {designed}): {seen}")
        sys.exit(1)
    print("\n✓ All anchors meet threshold")


if __name__ == "__main__":
    main()
