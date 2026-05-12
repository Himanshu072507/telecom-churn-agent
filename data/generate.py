"""Generate seeded synthetic telecom customer data.

8 anchor customers (C0001–C0008) are hand-tuned, 2 per bucket, so the demo
reliably shows Safe / Watch / At-Risk / Critical. The remaining 192 are seeded
random rows.
"""
import random
from pathlib import Path

import pandas as pd
from faker import Faker

SEED = 42
N_ROWS = 200
OUT_PATH = Path(__file__).parent / "customers.csv"


ANCHOR_CUSTOMERS = [
    {
        "customer_id": "C0001", "name": "Ravi Sharma", "plan_type": "postpaid",
        "is_premium": True, "tenure_months": 96, "avg_monthly_arpu_inr": 1800.0,
        "complaints_last_90d": 0, "offers_availed_last_180d": 2,
        "data_usage_gb_trend": "rising", "last_recharge_days_ago": None,
        "bill_payment_delays_count": 0, "network_issue_tickets": 0,
        "call_drop_rate_pct": 0.5, "last_outage_days_ago": 220,
        "app_logins_last_30d": 25, "loyalty_points_balance": 8500,
        "family_plan_members": 3, "port_out_request_flag": False,
    },
    {
        "customer_id": "C0002", "name": "Priya Kapoor", "plan_type": "prepaid",
        "is_premium": False, "tenure_months": 60, "avg_monthly_arpu_inr": 450.0,
        "complaints_last_90d": 0, "offers_availed_last_180d": 1,
        "data_usage_gb_trend": "flat", "last_recharge_days_ago": 2,
        "bill_payment_delays_count": None, "network_issue_tickets": 0,
        "call_drop_rate_pct": 0.8, "last_outage_days_ago": 180,
        "app_logins_last_30d": 18, "loyalty_points_balance": 3200,
        "family_plan_members": 0, "port_out_request_flag": False,
    },
    {
        "customer_id": "C0003", "name": "Anjali Mehta", "plan_type": "postpaid",
        "is_premium": False, "tenure_months": 18, "avg_monthly_arpu_inr": 700.0,
        "complaints_last_90d": 1, "offers_availed_last_180d": 0,
        "data_usage_gb_trend": "flat", "last_recharge_days_ago": None,
        "bill_payment_delays_count": 1, "network_issue_tickets": 1,
        "call_drop_rate_pct": 2.5, "last_outage_days_ago": 45,
        "app_logins_last_30d": 8, "loyalty_points_balance": 800,
        "family_plan_members": 1, "port_out_request_flag": False,
    },
    {
        "customer_id": "C0004", "name": "Arjun Reddy", "plan_type": "prepaid",
        "is_premium": False, "tenure_months": 30, "avg_monthly_arpu_inr": 350.0,
        "complaints_last_90d": 2, "offers_availed_last_180d": 1,
        "data_usage_gb_trend": "flat", "last_recharge_days_ago": 12,
        "bill_payment_delays_count": None, "network_issue_tickets": 1,
        "call_drop_rate_pct": 3.0, "last_outage_days_ago": 30,
        "app_logins_last_30d": 6, "loyalty_points_balance": 600,
        "family_plan_members": 0, "port_out_request_flag": False,
    },
    {
        "customer_id": "C0005", "name": "Sneha Iyer", "plan_type": "postpaid",
        "is_premium": False, "tenure_months": 12, "avg_monthly_arpu_inr": 850.0,
        "complaints_last_90d": 5, "offers_availed_last_180d": 0,
        "data_usage_gb_trend": "falling", "last_recharge_days_ago": None,
        "bill_payment_delays_count": 3, "network_issue_tickets": 3,
        "call_drop_rate_pct": 7.5, "last_outage_days_ago": 8,
        "app_logins_last_30d": 2, "loyalty_points_balance": 200,
        "family_plan_members": 0, "port_out_request_flag": False,
    },
    {
        "customer_id": "C0006", "name": "Vikram Singh", "plan_type": "prepaid",
        "is_premium": False, "tenure_months": 8, "avg_monthly_arpu_inr": 280.0,
        "complaints_last_90d": 6, "offers_availed_last_180d": 0,
        "data_usage_gb_trend": "falling", "last_recharge_days_ago": 50,
        "bill_payment_delays_count": None, "network_issue_tickets": 4,
        "call_drop_rate_pct": 9.0, "last_outage_days_ago": 3,
        "app_logins_last_30d": 1, "loyalty_points_balance": 150,
        "family_plan_members": 0, "port_out_request_flag": False,
    },
    {
        "customer_id": "C0007", "name": "Neha Bhatt", "plan_type": "postpaid",
        "is_premium": True, "tenure_months": 36, "avg_monthly_arpu_inr": 1200.0,
        "complaints_last_90d": 7, "offers_availed_last_180d": 0,
        "data_usage_gb_trend": "falling", "last_recharge_days_ago": None,
        "bill_payment_delays_count": 4, "network_issue_tickets": 5,
        "call_drop_rate_pct": 11.0, "last_outage_days_ago": 2,
        "app_logins_last_30d": 0, "loyalty_points_balance": 100,
        "family_plan_members": 2, "port_out_request_flag": True,
    },
    {
        "customer_id": "C0008", "name": "Karan Patel", "plan_type": "prepaid",
        "is_premium": False, "tenure_months": 24, "avg_monthly_arpu_inr": 320.0,
        "complaints_last_90d": 8, "offers_availed_last_180d": 0,
        "data_usage_gb_trend": "falling", "last_recharge_days_ago": 55,
        "bill_payment_delays_count": None, "network_issue_tickets": 5,
        "call_drop_rate_pct": 13.5, "last_outage_days_ago": 1,
        "app_logins_last_30d": 0, "loyalty_points_balance": 50,
        "family_plan_members": 0, "port_out_request_flag": True,
    },
]


def generate() -> pd.DataFrame:
    random.seed(SEED)
    Faker.seed(SEED)
    fake = Faker("en_IN")

    rows = [dict(c) for c in ANCHOR_CUSTOMERS]

    for i in range(len(ANCHOR_CUSTOMERS) + 1, N_ROWS + 1):
        plan_type = random.choices(["prepaid", "postpaid"], weights=[0.6, 0.4])[0]
        is_premium = random.random() < 0.15
        tenure = random.randint(1, 120)
        arpu = round(random.uniform(150, 2500), 2)
        port_out = random.random() < 0.05

        complaints = random.choices(
            range(0, 9), weights=[40, 20, 15, 10, 6, 4, 2, 2, 1]
        )[0]
        offers_availed = random.choices(
            range(0, 6), weights=[30, 25, 20, 12, 8, 5]
        )[0]
        trend = random.choices(
            ["rising", "flat", "falling"], weights=[0.35, 0.4, 0.25]
        )[0]
        network_issues = random.choices(
            range(0, 6), weights=[50, 20, 12, 8, 6, 4]
        )[0]
        call_drops = round(random.uniform(0, 15), 1)
        last_outage = random.randint(0, 365)
        app_logins = random.randint(0, 60)
        loyalty = random.randint(0, 10000)
        family = random.choices(range(0, 7), weights=[60, 10, 10, 8, 6, 4, 2])[0]
        last_recharge = random.randint(0, 60) if plan_type == "prepaid" else None
        bill_delays = (
            random.choices(range(0, 7), weights=[40, 25, 15, 10, 5, 3, 2])[0]
            if plan_type == "postpaid"
            else None
        )

        row = {
            "customer_id": f"C{i:04d}",
            "name": fake.name(),
            "plan_type": plan_type,
            "is_premium": is_premium,
            "tenure_months": tenure,
            "avg_monthly_arpu_inr": arpu,
            "complaints_last_90d": complaints,
            "offers_availed_last_180d": offers_availed,
            "data_usage_gb_trend": trend,
            "last_recharge_days_ago": last_recharge,
            "bill_payment_delays_count": bill_delays,
            "network_issue_tickets": network_issues,
            "call_drop_rate_pct": call_drops,
            "last_outage_days_ago": last_outage,
            "app_logins_last_30d": app_logins,
            "loyalty_points_balance": loyalty,
            "family_plan_members": family,
            "port_out_request_flag": port_out,
        }
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
