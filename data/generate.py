"""Generate seeded synthetic telecom customer data."""
import random
from pathlib import Path

import pandas as pd
from faker import Faker

SEED = 42
N_ROWS = 200
OUT_PATH = Path(__file__).parent / "customers.csv"


def generate() -> pd.DataFrame:
    random.seed(SEED)
    fake = Faker("en_IN")
    Faker.seed(SEED)

    rows = []
    for i in range(1, N_ROWS + 1):
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
            "last_recharge_days_ago": random.randint(0, 60) if plan_type == "prepaid" else None,
            "bill_payment_delays_count": random.choices(range(0, 7), weights=[40, 25, 15, 10, 5, 3, 2])[0] if plan_type == "postpaid" else None,
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
