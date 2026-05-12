import pandas as pd
import pytest
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "customers.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)


def test_csv_has_200_rows(df):
    assert len(df) == 200


def test_csv_has_all_expected_columns(df):
    expected = {
        "customer_id", "name", "plan_type", "is_premium",
        "tenure_months", "avg_monthly_arpu_inr",
        "complaints_last_90d", "offers_availed_last_180d",
        "data_usage_gb_trend",
        "last_recharge_days_ago", "bill_payment_delays_count",
        "network_issue_tickets", "call_drop_rate_pct",
        "last_outage_days_ago", "app_logins_last_30d",
        "loyalty_points_balance", "family_plan_members",
        "port_out_request_flag",
    }
    assert set(df.columns) == expected


def test_prepaid_customers_have_no_bill_delays(df):
    prepaid = df[df["plan_type"] == "prepaid"]
    assert prepaid["bill_payment_delays_count"].isna().all()


def test_postpaid_customers_have_no_last_recharge(df):
    postpaid = df[df["plan_type"] == "postpaid"]
    assert postpaid["last_recharge_days_ago"].isna().all()


def test_port_out_flag_in_5_to_10_percent_range(df):
    pct = df["port_out_request_flag"].mean() * 100
    assert 3 <= pct <= 12


def test_premium_flag_in_10_to_25_percent_range(df):
    pct = df["is_premium"].mean() * 100
    assert 10 <= pct <= 25
