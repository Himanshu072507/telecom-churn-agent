"""Pydantic models and enums shared across agents."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, conint, confloat


class Bucket(str, Enum):
    SAFE = "SAFE"
    WATCH = "WATCH"
    AT_RISK = "AT_RISK"
    CRITICAL = "CRITICAL"


class OfferType(str, Enum):
    DATA_BOOST = "DATA_BOOST"
    BILL_DISCOUNT = "BILL_DISCOUNT"
    LOYALTY_UPGRADE = "LOYALTY_UPGRADE"
    DEVICE_OFFER = "DEVICE_OFFER"
    PLAN_UPGRADE = "PLAN_UPGRADE"


class PlanType(str, Enum):
    PREPAID = "prepaid"
    POSTPAID = "postpaid"


class DataTrend(str, Enum):
    RISING = "rising"
    FLAT = "flat"
    FALLING = "falling"


class RetentionLift(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class Customer(BaseModel):
    customer_id: str
    name: str
    plan_type: PlanType
    is_premium: bool
    tenure_months: conint(ge=0, le=240)
    avg_monthly_arpu_inr: confloat(ge=0)
    complaints_last_90d: conint(ge=0)
    offers_availed_last_180d: conint(ge=0)
    data_usage_gb_trend: DataTrend
    last_recharge_days_ago: Optional[conint(ge=0)] = None
    bill_payment_delays_count: Optional[conint(ge=0)] = None
    network_issue_tickets: conint(ge=0)
    call_drop_rate_pct: confloat(ge=0, le=100)
    last_outage_days_ago: conint(ge=0)
    app_logins_last_30d: conint(ge=0)
    loyalty_points_balance: conint(ge=0)
    family_plan_members: conint(ge=0)
    port_out_request_flag: bool


class AnalystOutput(BaseModel):
    customer_id: str
    risk_score: conint(ge=0, le=100)
    bucket: Bucket
    top_3_drivers: list[str] = Field(..., min_length=1, max_length=5)
    rationale: str = Field(..., min_length=10)


class OfferOutput(BaseModel):
    offer_type: OfferType
    offer_details: str = Field(..., min_length=5)
    monetary_value_inr: conint(ge=0)
    validity_days: conint(ge=1, le=365)
    justification: str = Field(..., min_length=10)
    expected_retention_lift: RetentionLift


class VoiceOutput(BaseModel):
    opening_line: str = Field(..., min_length=5)
    key_talking_points: list[str] = Field(..., min_length=1)
    full_script: str = Field(..., min_length=50)
    do_not_say: list[str] = Field(..., min_length=1)
    estimated_call_duration_sec: conint(ge=15, le=600)
