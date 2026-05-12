"""Telecom Churn Reduction Agent — Streamlit UI."""
import pandas as pd
import streamlit as st

from agents.analyst import analyze_all
from agents.executor import generate_offer
from agents.voice import generate_script, script_to_audio_bytes
from llm import provider_status
from orchestrator import gate, should_escalate, AgentName
from schemas import Bucket

BUCKET_COLOR = {
    Bucket.SAFE: "#10b981",
    Bucket.WATCH: "#f59e0b",
    Bucket.AT_RISK: "#f97316",
    Bucket.CRITICAL: "#ef4444",
}

st.set_page_config(page_title="Telecom Churn Agent", layout="wide")


@st.cache_data(show_spinner=False)
def load_customers() -> pd.DataFrame:
    return pd.read_csv("data/customers.csv")


@st.cache_data(show_spinner="Running Agent 1 on all customers…")
def run_analysis(rows: list[dict]) -> dict:
    results = analyze_all(rows)
    return {r.customer_id: r.model_dump() for r in results}


def render_sidebar():
    st.sidebar.title("Status")
    status = provider_status()
    if status["groq_configured"]:
        st.sidebar.success(f"Groq: {status['groq_model']}")
    else:
        st.sidebar.info("Groq: not configured (Ollama fallback)")
    st.sidebar.caption(f"Ollama fallback: {status['ollama_model']}")
    if st.sidebar.button("Regenerate analysis"):
        st.cache_data.clear()
        st.rerun()
    with st.sidebar.expander("Methodology & ethics"):
        st.markdown(
            "**Bucket cutoffs**\n\n"
            "- SAFE: 0–30\n- WATCH: 31–55\n- AT_RISK: 56–80\n- CRITICAL: 81–100\n\n"
            "Port-out request forces CRITICAL.\n\n"
            "**Ethical guardrails**\n\n"
            "- Name/ID never used as scoring input\n"
            "- No sensitive attributes (caste, religion, gender, age)\n"
            "- Scripts blocked from competitor mentions and pressure tactics\n"
            "- All outputs include a rationale field"
        )


def render_banner(analysis_by_id: dict):
    counts = {b: 0 for b in Bucket}
    for r in analysis_by_id.values():
        counts[Bucket(r["bucket"])] += 1
    critical = counts[Bucket.CRITICAL]
    at_risk = counts[Bucket.AT_RISK]
    if critical or at_risk:
        st.markdown(
            f"<div style='padding:12px;border-radius:8px;background:#fef2f2;"
            f"border:1px solid #ef4444;color:#7f1d1d;'>"
            f"<b>⚠ {critical} customers flagged Critical · {at_risk} At-Risk · "
            f"Action recommended</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.success("All customers in Safe/Watch buckets.")


def render_dashboard(df: pd.DataFrame, analysis_by_id: dict) -> str | None:
    st.subheader("Customer dashboard")

    bucket_filter = st.radio(
        "Filter",
        options=["All", "Safe", "Watch", "At-Risk", "Critical", "Premium only"],
        horizontal=True,
    )

    df = df.copy()
    df["bucket"] = df["customer_id"].map(lambda cid: analysis_by_id[cid]["bucket"])
    df["risk_score"] = df["customer_id"].map(lambda cid: analysis_by_id[cid]["risk_score"])
    df["top_driver"] = df["customer_id"].map(
        lambda cid: analysis_by_id[cid]["top_3_drivers"][0]
    )

    if bucket_filter == "Premium only":
        df = df[df["is_premium"]]
    elif bucket_filter != "All":
        mapping = {"Safe": "SAFE", "Watch": "WATCH", "At-Risk": "AT_RISK", "Critical": "CRITICAL"}
        df = df[df["bucket"] == mapping[bucket_filter]]

    display = df[[
        "customer_id", "name", "bucket", "risk_score",
        "tenure_months", "avg_monthly_arpu_inr", "top_driver",
    ]].sort_values("risk_score", ascending=False)

    event = st.dataframe(
        display,
        use_container_width=True,
        height=420,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = event.selection.rows if event and hasattr(event, "selection") else []
    if selected_rows:
        return display.iloc[selected_rows[0]]["customer_id"]
    return None


def render_drilldown(
    df: pd.DataFrame, analysis_by_id: dict, customer_id: str
):
    row = df[df["customer_id"] == customer_id].iloc[0].to_dict()
    analysis = analysis_by_id[customer_id]
    bucket = Bucket(analysis["bucket"])

    st.divider()
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader(f"Customer {row['customer_id']} — {row['name']}")
        st.markdown(f"**Plan:** {row['plan_type']}  ·  **Premium:** {row['is_premium']}  ·  **Tenure:** {row['tenure_months']} months")
        st.markdown(f"**ARPU:** ₹{row['avg_monthly_arpu_inr']:.0f}/mo  ·  **Loyalty points:** {row['loyalty_points_balance']}")
        st.markdown(
            f"**Complaints (90d):** {row['complaints_last_90d']}  ·  "
            f"**Offers availed (180d):** {row['offers_availed_last_180d']}  ·  "
            f"**Data trend:** {row['data_usage_gb_trend']}"
        )
        st.markdown(
            f"**Network tickets:** {row['network_issue_tickets']}  ·  "
            f"**Call drops:** {row['call_drop_rate_pct']}%  ·  "
            f"**App logins (30d):** {row['app_logins_last_30d']}"
        )
        if row["port_out_request_flag"]:
            st.error("🚨 Port-out request filed")

    with col_b:
        color = BUCKET_COLOR[bucket]
        st.markdown(
            f"<div style='padding:8px 12px;border-radius:6px;"
            f"background:{color};color:white;display:inline-block;font-weight:600;'>"
            f"{bucket.value}</div>",
            unsafe_allow_html=True,
        )
        st.metric("Risk score", analysis["risk_score"], help="0–100, higher = more likely to churn")
        st.markdown("**Top drivers:**")
        for d in analysis["top_3_drivers"]:
            st.markdown(f"- {d}")
        st.markdown(f"_{analysis['rationale']}_")

    follow_ups = gate(bucket)
    disabled = AgentName.EXECUTOR not in follow_ups
    tooltip = "No retention action needed for Safe/Watch buckets" if disabled else None

    if st.button(
        "Run Retention Flow",
        type="primary",
        disabled=disabled,
        help=tooltip,
        key=f"flow_{customer_id}",
    ):
        run_retention_flow(row, analysis_by_id[customer_id])

    if customer_id in st.session_state.get("flow_results", {}):
        _render_flow_results(st.session_state["flow_results"][customer_id], bucket)


def run_retention_flow(customer: dict, analysis_dict: dict):
    from schemas import AnalystOutput
    analysis = AnalystOutput(**analysis_dict)
    with st.spinner("Generating offer…"):
        offer = generate_offer(customer, analysis)
    with st.spinner("Generating retention script…"):
        script = generate_script(customer, analysis, offer)
    try:
        audio = script_to_audio_bytes(script)
    except Exception:
        audio = None
    st.session_state.setdefault("flow_results", {})[customer["customer_id"]] = {
        "offer": offer.model_dump(),
        "script": script.model_dump(),
        "audio": audio,
    }
    st.rerun()


def _render_flow_results(result: dict, bucket: Bucket):
    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Retention Offer")
        offer = result["offer"]
        st.markdown(
            f"<div style='padding:6px 10px;border-radius:4px;background:#dbeafe;"
            f"color:#1e3a8a;display:inline-block;font-weight:600;'>{offer['offer_type']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{offer['offer_details']}**")
        st.caption(
            f"Value: ₹{offer['monetary_value_inr']}  ·  "
            f"Valid: {offer['validity_days']} days  ·  "
            f"Expected lift: {offer['expected_retention_lift']}"
        )
        st.markdown(f"_{offer['justification']}_")

    with col2:
        st.subheader("Retention Script")
        script = result["script"]
        st.markdown(f"**Opening:** {script['opening_line']}")
        st.text_area("Full script", script["full_script"], height=200, label_visibility="collapsed")
        if result["audio"] is not None:
            st.audio(result["audio"], format="audio/mp3")
        else:
            st.caption("Audio unavailable (TTS requires internet connection).")
        with st.expander("Do not say"):
            for item in script["do_not_say"]:
                st.markdown(f"- {item}")
        with st.expander("Talking points"):
            for tp in script["key_talking_points"]:
                st.markdown(f"- {tp}")

    if should_escalate(bucket):
        st.divider()
        if st.button("🚨 Escalate to retention manager", key="escalate"):
            st.toast("Escalation logged (demo). In production, this would notify the retention manager.")


def main():
    df = load_customers()
    analysis_by_id = run_analysis(df.to_dict(orient="records"))

    render_sidebar()
    st.title("Telecom Churn Reduction Agent")
    st.caption("Demo data only. Not for production retention decisions.")

    render_banner(analysis_by_id)
    selected_id = render_dashboard(df, analysis_by_id)

    if selected_id:
        st.session_state["selected_id"] = selected_id

    if selected_id:
        render_drilldown(df, analysis_by_id, selected_id)


if __name__ == "__main__":
    main()
