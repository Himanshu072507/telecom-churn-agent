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

    st.divider()
    st.caption("Select a customer above to view details (drill-down panel — Task 12).")


if __name__ == "__main__":
    main()
