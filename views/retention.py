"""View 4: Retention Intelligence."""

import streamlit as st
import pandas as pd
from utils.formatting import format_currency, format_number
from src.retention import RetentionEngine


def render_retention(df: pd.DataFrame, retention_engine: RetentionEngine = None) -> None:
    """Render simplified Retention Intelligence page."""
    st.markdown("## Retention Intelligence")
    st.caption("Identify customers who may warrant further retention review.")

    if df.empty:
        st.warning("⚠️ No customer records available.")
        return

    if retention_engine is None:
        retention_engine = RetentionEngine()

    quad_counts = retention_engine.get_quadrant_summary(df)

    # 1. Three Simple Groups
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 3px solid #EF4444;">
                <div class="kpi-label">HIGH RISK + HIGH VALUE</div>
                <div class="kpi-value" style="color: #F87171;">{format_number(quad_counts.get('High Risk + High Value', 0))}</div>
                <div class="kpi-sub">Priority 1 • Immediate review</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 3px solid #F59E0B;">
                <div class="kpi-label">HIGH RISK + MEDIUM VALUE</div>
                <div class="kpi-value" style="color: #FBBF24;">{format_number(quad_counts.get('High Risk + Medium Value', 0))}</div>
                <div class="kpi-sub">Priority 2 • Targeted outreach</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 3px solid #64748B;">
                <div class="kpi-label">HIGH RISK + LOW VALUE</div>
                <div class="kpi-value" style="color: #CBD5E1;">{format_number(quad_counts.get('High Risk + Low Value', 0))}</div>
                <div class="kpi-sub">Priority 3 • Digital campaigns</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 2. Priority Customer List Table
    st.markdown("### Priority Customer List")
    
    # Optional compact filter by priority
    prio_choice = st.selectbox(
        "Filter Priority Tier:",
        options=["All Prioritized", "P1 - Critical", "P2 - High", "P3 - Medium"],
        index=0
    )

    df_prio = df.copy()
    if prio_choice != "All Prioritized" and "Priority" in df_prio.columns:
        df_prio = df_prio[df_prio["Priority"] == prio_choice]

    cols_order = ["Customer_ID", "Risk_Level", "Churn_Probability", "Customer_Value", "Main_Risk_Factors", "Suggested_Review_Areas"]
    avail_cols = [c for c in cols_order if c in df_prio.columns]

    st_c1, st_c2 = st.columns([3, 1])
    with st_c1:
        st.caption(f"Showing **{len(df_prio):,}** customers in prioritized review queue.")
    with st_c2:
        csv_data = df_prio[avail_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Queue (CSV)",
            data=csv_data,
            file_name="retention_priority_queue.csv",
            mime="text/csv",
        )

    # Formatted display dataframe
    df_display = df_prio[avail_cols].sort_values(by="Churn_Probability", ascending=False).head(300).copy()
    if "Churn_Probability" in df_display.columns:
        df_display["Churn_Probability"] = df_display["Churn_Probability"].apply(lambda p: f"{p*100:.1f}%")
    if "Customer_Value" in df_display.columns:
        df_display["Customer_Value"] = df_display["Customer_Value"].apply(lambda v: format_currency(v))
    
    # Rename columns to match prompt: Customer | Risk | Probability | Customer Value | Main Risk Factor | Suggested Review Area
    rename_dict = {
        "Customer_ID": "Customer",
        "Risk_Level": "Risk",
        "Churn_Probability": "Probability",
        "Customer_Value": "Customer Value",
        "Main_Risk_Factors": "Main Risk Factor",
        "Suggested_Review_Areas": "Suggested Review Area"
    }
    df_display = df_display.rename(columns=rename_dict)

    st.dataframe(df_display, use_container_width=True, hide_index=True)
