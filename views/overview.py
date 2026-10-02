"""View 1: Executive Overview."""

import streamlit as st
import pandas as pd
from utils.formatting import format_currency, format_percentage, format_number
from utils.calculations import compute_kpis
from src.eda import (
    plot_churn_distribution,
    plot_risk_distribution,
    plot_top_churn_drivers_overview,
    get_empty_chart,
)
from src.explainability import ModelExplainer


def render_overview(df: pd.DataFrame, model_explainer: ModelExplainer = None) -> None:
    """Render the simplified, high-impact Executive Overview dashboard page."""
    if df.empty:
        st.warning("⚠️ No data available. Please load the sample dataset or upload a CSV file.")
        return

    # Compute KPIs dynamically from data
    kpis = compute_kpis(df)

    # Section 1: KPI Cards (Compact, High-Contrast Dark Theme)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">TOTAL CUSTOMERS</div>
                <div class="kpi-value">{format_number(kpis["total_customers"])}</div>
                <div class="kpi-sub">Active evaluated accounts</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">CHURN RATE</div>
                <div class="kpi-value" style="color: #F87171;">{format_percentage(kpis["churn_rate"])}</div>
                <div class="kpi-sub">{format_number(kpis["churned_customers"])} churned historical accounts</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">HIGH-RISK CUSTOMERS</div>
                <div class="kpi-value" style="color: #FB923C;">{format_number(kpis["high_risk_count"])}</div>
                <div class="kpi-sub">{kpis["high_risk_pct"]:.1f}% of current customer base</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">ESTIMATED REVENUE AT RISK</div>
                <div class="kpi-value" style="color: #FBBF24;">{format_currency(kpis["revenue_at_risk_annual"], compact=True)}</div>
                <div class="kpi-sub">~{format_currency(kpis["revenue_at_risk_monthly"], compact=True)} monthly run-rate</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Section 2: Visual 1 (Churn Distribution) & Visual 2 (Risk Segmentation)
    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        st.plotly_chart(plot_churn_distribution(df), use_container_width=True)

    with col_vis2:
        st.plotly_chart(plot_risk_distribution(df), use_container_width=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Section 3: Visual 3 (Top 5 Churn Drivers)
    if model_explainer is not None:
        df_imp = model_explainer.get_global_importance(df.sample(min(len(df), 500), random_state=42), max_features=5)
        st.plotly_chart(plot_top_churn_drivers_overview(df_imp, top_n=5), use_container_width=True)
    else:
        st.info("Train or load a model to view top churn drivers.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Section 4: High-Risk Customer Snapshot
    st.markdown("#### 🚨 High-Risk Customer Snapshot")
    st.caption("Top elevated-risk customers prioritized for immediate business review.")

    cols_snapshot = ["Customer_ID", "Risk_Level", "Churn_Probability", "Customer_Value", "Main_Risk_Factors"]
    avail_cols = [c for c in cols_snapshot if c in df.columns]

    if "Churn_Probability" in df.columns:
        df_high = df[df["Risk_Level"] == "High"] if "Risk_Level" in df.columns else df
        if df_high.empty:
            df_high = df
        snapshot_df = df_high[avail_cols].sort_values(by="Churn_Probability", ascending=False).head(8).copy()
        
        # Format for clean display
        if "Churn_Probability" in snapshot_df.columns:
            snapshot_df["Churn_Probability"] = snapshot_df["Churn_Probability"].apply(lambda p: f"{p*100:.1f}%")
        if "Customer_Value" in snapshot_df.columns:
            snapshot_df["Customer_Value"] = snapshot_df["Customer_Value"].apply(lambda v: format_currency(v))
        
        snapshot_df.columns = [c.replace("_", " ") for c in snapshot_df.columns]
        st.dataframe(snapshot_df, use_container_width=True, hide_index=True)
