"""View 2: Customer Risk Investigation."""

import streamlit as st
import pandas as pd
from utils.formatting import format_currency, format_percentage, format_number, get_risk_badge, get_priority_badge
from src.explainability import ModelExplainer, plot_customer_explanation_waterfall


def render_customer_risk(df: pd.DataFrame, model_explainer: ModelExplainer = None) -> None:
    """Render simplified Customer Risk page."""
    st.markdown("## Customer Risk")
    st.caption("Find and understand customers with elevated model-predicted churn risk.")

    if df.empty:
        st.warning("⚠️ No customer records available.")
        return

    # 1. Search Customer ID
    all_cust_ids = df["Customer_ID"].tolist() if "Customer_ID" in df.columns else [f"CUST-{i}" for i in range(len(df))]
    selected_id = st.selectbox(
        "Search Customer ID:",
        options=all_cust_ids,
        index=0,
        help="Select or type a Customer ID to inspect risk analysis."
    )

    cust = df[df["Customer_ID"] == selected_id].iloc[0] if "Customer_ID" in df.columns else df.iloc[0]
    prob = cust.get("Churn_Probability", 0.0)
    risk_level = cust.get("Risk_Level", "Low")
    cust_val = cust.get("Customer_Value", 0.0)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 2. Three Compact Cards: Churn Probability | Risk Level | Customer Value
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">CHURN PROBABILITY</div>
                <div class="kpi-value" style="color: {'#EF4444' if prob > 0.7 else ('#F59E0B' if prob > 0.3 else '#10B981')};">
                    {prob * 100:.1f}%
                </div>
                <div class="kpi-sub">Model-estimated probability</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">RISK LEVEL</div>
                <div class="kpi-value">{risk_level} Risk</div>
                <div class="kpi-sub">Priority: {cust.get('Priority', 'P4')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">ESTIMATED CUSTOMER VALUE</div>
                <div class="kpi-value" style="color: #818CF8;">{format_currency(cust_val)}</div>
                <div class="kpi-sub">Realized + Forward Run-rate</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 3. Why is this customer at risk? (Top 3-5 factors)
    st.markdown("### Why is this customer at risk?")
    if model_explainer is not None:
        contributions = model_explainer.explain_individual(cust, df, top_k=5)
        if contributions:
            col_chart, col_reasons = st.columns([3, 2])
            with col_chart:
                st.plotly_chart(
                    plot_customer_explanation_waterfall(contributions, str(cust.get("Customer_ID", selected_id))),
                    use_container_width=True
                )
            with col_reasons:
                st.markdown("**Top Model Factors:**")
                for c in contributions:
                    icon = "🔴" if "Increase" in c["Direction"] or "Key" in c["Direction"] else "🟢"
                    st.markdown(f"- {icon} **{c['Display_Feature']}**: {c['Direction']}")
                st.caption("ℹ️ Model factors show statistical associations, not causal guarantees.")
        else:
            st.info("Individual factor attribution unavailable for this customer.")
    else:
        st.info("Train or load a model to view personalized risk factors.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 4. Customer Details (Important fields only)
    st.markdown("### Customer Details")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(f"**Tenure:** {cust.get('Tenure', 0)} months")
        st.markdown(f"**Contract:** `{cust.get('Contract_Type', 'N/A')}`")
    with d2:
        st.markdown(f"**Monthly Charges:** {format_currency(cust.get('Monthly_Charges', 0))}")
        st.markdown(f"**Satisfaction:** {'⭐' * int(cust.get('Satisfaction_Score', 3))} ({cust.get('Satisfaction_Score', 3)}/5)")
    with d3:
        st.markdown(f"**Complaints:** {cust.get('Complaints', 0)}")
        st.markdown(f"**Payment Failures:** {cust.get('Payment_Failures', 0)}")

    st.markdown(f"💡 **Suggested Review Area:** {cust.get('Suggested_Review_Areas', 'Standard account monitoring')}")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 5. Customer Risk Portfolio Table
    st.markdown("### Customer Portfolio Table")
    display_cols = ["Customer_ID", "Risk_Level", "Churn_Probability", "Monthly_Charges", "Tenure", "Contract_Type", "Customer_Value", "Priority"]
    avail_cols = [c for c in display_cols if c in df.columns]

    st_c1, st_c2 = st.columns([3, 1])
    with st_c1:
        st.caption(f"Displaying {min(len(df), 300):,} customers in active portfolio.")
    with st_c2:
        csv_data = df[avail_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Risk List (CSV)",
            data=csv_data,
            file_name="customer_risk_list.csv",
            mime="text/csv",
        )

    st.dataframe(
        df[avail_cols].sort_values(by="Churn_Probability", ascending=False).head(300),
        use_container_width=True,
        hide_index=True
    )
