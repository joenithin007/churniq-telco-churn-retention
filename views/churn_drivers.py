"""View 3: Churn Drivers."""

import streamlit as st
import pandas as pd
from src.eda import (
    plot_churn_rate_by_category,
    plot_churn_distribution_numerical,
)
from src.explainability import ModelExplainer, plot_global_feature_importance


def render_churn_drivers(df: pd.DataFrame, model_explainer: ModelExplainer = None) -> None:
    """Render simplified Churn Drivers page answering 'Why are customers at risk?'."""
    st.markdown("## Why are customers at risk?")
    st.caption("Key empirical and model-identified factors associated with customer churn.")

    if df.empty:
        st.warning("⚠️ No data available.")
        return

    # Visual 1 & 2: Top Model Features & Churn by Contract
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        if model_explainer is not None:
            df_imp = model_explainer.get_global_importance(df.sample(min(len(df), 500), random_state=42), max_features=8)
            st.plotly_chart(plot_global_feature_importance(df_imp), use_container_width=True)
        else:
            st.info("Train or load a model to view top model features.")

    with row1_c2:
        if "Contract_Type" in df.columns:
            st.plotly_chart(plot_churn_rate_by_category(df, "Contract_Type", "Churn Rate by Contract Type"), use_container_width=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Visual 3 & 4: Churn by Tenure & Churn by Satisfaction
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        if "Tenure_Group" in df.columns:
            st.plotly_chart(plot_churn_rate_by_category(df, "Tenure_Group", "Churn Rate by Tenure Cohort"), use_container_width=True)
        elif "Tenure" in df.columns:
            st.plotly_chart(plot_churn_distribution_numerical(df, "Tenure", "Tenure Distribution (Months)"), use_container_width=True)

    with row2_c2:
        if "Satisfaction_Score" in df.columns:
            st.plotly_chart(plot_churn_rate_by_category(df, "Satisfaction_Score", "Churn Rate by Satisfaction Score (1-5)"), use_container_width=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Visual 5: Churn by Monthly Charges
    if "Monthly_Charges" in df.columns:
        st.plotly_chart(plot_churn_distribution_numerical(df, "Monthly_Charges", "Monthly Charges Distribution (Retained vs Churned)"), use_container_width=True)
