"""View 5: Model Performance."""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
from utils.formatting import format_percentage
from src.evaluate_model import (
    plot_confusion_matrix,
    plot_roc_curve,
    format_model_comparison_table,
)
from src.train_model import ModelTrainer


def render_model_performance(
    active_metrics: Optional[Dict[str, Any]],
    comparison_metrics: Optional[List[Dict[str, Any]]],
    model_name: str = "Trained Model",
    df: Optional[pd.DataFrame] = None,
    on_retrain_callback = None
) -> None:
    """Render technical Model Performance evaluation page."""
    st.markdown("## Model Performance")
    st.caption("Technical model evaluation strictly assessed on holdout test data (80/20 stratified split).")

    if not active_metrics:
        st.warning("⚠️ No active model evaluation metrics found.")
        return

    # 1. Selected Model & 5 Core Metrics Cards
    st.markdown(f"#### Active Model: **{model_name}**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">ROC-AUC</div>
                <div class="kpi-value" style="color: #818CF8;">{active_metrics.get('roc_auc', 0.0):.3f}</div>
                <div class="kpi-sub">Discrimination power</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">RECALL (CHURN)</div>
                <div class="kpi-value" style="color: #34D399;">{format_percentage(active_metrics.get('recall', 0.0))}</div>
                <div class="kpi-sub">True churners caught</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">PRECISION</div>
                <div class="kpi-value">{format_percentage(active_metrics.get('precision', 0.0))}</div>
                <div class="kpi-sub">Positive accuracy</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">F1 SCORE</div>
                <div class="kpi-value">{format_percentage(active_metrics.get('f1', 0.0))}</div>
                <div class="kpi-sub">Harmonic balance</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">ACCURACY</div>
                <div class="kpi-value">{format_percentage(active_metrics.get('accuracy', 0.0))}</div>
                <div class="kpi-sub">Overall correct</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Confusion Matrix & ROC Curve
    m_c1, m_c2 = st.columns(2)
    with m_c1:
        cm = active_metrics.get("confusion_matrix")
        if cm:
            st.plotly_chart(plot_confusion_matrix(cm, model_name=model_name), use_container_width=True)
    with m_c2:
        fpr = active_metrics.get("fpr")
        tpr = active_metrics.get("tpr")
        auc = active_metrics.get("roc_auc", 0.5)
        if fpr and tpr:
            st.plotly_chart(plot_roc_curve(fpr, tpr, auc, model_name=model_name), use_container_width=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 3. Model Comparison Table
    st.markdown("### Model Comparison")
    st.caption("Side-by-side benchmark comparison on identical stratified test holdout.")
    if comparison_metrics:
        df_comp = format_model_comparison_table(comparison_metrics)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # Retrain expander
    with st.expander("Model Calibration & Retraining", expanded=False):
        if df is not None and not df.empty and "Churn" in df.columns:
            r1, r2 = st.columns(2)
            with r1:
                test_size_choice = st.slider("Holdout Test Size (%)", min_value=10, max_value=30, value=20, step=5) / 100.0
            with r2:
                model_choice = st.selectbox(
                    "Model Algorithm:",
                    options=["Automated Best", "Logistic Regression", "Random Forest", "XGBoost"],
                    index=0
                )
            if st.button("Re-train Models", type="primary"):
                with st.spinner("Training models..."):
                    trainer = ModelTrainer(test_size=test_size_choice, random_state=42)
                    pref = None if "Automated" in model_choice else model_choice
                    results = trainer.train_and_evaluate_all(df, selected_model_type=pref)
                    st.success(f"✓ Model trained: {results['best_model_name']} (ROC-AUC: {results['chosen_metrics']['roc_auc']:.3f})")
                    if on_retrain_callback:
                        on_retrain_callback(results)
                        st.rerun()
