"""Customer Churn Prediction & Retention Intelligence Dashboard.

Predict customer churn. Understand risk. Prioritize retention.
"""

import os
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Dark Enterprise Theme CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit default multipage navigation completely */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Sidebar Background & Width */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
        min-width: 260px !important;
    }
    section[data-testid="stSidebar"] div.block-container {
        padding-top: 1.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    /* Sidebar Clean Navigation Buttons */
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 9px 14px !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        border: 1px solid transparent !important;
        margin-bottom: 4px !important;
        color: #94A3B8 !important;
        background-color: transparent !important;
        transition: background 0.15s ease, color 0.15s ease;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        color: #F8FAFC !important;
        background-color: #1E293B !important;
        border-color: #334155 !important;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background-color: rgba(99, 102, 241, 0.18) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: 1px solid #6366F1 !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1380px !important;
    }

    /* High-Contrast KPI Cards */
    .kpi-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 18px;
        min-height: 105px;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.2;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 0.80rem;
        color: #64748B;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        letter-spacing: -0.02em;
    }

    /* Filter Bar Styling */
    div[data-testid="stHorizontalBlock"] {
        align-items: flex-end;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Core pipeline imports
from src.data_loader import load_dataset
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.train_model import ModelTrainer, load_saved_model
from src.predict import predict_churn
from src.risk_segmentation import RiskSegmenter
from src.retention import RetentionEngine
from src.explainability import ModelExplainer
from utils.validation import map_columns

# Page Views
from views.overview import render_overview
from views.customer_risk import render_customer_risk
from views.churn_drivers import render_churn_drivers
from views.retention import render_retention
from views.model_performance import render_model_performance
from views.data_quality import render_data_quality


def initialize_app_pipeline():
    """Ensure baseline dataset, model, and predictions exist on startup."""
    if "pipeline_ready" in st.session_state and st.session_state.pipeline_ready:
        return

    # 1. Load or Generate Sample Data
    df_raw = load_dataset()
    st.session_state.df_raw = df_raw

    # 2. Clean Data
    cleaner = DataCleaner()
    df_clean, cleaning_summary = cleaner.clean(df_raw)
    st.session_state.df_clean = df_clean
    st.session_state.cleaning_summary = cleaning_summary

    # 3. Engineer Features
    engineer = FeatureEngineer()
    df_feat = engineer.transform(df_clean)

    # 4. Check or Train Model
    model, preprocessor, metadata = load_saved_model()
    if model is None:
        trainer = ModelTrainer(test_size=0.20, random_state=42)
        train_res = trainer.train_and_evaluate_all(df_feat)
        model = train_res["best_model"]
        preprocessor = train_res["preprocessor"]
        feature_names = train_res["feature_names"]
        metadata = {
            "model_type": train_res["best_model_name"],
            "model_version": "1.0.0",
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": train_res["chosen_metrics"],
            "comparison": train_res["all_metrics"],
            "feature_names": feature_names,
        }
    else:
        feature_names = metadata.get("feature_names", [])

    st.session_state.model = model
    st.session_state.preprocessor = preprocessor
    st.session_state.metadata = metadata

    # 5. Initialize Explainer
    explainer = ModelExplainer(model, preprocessor, feature_names)
    st.session_state.model_explainer = explainer

    # 6. Predict and Segment Churn Risk
    segmenter = RiskSegmenter(low_threshold=0.30, high_threshold=0.70)
    retention_engine = RetentionEngine()
    df_predicted = predict_churn(df_feat, model, preprocessor, segmenter, retention_engine)
    
    st.session_state.df_predicted = df_predicted
    st.session_state.segmenter = segmenter
    st.session_state.retention_engine = retention_engine
    st.session_state.pipeline_ready = True


initialize_app_pipeline()


# --- SIDEBAR (Section 3, 4, 5, 6, 22, 23, 24, 33) ---
with st.sidebar:
    st.markdown("### CUSTOMER CHURN")
    st.markdown("<span style='color: #818CF8; font-size: 0.88rem; font-weight: 600;'>Intelligence Dashboard</span>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 0; border-top: 1px solid #1E293B; margin: 12px 0 16px 0;'>", unsafe_allow_html=True)

    # Navigation Items (Section 3 & 4)
    NAV_PAGES = [
        ("📊  Overview", "overview"),
        ("👤  Customer Risk", "customer_risk"),
        ("📈  Churn Drivers", "churn_drivers"),
        ("🎯  Retention", "retention"),
        ("🤖  Model Performance", "model_performance"),
        ("✓  Data Quality", "data_quality"),
    ]

    if "active_page" not in st.session_state:
        st.session_state.active_page = "overview"

    for label, page_key in NAV_PAGES:
        is_active = (st.session_state.active_page == page_key)
        if st.button(
            label,
            key=f"nav_btn_{page_key}",
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_page = page_key
            st.rerun()

    st.markdown("<hr style='border: 0; border-top: 1px solid #1E293B; margin: 24px 0 16px 0;'>", unsafe_allow_html=True)

    # Compact Data Source Manager
    with st.expander("📁 Data Source", expanded=False):
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("Process CSV", use_container_width=True, type="primary"):
                with st.spinner("Processing..."):
                    try:
                        raw_upload = pd.read_csv(uploaded_file)
                        mapped_df, _ = map_columns(raw_upload)
                        cleaner = DataCleaner()
                        cleaned_df, c_summary = cleaner.clean(mapped_df)
                        engineer = FeatureEngineer()
                        feat_df = engineer.transform(cleaned_df)
                        df_pred = predict_churn(
                            feat_df,
                            st.session_state.model,
                            st.session_state.preprocessor,
                            st.session_state.segmenter,
                            st.session_state.retention_engine
                        )
                        st.session_state.df_raw = raw_upload
                        st.session_state.df_clean = cleaned_df
                        st.session_state.cleaning_summary = c_summary
                        st.session_state.df_predicted = df_pred
                        st.success(f"Loaded {len(df_pred):,} records.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        if st.button("Reset to Benchmark", use_container_width=True):
            st.session_state.pipeline_ready = False
            initialize_app_pipeline()
            st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Sidebar Footer (Section 6 & 33)
    meta = st.session_state.get("metadata", {})
    st.markdown(
        f"""
        <div style="font-size: 0.80rem; color: #64748B; line-height: 1.5;">
            <b style="color: #94A3B8;">Customer Churn Intelligence</b><br>
            Predict • Explain • Retain<br>
            <div style="margin-top: 6px;">
                <b>Model:</b> <span style="color: #818CF8;">{meta.get('model_type', 'Logistic Regression')}</span><br>
                <b>Status:</b> <span style="color: #34D399;">Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# --- MAIN HEADER BAR (Section 8 & 29) ---
head_col1, head_col2 = st.columns([3.5, 0.8])
with head_col1:
    st.markdown("# Customer Churn Prediction & Retention Intelligence")
    st.markdown("<p style='color: #94A3B8; font-size: 1rem; margin-top: -8px;'>Predict customer churn. Understand risk. Prioritize retention.</p>", unsafe_allow_html=True)

with head_col2:
    with st.popover("ⓘ Definitions"):
        st.markdown(
            """
            ### Business & Analytical Definitions
            * **Churn Rate:** Percentage of customers who have discontinued service (`Churn = 1`).
            * **Churn Probability:** Statistical probability that a customer belongs to the churn class (0.0 to 1.0).
            * **High Risk:** Accounts with model-predicted churn probability > 70%.
            * **Revenue at Risk:** Estimated annualized revenue exposed to churn (Sum of Monthly Charges × 12 × Churn Probability).
            * **Customer Value:** Historical total charges plus forward 12-month run-rate revenue.
            """
        )

st.markdown("<hr style='border: 0; border-top: 1px solid #1E293B; margin: 4px 0 16px 0;'>", unsafe_allow_html=True)


# --- COMPACT HORIZONTAL FILTER BAR (Section 20, 21, 34) ---
# Initialize filter session state
if "filter_risk" not in st.session_state:
    st.session_state.filter_risk = "All"
if "filter_contract" not in st.session_state:
    st.session_state.filter_contract = "All"
if "filter_location" not in st.session_state:
    st.session_state.filter_location = "All"
if "filter_tenure" not in st.session_state:
    st.session_state.filter_tenure = "All"

df_full = st.session_state.df_predicted

# Filter Options
risk_options = ["All"] + sorted([str(x) for x in df_full["Risk_Level"].dropna().unique()]) if "Risk_Level" in df_full.columns else ["All"]
contract_options = ["All"] + sorted([str(x) for x in df_full["Contract_Type"].dropna().unique()]) if "Contract_Type" in df_full.columns else ["All"]
loc_options = ["All"] + sorted([str(x) for x in df_full["Location"].dropna().unique()]) if "Location" in df_full.columns else ["All"]
tenure_options = ["All", "0–6 months", "7–12 months", "13–24 months", "25+ months"]

fc1, fc2, fc3, fc4, fc5 = st.columns([1.2, 1.2, 1.2, 1.2, 0.8])

with fc1:
    f_risk = st.selectbox("Risk Level", risk_options, index=risk_options.index(st.session_state.filter_risk) if st.session_state.filter_risk in risk_options else 0, key="sb_risk")
    st.session_state.filter_risk = f_risk

with fc2:
    f_contract = st.selectbox("Contract", contract_options, index=contract_options.index(st.session_state.filter_contract) if st.session_state.filter_contract in contract_options else 0, key="sb_contract")
    st.session_state.filter_contract = f_contract

with fc3:
    f_loc = st.selectbox("Location", loc_options, index=loc_options.index(st.session_state.filter_location) if st.session_state.filter_location in loc_options else 0, key="sb_loc")
    st.session_state.filter_location = f_loc

with fc4:
    f_tenure = st.selectbox("Tenure", tenure_options, index=tenure_options.index(st.session_state.filter_tenure) if st.session_state.filter_tenure in tenure_options else 0, key="sb_tenure")
    st.session_state.filter_tenure = f_tenure

with fc5:
    if st.button("Reset", key="reset_filters_btn", use_container_width=True):
        st.session_state.filter_risk = "All"
        st.session_state.filter_contract = "All"
        st.session_state.filter_location = "All"
        st.session_state.filter_tenure = "All"
        st.rerun()

# Apply filters
df_active = df_full.copy()
if st.session_state.filter_risk != "All" and "Risk_Level" in df_active.columns:
    df_active = df_active[df_active["Risk_Level"] == st.session_state.filter_risk]

if st.session_state.filter_contract != "All" and "Contract_Type" in df_active.columns:
    df_active = df_active[df_active["Contract_Type"] == st.session_state.filter_contract]

if st.session_state.filter_location != "All" and "Location" in df_active.columns:
    df_active = df_active[df_active["Location"] == st.session_state.filter_location]

if st.session_state.filter_tenure != "All" and "Tenure" in df_active.columns:
    if st.session_state.filter_tenure == "0–6 months":
        df_active = df_active[df_active["Tenure"] <= 6]
    elif st.session_state.filter_tenure == "7–12 months":
        df_active = df_active[(df_active["Tenure"] > 6) & (df_active["Tenure"] <= 12)]
    elif st.session_state.filter_tenure == "13–24 months":
        df_active = df_active[(df_active["Tenure"] > 12) & (df_active["Tenure"] <= 24)]
    elif st.session_state.filter_tenure == "25+ months":
        df_active = df_active[df_active["Tenure"] > 24]

st.markdown("<hr style='border: 0; border-top: 1px solid #1E293B; margin: 12px 0 18px 0;'>", unsafe_allow_html=True)


# --- ROUTING TO ACTIVE VIEW ---
active_page = st.session_state.active_page

if active_page == "overview":
    render_overview(df_active, st.session_state.get("model_explainer"))

elif active_page == "customer_risk":
    render_customer_risk(df_active, st.session_state.get("model_explainer"))

elif active_page == "churn_drivers":
    render_churn_drivers(df_active, st.session_state.get("model_explainer"))

elif active_page == "retention":
    render_retention(df_active, st.session_state.get("retention_engine"))

elif active_page == "model_performance":
    meta = st.session_state.get("metadata", {})
    def on_model_retrained(train_results):
        st.session_state.model = train_results["best_model"]
        st.session_state.preprocessor = train_results["preprocessor"]
        st.session_state.metadata = {
            "model_type": train_results["best_model_name"],
            "model_version": "1.0.0",
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": train_results["chosen_metrics"],
            "comparison": train_results["all_metrics"],
            "feature_names": train_results["feature_names"],
        }
        st.session_state.model_explainer = ModelExplainer(
            train_results["best_model"],
            train_results["preprocessor"],
            train_results["feature_names"]
        )
        df_pred_new = predict_churn(
            st.session_state.df_clean,
            train_results["best_model"],
            train_results["preprocessor"],
            st.session_state.segmenter,
            st.session_state.retention_engine
        )
        st.session_state.df_predicted = df_pred_new

    render_model_performance(
        active_metrics=meta.get("metrics"),
        comparison_metrics=meta.get("comparison"),
        model_name=meta.get("model_type", "Selected Model"),
        df=st.session_state.get("df_clean"),
        on_retrain_callback=on_model_retrained
    )

elif active_page == "data_quality":
    render_data_quality(
        st.session_state.get("df_raw", df_active),
        st.session_state.get("cleaning_summary")
    )
