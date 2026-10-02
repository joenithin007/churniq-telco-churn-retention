"""View 6: Data Quality."""

import streamlit as st
import pandas as pd
from utils.validation import inspect_data_quality
from utils.formatting import format_number


def render_data_quality(df: pd.DataFrame, cleaning_summary: dict = None) -> None:
    """Render simplified Data Quality page."""
    st.markdown("## Data Quality")
    st.caption("Verification of dataset health, schema integrity, and automated cleaning audits.")

    if df.empty:
        st.warning("⚠️ No data available to inspect.")
        return

    dq = inspect_data_quality(df)

    # 1. Overall Status Callout
    status = dq["overall_status"]
    if status == "Excellent" or "Minor" in status:
        st.success("✅ **Data Quality: Good** — Schema verified, duplicates handled, all features validated.")
    else:
        st.warning("⚠️ **Data Quality: Needs Attention** — Check detected anomalies or duplicates below.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 2. Dataset Status KPI Cards
    st.markdown("### Dataset Status")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">TOTAL ROWS</div>
                <div class="kpi-value">{format_number(dq["total_rows"])}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">COLUMNS</div>
                <div class="kpi-value">{dq["total_columns"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">MISSING CELLS</div>
                <div class="kpi-value">{dq["total_missing_cells"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">DUPLICATE IDS</div>
                <div class="kpi-value">{dq["duplicate_ids"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">INVALID VALUES</div>
                <div class="kpi-value">{len(dq["anomalies"])}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m6:
        target_str = "Available" if dq["has_target"] else "Missing"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">TARGET AVAILABILITY</div>
                <div class="kpi-value" style="color: {'#34D399' if dq['has_target'] else '#EF4444'};">{target_str}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 3. Anomaly Checks
    if dq["anomalies"]:
        st.markdown("### Detected Value Checks")
        for anom in dq["anomalies"]:
            st.markdown(f"- ⚠️ **{anom['column']}**: {anom['issue']}")
    else:
        st.markdown("✓ **Zero out-of-bounds or invalid values detected.** All features pass verification.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 4. Schema Breakdown Table
    st.markdown("### Feature Schema Breakdown")
    cols_data = []
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        cols_data.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Missing Records": n_missing,
            "Unique Count": int(df[col].nunique()),
            "Sample Value": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "None"
        })
    df_schema = pd.DataFrame(cols_data)
    st.dataframe(df_schema, use_container_width=True, hide_index=True)
