"""Unit tests for analytical calculations, customer value proxy, and KPI summaries."""

import pytest
import pandas as pd
import numpy as np
from utils.calculations import compute_kpis, categorize_risk, calculate_customer_value_proxy, assign_priority
from src.retention import RetentionEngine
from utils.formatting import format_currency, format_percentage


def test_categorize_risk():
    """Verify threshold boundary assignment for risk tiers."""
    assert categorize_risk(0.15) == "Low"
    assert categorize_risk(0.30) == "Low"
    assert categorize_risk(0.31) == "Medium"
    assert categorize_risk(0.70) == "Medium"
    assert categorize_risk(0.71) == "High"
    assert categorize_risk(0.99) == "High"


def test_customer_value_proxy():
    """Verify transparent customer value proxy formula."""
    df = pd.DataFrame({
        "Total_Charges": [12000.0, 50000.0],
        "Monthly_Charges": [1000.0, 2500.0],
    })
    val = calculate_customer_value_proxy(df)
    # Total_Charges + (Monthly_Charges * 12)
    assert val.iloc[0] == 12000.0 + (1000.0 * 12) == 24000.0
    assert val.iloc[1] == 50000.0 + (2500.0 * 12) == 80000.0


def test_retention_priority_assignment():
    """Verify priority mapping for high-risk and high-value customers."""
    engine = RetentionEngine(value_threshold_high=80000.0, value_threshold_med=30000.0)

    # High Risk + High Value -> P1
    row_p1 = pd.Series({
        "Churn_Probability": 0.85,
        "Customer_Value": 95000.0,
        "Contract_Type": "Month-to-Month",
        "Monthly_Charges": 2200.0,
        "Complaints": 3,
        "Satisfaction_Score": 2,
    })
    factors, reviews = engine.generate_review_areas_and_factors(row_p1)
    assert "Month-to-month" in factors
    assert "Customer Support" in reviews or "Pricing" in reviews

    prio, _ = assign_priority(prob=0.85, value=95000.0, value_quantiles=(30000.0, 80000.0))
    assert prio == "P1 - Critical"

    # Low Risk -> P4
    prio_low, _ = assign_priority(prob=0.10, value=20000.0, value_quantiles=(30000.0, 80000.0))
    assert prio_low == "P4 - Low"


def test_compute_kpis():
    """Verify KPI calculations with non-negative, accurate outputs."""
    df = pd.DataFrame({
        "Churn": [0, 0, 1, 1],
        "Risk_Level": ["Low", "Low", "Medium", "High"],
        "Churn_Probability": [0.10, 0.20, 0.50, 0.85],
        "Monthly_Charges": [1000.0, 1200.0, 1500.0, 2000.0],
        "Tenure": [12, 24, 6, 3],
        "Satisfaction_Score": [5, 4, 3, 1],
    })
    kpis = compute_kpis(df)
    assert kpis["total_customers"] == 4
    assert kpis["churned_customers"] == 2
    assert kpis["churn_rate"] == 50.0
    assert kpis["high_risk_count"] == 1
    assert kpis["high_risk_pct"] == 25.0
    assert kpis["avg_monthly_charges"] == 1425.0
    assert kpis["revenue_at_risk_annual"] > 0.0


def test_formatting_utils():
    """Verify Indian currency and percentage formatting."""
    assert "₹" in format_currency(125000)
    assert format_percentage(0.187) == "18.7%"
    assert format_percentage(18.7) == "18.7%"
