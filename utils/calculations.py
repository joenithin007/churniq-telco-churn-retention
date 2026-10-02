"""Analytical calculations for Customer Churn Prediction & Retention Intelligence Dashboard."""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def compute_kpis(df: pd.DataFrame, risk_col: str = "Risk_Level", prob_col: str = "Churn_Probability") -> Dict[str, Any]:
    """Compute high-level executive KPIs from current dataframe.
    
    All calculations are derived dynamically from the dataset columns.
    """
    total_customers = len(df)
    if total_customers == 0:
        return {
            "total_customers": 0,
            "churned_customers": 0,
            "retained_customers": 0,
            "churn_rate": 0.0,
            "high_risk_count": 0,
            "high_risk_pct": 0.0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "avg_monthly_charges": 0.0,
            "avg_tenure": 0.0,
            "avg_satisfaction": 0.0,
            "total_monthly_revenue": 0.0,
            "revenue_at_risk_annual": 0.0,
            "revenue_at_risk_monthly": 0.0,
        }

    # Historical target if present
    churn_col = "Churn" if "Churn" in df.columns else None
    if churn_col:
        churned_count = int((df[churn_col] == 1).sum())
        retained_count = int((df[churn_col] == 0).sum())
        churn_rate = (churned_count / total_customers) * 100.0
    else:
        churned_count = 0
        retained_count = total_customers
        churn_rate = 0.0

    # Risk segment counts
    if risk_col in df.columns:
        high_risk_count = int((df[risk_col] == "High").sum())
        medium_risk_count = int((df[risk_col] == "Medium").sum())
        low_risk_count = int((df[risk_col] == "Low").sum())
    else:
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
    high_risk_pct = (high_risk_count / total_customers * 100.0) if total_customers > 0 else 0.0

    # Averages
    monthly_charge_col = "Monthly_Charges" if "Monthly_Charges" in df.columns else None
    avg_monthly_charges = float(df[monthly_charge_col].mean()) if monthly_charge_col else 0.0
    total_monthly_revenue = float(df[monthly_charge_col].sum()) if monthly_charge_col else 0.0

    tenure_col = "Tenure" if "Tenure" in df.columns else None
    avg_tenure = float(df[tenure_col].mean()) if tenure_col else 0.0

    sat_col = "Satisfaction_Score" if "Satisfaction_Score" in df.columns else None
    avg_satisfaction = float(df[sat_col].mean()) if sat_col else 0.0

    # Revenue at Risk (Analytical estimate)
    # Calculation: Sum of (Monthly_Charges * 12 * Churn_Probability) for each customer.
    # If Churn_Probability is not yet available, fallback to High-Risk sum.
    if monthly_charge_col and prob_col in df.columns:
        expected_monthly_loss = (df[monthly_charge_col] * df[prob_col]).sum()
        revenue_at_risk_monthly = float(expected_monthly_loss)
        revenue_at_risk_annual = float(expected_monthly_loss * 12.0)
    elif monthly_charge_col and risk_col in df.columns:
        high_risk_monthly = df[df[risk_col] == "High"][monthly_charge_col].sum()
        revenue_at_risk_monthly = float(high_risk_monthly)
        revenue_at_risk_annual = float(high_risk_monthly * 12.0)
    else:
        revenue_at_risk_monthly = 0.0
        revenue_at_risk_annual = 0.0

    return {
        "total_customers": total_customers,
        "churned_customers": churned_count,
        "retained_customers": retained_count,
        "churn_rate": churn_rate,
        "high_risk_count": high_risk_count,
        "high_risk_pct": high_risk_pct,
        "medium_risk_count": medium_risk_count,
        "low_risk_count": low_risk_count,
        "avg_monthly_charges": avg_monthly_charges,
        "avg_tenure": avg_tenure,
        "avg_satisfaction": avg_satisfaction,
        "total_monthly_revenue": total_monthly_revenue,
        "revenue_at_risk_annual": revenue_at_risk_annual,
        "revenue_at_risk_monthly": revenue_at_risk_monthly,
    }


def categorize_risk(prob: float, low_threshold: float = 0.30, high_threshold: float = 0.70) -> str:
    """Categorize churn probability into business risk tier."""
    if pd.isna(prob):
        return "Unknown"
    if prob <= low_threshold:
        return "Low"
    elif prob <= high_threshold:
        return "Medium"
    else:
        return "High"


def calculate_customer_value_proxy(df: pd.DataFrame) -> pd.Series:
    """Compute transparent Customer Value proxy.
    
    Formula:
      Estimated Customer Value = Total_Charges + (Monthly_Charges * 12)
    This combines actual realized lifetime revenue with 1-year forward run-rate.
    If Total_Charges is missing, defaults to Monthly_Charges * (Tenure + 12).
    """
    if "Total_Charges" in df.columns and "Monthly_Charges" in df.columns:
        val = df["Total_Charges"].fillna(0) + (df["Monthly_Charges"].fillna(0) * 12)
    elif "Monthly_Charges" in df.columns and "Tenure" in df.columns:
        val = df["Monthly_Charges"].fillna(0) * (df["Tenure"].fillna(1) + 12)
    elif "Monthly_Charges" in df.columns:
        val = df["Monthly_Charges"].fillna(0) * 24
    else:
        val = pd.Series(0.0, index=df.index)
    return val.round(2)


def assign_priority(
    prob: float,
    value: float,
    value_quantiles: Tuple[float, float] = (50000.0, 120000.0),
    high_risk_thresh: float = 0.70,
    med_risk_thresh: float = 0.30
) -> Tuple[str, str]:
    """Assign Retention Priority (P1 to P4) based on Churn Risk and Customer Value.
    
    Returns:
        (priority_code, reason)
    """
    v_med, v_high = value_quantiles
    is_high_val = value >= v_high
    is_med_val = (value >= v_med) and not is_high_val
    is_low_val = value < v_med

    is_high_risk = prob >= high_risk_thresh
    is_med_risk = (prob >= med_risk_thresh) and not is_high_risk
    is_low_risk = prob < med_risk_thresh

    if is_high_risk and is_high_val:
        return "P1 - Critical", "High churn probability combined with top-tier economic customer value."
    elif is_high_risk and is_med_val:
        return "P2 - High", "High churn probability with moderate customer value."
    elif is_med_risk and is_high_val:
        return "P2 - High", "Moderate churn risk on a very high-value account deserving early intervention."
    elif is_high_risk and is_low_val:
        return "P3 - Medium", "High churn probability but lower overall economic footprint."
    elif is_med_risk and is_med_val:
        return "P3 - Medium", "Moderate churn risk with moderate account value."
    else:
        return "P4 - Low", "Low churn probability or lower portfolio exposure."
