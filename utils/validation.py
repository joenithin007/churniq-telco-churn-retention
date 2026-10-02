"""Data validation and quality inspection utilities."""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


# Standard expected schema and canonical names
EXPECTED_COLUMNS = {
    "Customer_ID": "string",
    "Age": "numeric",
    "Gender": "category",
    "Location": "category",
    "Occupation": "category",
    "Income": "numeric",
    "Tenure": "numeric",
    "Service_Type": "category",
    "Subscription_Plan": "category",
    "Contract_Type": "category",
    "Number_of_Products": "numeric",
    "Add_On_Services": "category",
    "Login_Frequency": "numeric",
    "Call_Duration": "numeric",
    "Data_Usage": "numeric",
    "Number_of_Transactions": "numeric",
    "App_Usage": "numeric",
    "Website_Visits": "numeric",
    "Monthly_Charges": "numeric",
    "Total_Charges": "numeric",
    "Payment_Method": "category",
    "Payment_Failures": "numeric",
    "Outstanding_Balance": "numeric",
    "Complaints": "numeric",
    "Support_Calls": "numeric",
    "Tickets_Raised": "numeric",
    "Resolution_Time": "numeric",
    "Satisfaction_Score": "numeric",
    "Churn": "numeric",
}

# Synonyms for flexible column mapping
COLUMN_SYNONYMS = {
    "customer_id": "Customer_ID",
    "cust_id": "Customer_ID",
    "id": "Customer_ID",
    "customerid": "Customer_ID",
    "age": "Age",
    "gender": "Gender",
    "sex": "Gender",
    "location": "Location",
    "city": "Location",
    "state": "Location",
    "occupation": "Occupation",
    "job": "Occupation",
    "income": "Income",
    "annual_income": "Income",
    "tenure": "Tenure",
    "tenure_months": "Tenure",
    "service_type": "Service_Type",
    "service": "Service_Type",
    "subscription_plan": "Subscription_Plan",
    "plan": "Subscription_Plan",
    "contract_type": "Contract_Type",
    "contract": "Contract_Type",
    "number_of_products": "Number_of_Products",
    "num_products": "Number_of_Products",
    "products_count": "Number_of_Products",
    "add_on_services": "Add_On_Services",
    "addons": "Add_On_Services",
    "login_frequency": "Login_Frequency",
    "logins": "Login_Frequency",
    "call_duration": "Call_Duration",
    "data_usage": "Data_Usage",
    "number_of_transactions": "Number_of_Transactions",
    "transactions": "Number_of_Transactions",
    "app_usage": "App_Usage",
    "website_visits": "Website_Visits",
    "web_visits": "Website_Visits",
    "monthly_charges": "Monthly_Charges",
    "monthlycharges": "Monthly_Charges",
    "monthly_charge": "Monthly_Charges",
    "total_charges": "Total_Charges",
    "totalcharges": "Total_Charges",
    "total_charge": "Total_Charges",
    "payment_method": "Payment_Method",
    "paymentmethod": "Payment_Method",
    "payment_failures": "Payment_Failures",
    "outstanding_balance": "Outstanding_Balance",
    "balance": "Outstanding_Balance",
    "complaints": "Complaints",
    "support_calls": "Support_Calls",
    "tickets_raised": "Tickets_Raised",
    "tickets": "Tickets_Raised",
    "resolution_time": "Resolution_Time",
    "satisfaction_score": "Satisfaction_Score",
    "satisfaction": "Satisfaction_Score",
    "churn": "Churn",
    "churned": "Churn",
    "target": "Churn",
}


def map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Flexibly standardize dataframe column names using synonym dictionary."""
    renamed = {}
    new_cols = []
    for c in df.columns:
        norm = str(c).strip().lower().replace("-", "_").replace(" ", "_")
        if norm in COLUMN_SYNONYMS:
            canonical = COLUMN_SYNONYMS[norm]
            renamed[c] = canonical
            new_cols.append(canonical)
        else:
            new_cols.append(c)
    
    df_copy = df.copy()
    df_copy.columns = new_cols
    return df_copy, renamed


def inspect_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive data quality inspection.
    
    Returns structured metrics on missing values, duplicates, out-of-bounds,
    and type validations without silently modifying data.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # 1. Duplicates
    if "Customer_ID" in df.columns:
        duplicate_ids = int(df["Customer_ID"].duplicated().sum())
    else:
        duplicate_ids = 0
    duplicate_rows = int(df.duplicated().sum())

    # 2. Missing values
    missing_by_col = df.isnull().sum().to_dict()
    total_missing_cells = int(df.isnull().sum().sum())
    cols_with_missing = {k: v for k, v in missing_by_col.items() if v > 0}

    # 3. Target verification
    has_target = "Churn" in df.columns
    target_missing = int(df["Churn"].isnull().sum()) if has_target else None
    target_distribution = None
    if has_target and df["Churn"].dropna().isin([0, 1]).all():
        counts = df["Churn"].value_counts().to_dict()
        target_distribution = {str(k): int(v) for k, v in counts.items()}

    # 4. Value boundary anomalies
    anomalies: List[Dict[str, Any]] = []

    if "Age" in df.columns:
        invalid_age = ((df["Age"] < 0) | (df["Age"] > 120)).sum()
        if invalid_age > 0:
            anomalies.append({
                "column": "Age",
                "issue": f"{invalid_age} records with Age < 0 or > 120",
                "count": int(invalid_age),
                "severity": "Warning"
            })

    if "Monthly_Charges" in df.columns:
        neg_charges = (df["Monthly_Charges"] < 0).sum()
        if neg_charges > 0:
            anomalies.append({
                "column": "Monthly_Charges",
                "issue": f"{neg_charges} records with negative Monthly_Charges",
                "count": int(neg_charges),
                "severity": "Error"
            })

    if "Tenure" in df.columns:
        neg_tenure = (df["Tenure"] < 0).sum()
        if neg_tenure > 0:
            anomalies.append({
                "column": "Tenure",
                "issue": f"{neg_tenure} records with negative Tenure",
                "count": int(neg_tenure),
                "severity": "Error"
            })

    if "Satisfaction_Score" in df.columns:
        invalid_sat = ((df["Satisfaction_Score"] < 1) | (df["Satisfaction_Score"] > 5)).sum()
        if invalid_sat > 0:
            anomalies.append({
                "column": "Satisfaction_Score",
                "issue": f"{invalid_sat} records outside valid range (1 to 5)",
                "count": int(invalid_sat),
                "severity": "Warning"
            })

    for col in ["Call_Duration", "Data_Usage", "Number_of_Transactions", "Total_Charges", "Payment_Failures"]:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                anomalies.append({
                    "column": col,
                    "issue": f"{neg_count} records with negative {col}",
                    "count": int(neg_count),
                    "severity": "Error"
                })

    # Overall Status Summary
    issues_count = duplicate_ids + len(anomalies) + len(cols_with_missing)
    if issues_count == 0:
        overall_status = "Excellent"
    elif duplicate_ids > 0 or any(a["severity"] == "Error" for a in anomalies):
        overall_status = "Needs Attention"
    else:
        overall_status = "Good (Minor Warnings)"

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "total_missing_cells": total_missing_cells,
        "cols_with_missing": cols_with_missing,
        "has_target": has_target,
        "target_missing": target_missing,
        "target_distribution": target_distribution,
        "anomalies": anomalies,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "overall_status": overall_status,
    }
