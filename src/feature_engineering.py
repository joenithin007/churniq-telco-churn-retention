"""Feature engineering module for Customer Churn Prediction & Retention Intelligence."""

from typing import Tuple, List
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Transforms cleaned customer dataset into analytically rich feature representations."""

    def __init__(self):
        self.engineered_feature_names: List[str] = [
            "Average_Monthly_Spend",
            "Support_Intensity",
            "Complaint_Rate",
            "Payment_Failure_Rate",
            "Engagement_Score",
            "Customer_Value",
            "Value_Tier",
            "Tenure_Group",
            "Age_Group",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features with robust safeguards against division by zero."""
        df_feat = df.copy()

        # Safe tenure representation for ratios
        safe_tenure = df_feat["Tenure"].clip(lower=1.0) if "Tenure" in df_feat.columns else pd.Series(1.0, index=df_feat.index)

        # 1. Average Monthly Spend
        if "Total_Charges" in df_feat.columns:
            df_feat["Average_Monthly_Spend"] = (df_feat["Total_Charges"] / safe_tenure).round(2)
        elif "Monthly_Charges" in df_feat.columns:
            df_feat["Average_Monthly_Spend"] = df_feat["Monthly_Charges"].round(2)
        else:
            df_feat["Average_Monthly_Spend"] = 0.0

        # 2. Support Intensity (Support Calls per month of tenure)
        if "Support_Calls" in df_feat.columns:
            df_feat["Support_Intensity"] = (df_feat["Support_Calls"] / safe_tenure).round(3)
        else:
            df_feat["Support_Intensity"] = 0.0

        # 3. Complaint Rate (Complaints per month of tenure)
        if "Complaints" in df_feat.columns:
            df_feat["Complaint_Rate"] = (df_feat["Complaints"] / safe_tenure).round(3)
        else:
            df_feat["Complaint_Rate"] = 0.0

        # 4. Payment Failure Rate
        if "Payment_Failures" in df_feat.columns and "Number_of_Transactions" in df_feat.columns:
            total_attempts = (df_feat["Number_of_Transactions"] + df_feat["Payment_Failures"]).replace(0, 1)
            df_feat["Payment_Failure_Rate"] = (df_feat["Payment_Failures"] / total_attempts).round(3)
        elif "Payment_Failures" in df_feat.columns:
            df_feat["Payment_Failure_Rate"] = (df_feat["Payment_Failures"] / safe_tenure).round(3)
        else:
            df_feat["Payment_Failure_Rate"] = 0.0

        # 5. Composite Normalized Engagement Score (0 to 100)
        engagement_parts = []
        if "Login_Frequency" in df_feat.columns:
            # Login frequency max ~ 60
            norm_login = (df_feat["Login_Frequency"] / 60.0).clip(0, 1)
            engagement_parts.append(norm_login * 0.30)
        if "App_Usage" in df_feat.columns:
            # App usage max ~ 80 hrs
            norm_app = (df_feat["App_Usage"] / 80.0).clip(0, 1)
            engagement_parts.append(norm_app * 0.30)
        if "Data_Usage" in df_feat.columns:
            # Data usage max ~ 600 GB
            norm_data = (df_feat["Data_Usage"] / 600.0).clip(0, 1)
            engagement_parts.append(norm_data * 0.25)
        if "Website_Visits" in df_feat.columns:
            # Web visits max ~ 50
            norm_web = (df_feat["Website_Visits"] / 50.0).clip(0, 1)
            engagement_parts.append(norm_web * 0.15)

        if engagement_parts:
            sum_norm = sum(engagement_parts)
            df_feat["Engagement_Score"] = (sum_norm * 100.0).round(1).clip(0.0, 100.0)
        else:
            df_feat["Engagement_Score"] = 50.0

        # 6. Transparent Estimated Customer Value Proxy
        if "Total_Charges" in df_feat.columns and "Monthly_Charges" in df_feat.columns:
            df_feat["Customer_Value"] = (df_feat["Total_Charges"] + (df_feat["Monthly_Charges"] * 12)).round(2)
        elif "Monthly_Charges" in df_feat.columns:
            df_feat["Customer_Value"] = (df_feat["Monthly_Charges"] * (safe_tenure + 12)).round(2)
        else:
            df_feat["Customer_Value"] = 0.0

        # 7. Customer Value Tier (Low, Medium, High)
        # Using clear absolute/quantile thresholds for business interpretability
        v_low = 35000.0
        v_high = 90000.0
        df_feat["Value_Tier"] = pd.cut(
            df_feat["Customer_Value"],
            bins=[-np.inf, v_low, v_high, np.inf],
            labels=["Low Value", "Medium Value", "High Value"]
        ).astype(str)

        # 8. Demographic & Tenure Bins for Dashboard Filtering & Stratification
        if "Tenure" in df_feat.columns:
            df_feat["Tenure_Group"] = pd.cut(
                df_feat["Tenure"],
                bins=[-np.inf, 12, 24, 48, np.inf],
                labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]
            ).astype(str)

        if "Age" in df_feat.columns:
            df_feat["Age_Group"] = pd.cut(
                df_feat["Age"],
                bins=[-np.inf, 25, 35, 50, 65, np.inf],
                labels=["< 25", "25-35", "36-50", "51-65", "65+"]
            ).astype(str)

        return df_feat
