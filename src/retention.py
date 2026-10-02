"""Retention intelligence, customer prioritization matrix, and business review rules."""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np


class RetentionEngine:
    """Evaluates customer risk factors and generates actionable, non-causal review suggestions."""

    def __init__(
        self,
        value_threshold_high: float = 90000.0,
        value_threshold_med: float = 35000.0,
        risk_threshold_high: float = 0.70,
        risk_threshold_med: float = 0.30
    ):
        self.val_high = value_threshold_high
        self.val_med = value_threshold_med
        self.risk_high = risk_threshold_high
        self.risk_med = risk_threshold_med

    def generate_review_areas_and_factors(self, row: pd.Series) -> Tuple[str, str]:
        """Analyze individual customer attributes to identify risk factors and suggested review areas.
        
        Strict analytical language: identifies associations, never claims certainty or causation.
        """
        factors = []
        review_areas = []

        prob = row.get("Churn_Probability", 0.0)
        tenure = row.get("Tenure", 24)
        contract = str(row.get("Contract_Type", ""))
        monthly_charges = row.get("Monthly_Charges", 1000)
        complaints = row.get("Complaints", 0)
        satisfaction = row.get("Satisfaction_Score", 4)
        pay_failures = row.get("Payment_Failures", 0)
        balance = row.get("Outstanding_Balance", 0)
        engagement = row.get("Engagement_Score", 50)

        # Contributing Risk Factors
        if contract == "Month-to-Month":
            factors.append("Month-to-month contract")
            review_areas.append("Contract / Annual Plan Incentive Review")

        if tenure <= 6:
            factors.append("Early lifecycle tenure (≤6 mo)")
            review_areas.append("Onboarding & Early Lifecycle Experience")
        elif tenure <= 12:
            factors.append("Tenure under 1 year")

        if monthly_charges >= 1800:
            factors.append(f"High monthly charges (₹{monthly_charges:,.0f})")
            review_areas.append("Pricing & Tier Suitability Review")

        if complaints >= 2 or satisfaction <= 2:
            factors.append(f"Elevated complaints ({complaints}) / Low satisfaction ({satisfaction}/5)")
            review_areas.append("Customer Support & Service Quality Review")

        if pay_failures >= 1 or balance > 0:
            factors.append(f"Payment issues ({pay_failures} failures / ₹{balance:,.0f} balance)")
            review_areas.append("Billing & Payment Method Assistance")

        if engagement < 35:
            factors.append(f"Subdued engagement score ({engagement:.0f}/100)")
            review_areas.append("Product Adoption & Engagement Campaign")

        if not factors:
            factors.append("General lifecycle variance")
        if not review_areas:
            review_areas.append("Standard Relationship Monitoring")

        # Deduplicate while preserving order
        unique_factors = list(dict.fromkeys(factors))
        unique_reviews = list(dict.fromkeys(review_areas))

        return " • ".join(unique_factors[:4]), " • ".join(unique_reviews[:3])

    def enrich_retention_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add retention priority, risk factors, and suggested review areas across dataframe."""
        df_ret = df.copy()

        priorities = []
        factors_list = []
        reviews_list = []

        for _, row in df_ret.iterrows():
            prob = row.get("Churn_Probability", 0.0)
            val = row.get("Customer_Value", 0.0)

            # Assign Priority
            if prob >= self.risk_high and val >= self.val_high:
                prio = "P1 - Critical"
            elif prob >= self.risk_high and val >= self.val_med:
                prio = "P2 - High"
            elif prob >= self.risk_med and val >= self.val_high:
                prio = "P2 - High"
            elif prob >= self.risk_high and val < self.val_med:
                prio = "P3 - Medium"
            elif prob >= self.risk_med and val >= self.val_med:
                prio = "P3 - Medium"
            else:
                prio = "P4 - Low"

            priorities.append(prio)
            factors, reviews = self.generate_review_areas_and_factors(row)
            factors_list.append(factors)
            reviews_list.append(reviews)

        df_ret["Priority"] = priorities
        df_ret["Main_Risk_Factors"] = factors_list
        df_ret["Suggested_Review_Areas"] = reviews_list

        return df_ret

    def get_quadrant_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calculate counts for the 4 key Value-Risk review quadrants."""
        if "Churn_Probability" not in df.columns or "Customer_Value" not in df.columns:
            return {
                "High Risk + High Value": 0,
                "High Risk + Medium Value": 0,
                "High Risk + Low Value": 0,
                "Medium Risk + High Value": 0,
            }

        high_risk = df["Churn_Probability"] >= self.risk_high
        med_risk = (df["Churn_Probability"] >= self.risk_med) & (df["Churn_Probability"] < self.risk_high)

        high_val = df["Customer_Value"] >= self.val_high
        med_val = (df["Customer_Value"] >= self.val_med) & (df["Customer_Value"] < self.val_high)
        low_val = df["Customer_Value"] < self.val_med

        return {
            "High Risk + High Value": int((high_risk & high_val).sum()),
            "High Risk + Medium Value": int((high_risk & med_val).sum()),
            "High Risk + Low Value": int((high_risk & low_val).sum()),
            "Medium Risk + High Value": int((med_risk & high_val).sum()),
        }
