"""Customer risk tier segmentation and threshold management."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class RiskSegmenter:
    """Configurable risk segmentation engine."""

    def __init__(self, low_threshold: float = 0.30, high_threshold: float = 0.70):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def segment(self, probabilities: np.ndarray) -> np.ndarray:
        """Assign risk labels ('Low', 'Medium', 'High') based on thresholds."""
        conditions = [
            probabilities <= self.low_threshold,
            (probabilities > self.low_threshold) & (probabilities <= self.high_threshold),
            probabilities > self.high_threshold,
        ]
        choices = ["Low", "Medium", "High"]
        return np.select(conditions, choices, default="Medium")

    def summarize_segments(self, df: pd.DataFrame, prob_col: str = "Churn_Probability") -> pd.DataFrame:
        """Generate business summary statistics for each risk segment."""
        if "Risk_Level" not in df.columns or df.empty:
            return pd.DataFrame()

        tier_order = ["High", "Medium", "Low"]
        agg_list = []

        total_customers = len(df)
        total_monthly_rev = df["Monthly_Charges"].sum() if "Monthly_Charges" in df.columns else 0.0

        for tier in tier_order:
            sub = df[df["Risk_Level"] == tier]
            count = len(sub)
            pct = (count / total_customers * 100.0) if total_customers > 0 else 0.0
            
            avg_prob = sub[prob_col].mean() if prob_col in sub.columns and count > 0 else 0.0
            avg_charges = sub["Monthly_Charges"].mean() if "Monthly_Charges" in sub.columns and count > 0 else 0.0
            tot_charges = sub["Monthly_Charges"].sum() if "Monthly_Charges" in sub.columns and count > 0 else 0.0
            ann_runrate = tot_charges * 12.0

            agg_list.append({
                "Risk Tier": f"{tier} Risk",
                "Customers": count,
                "Customer Share": f"{pct:.1f}%",
                "Avg Churn Probability": f"{avg_prob * 100:.1f}%",
                "Avg Monthly Spend": f"₹{avg_charges:,.0f}",
                "Monthly Revenue at Exposure": f"₹{tot_charges:,.0f}",
                "Annual Revenue at Exposure": f"₹{ann_runrate:,.0f}",
            })

        return pd.DataFrame(agg_list)
