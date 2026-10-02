"""Data cleaning and preprocessing module."""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np


class DataCleaner:
    """Automated and transparent data cleaning pipeline."""

    def __init__(self):
        self.cleaning_log: List[str] = []
        self.imputed_values: Dict[str, Any] = {}
        self.records_modified: int = 0
        self.duplicates_removed: int = 0

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean dataset adhering to analytical data quality rules.
        
        Rules:
          - Remove duplicate Customer_ID / duplicate rows
          - Numerical missing values -> Median imputation
          - Categorical missing values -> Mode imputation
          - Out-of-bound / Invalid values -> Transparent correction with audit log
        """
        self.cleaning_log = []
        self.imputed_values = {}
        df_clean = df.copy()

        initial_rows = len(df_clean)

        # 1. Handle Duplicates
        if "Customer_ID" in df_clean.columns:
            dup_cust = df_clean.duplicated(subset=["Customer_ID"], keep="first").sum()
            if dup_cust > 0:
                df_clean = df_clean.drop_duplicates(subset=["Customer_ID"], keep="first")
                self.duplicates_removed = int(dup_cust)
                self.cleaning_log.append(f"Removed {dup_cust} duplicate Customer_ID entries.")
        else:
            dup_rows = df_clean.duplicated().sum()
            if dup_rows > 0:
                df_clean = df_clean.drop_duplicates()
                self.duplicates_removed = int(dup_rows)
                self.cleaning_log.append(f"Removed {dup_rows} identical duplicate rows.")

        # 2. Fix Invalid Out-of-bounds Values
        if "Age" in df_clean.columns:
            invalid_age = ((df_clean["Age"] < 18) | (df_clean["Age"] > 100))
            if invalid_age.sum() > 0:
                median_age = df_clean.loc[~invalid_age, "Age"].median()
                if pd.isna(median_age):
                    median_age = 35.0
                df_clean.loc[invalid_age, "Age"] = median_age
                self.cleaning_log.append(f"Adjusted {invalid_age.sum()} out-of-range Age records to median ({int(median_age)}).")

        if "Monthly_Charges" in df_clean.columns:
            neg_charges = df_clean["Monthly_Charges"] < 0
            if neg_charges.sum() > 0:
                median_charge = df_clean.loc[~neg_charges, "Monthly_Charges"].median()
                df_clean.loc[neg_charges, "Monthly_Charges"] = median_charge
                self.cleaning_log.append(f"Corrected {neg_charges.sum()} negative Monthly_Charges records to median.")

        if "Tenure" in df_clean.columns:
            neg_tenure = df_clean["Tenure"] < 0
            if neg_tenure.sum() > 0:
                df_clean.loc[neg_tenure, "Tenure"] = 1
                self.cleaning_log.append(f"Corrected {neg_tenure.sum()} negative Tenure values to 1.")

        if "Satisfaction_Score" in df_clean.columns:
            invalid_sat = ((df_clean["Satisfaction_Score"] < 1) | (df_clean["Satisfaction_Score"] > 5))
            if invalid_sat.sum() > 0:
                df_clean.loc[invalid_sat, "Satisfaction_Score"] = df_clean.loc[invalid_sat, "Satisfaction_Score"].clip(1, 5)
                self.cleaning_log.append(f"Clipped {invalid_sat.sum()} Satisfaction_Score values outside the [1, 5] range.")

        # Non-negative checks on counts & duration
        for count_col in ["Call_Duration", "Data_Usage", "Number_of_Transactions", "Total_Charges", "Payment_Failures", "Complaints", "Support_Calls", "Tickets_Raised"]:
            if count_col in df_clean.columns:
                neg_mask = df_clean[count_col] < 0
                if neg_mask.sum() > 0:
                    df_clean.loc[neg_mask, count_col] = 0
                    self.cleaning_log.append(f"Set {neg_mask.sum()} negative values in '{count_col}' to 0.")

        # 3. Missing Value Imputation
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if "Churn" in numeric_cols:
            numeric_cols.remove("Churn")

        for col in numeric_cols:
            n_missing = df_clean[col].isnull().sum()
            if n_missing > 0:
                med_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(med_val)
                self.imputed_values[col] = {"type": "median", "value": float(med_val), "count": int(n_missing)}
                self.cleaning_log.append(f"Imputed {n_missing} missing values in numeric column '{col}' with median ({med_val:.2f}).")

        categorical_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
        if "Customer_ID" in categorical_cols:
            categorical_cols.remove("Customer_ID")

        for col in categorical_cols:
            n_missing = df_clean[col].isnull().sum()
            if n_missing > 0:
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                df_clean[col] = df_clean[col].fillna(mode_val)
                self.imputed_values[col] = {"type": "mode", "value": str(mode_val), "count": int(n_missing)}
                self.cleaning_log.append(f"Imputed {n_missing} missing values in categorical column '{col}' with mode ('{mode_val}').")

        # 4. Target variable handling: drop rows with missing target if any
        if "Churn" in df_clean.columns:
            target_missing = df_clean["Churn"].isnull().sum()
            if target_missing > 0:
                df_clean = df_clean.dropna(subset=["Churn"])
                self.cleaning_log.append(f"Dropped {target_missing} records with missing target 'Churn'.")
            # Ensure target is integer 0 or 1
            df_clean["Churn"] = df_clean["Churn"].astype(int)

        summary = {
            "initial_rows": initial_rows,
            "cleaned_rows": len(df_clean),
            "duplicates_removed": self.duplicates_removed,
            "imputed_columns": self.imputed_values,
            "cleaning_log": self.cleaning_log,
        }

        return df_clean, summary
