"""Unit tests for data generation, loading, and cleaning."""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import generate_synthetic_data
from src.data_cleaning import DataCleaner
from utils.validation import inspect_data_quality, map_columns


def test_synthetic_data_generation():
    """Verify synthetic dataset generation satisfies shape, schema, and target properties."""
    df = generate_synthetic_data(num_records=1000, random_seed=42)
    assert len(df) == 1000
    assert "Customer_ID" in df.columns
    assert "Churn" in df.columns
    assert df["Customer_ID"].nunique() == 1000  # Unique IDs
    assert set(df["Churn"].unique()).issubset({0, 1})
    
    # Check demographics and financials
    assert (df["Age"] >= 18).all() and (df["Age"] <= 100).all()
    assert (df["Monthly_Charges"] >= 0).all()
    assert (df["Tenure"] >= 1).all()


def test_data_cleaning_pipeline():
    """Verify missing value imputation and duplicate handling."""
    # Create sample with intentional anomalies
    raw_data = pd.DataFrame({
        "Customer_ID": ["C1", "C2", "C2", "C3"],
        "Age": [25, -5, 40, np.nan],
        "Monthly_Charges": [1000.0, -200.0, 1500.0, np.nan],
        "Tenure": [12, -2, 24, 6],
        "Satisfaction_Score": [4, 9, 2, 1],
        "Contract_Type": ["Month-to-Month", None, "Two-Year", "One-Year"],
        "Churn": [0, 1, 1, 0],
    })

    cleaner = DataCleaner()
    df_clean, summary = cleaner.clean(raw_data)

    # Duplicates removed
    assert df_clean["Customer_ID"].duplicated().sum() == 0
    assert summary["duplicates_removed"] >= 1

    # Missing values imputed
    assert df_clean["Age"].isnull().sum() == 0
    assert df_clean["Monthly_Charges"].isnull().sum() == 0
    assert df_clean["Contract_Type"].isnull().sum() == 0

    # Bounds corrected
    assert (df_clean["Age"] >= 18).all()
    assert (df_clean["Monthly_Charges"] >= 0).all()
    assert (df_clean["Tenure"] >= 1).all()
    assert (df_clean["Satisfaction_Score"] <= 5).all()


def test_flexible_column_mapping():
    """Verify synonym dictionary normalizes messy column names."""
    df_messy = pd.DataFrame({
        "customerid": ["C1"],
        "monthlycharges": [999.0],
        "tenure_months": [12],
        "target": [0]
    })
    mapped_df, renames = map_columns(df_messy)
    assert "Customer_ID" in mapped_df.columns
    assert "Monthly_Charges" in mapped_df.columns
    assert "Tenure" in mapped_df.columns
    assert "Churn" in mapped_df.columns
