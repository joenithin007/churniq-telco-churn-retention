"""Inference pipeline for churn probability prediction and risk enrichment."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.risk_segmentation import RiskSegmenter
from src.retention import RetentionEngine


def predict_churn(
    df: pd.DataFrame,
    model: object,
    preprocessor: ColumnTransformer,
    segmenter: Optional[RiskSegmenter] = None,
    retention_engine: Optional[RetentionEngine] = None,
) -> pd.DataFrame:
    """Generate churn probabilities, risk levels, and retention intelligence for a dataset."""
    if df.empty:
        return df

    if segmenter is None:
        segmenter = RiskSegmenter(low_threshold=0.30, high_threshold=0.70)
    if retention_engine is None:
        retention_engine = RetentionEngine()

    df_out = df.copy()

    # Identify features expected by preprocessor
    expected_cols = []
    for name, trans, cols in preprocessor.transformers_:
        if name != "remainder" and cols:
            expected_cols.extend(cols)

    # Check for missing expected features and supply default if needed
    missing = [c for c in expected_cols if c not in df_out.columns]
    if missing:
        raise ValueError(f"Input data is missing expected feature columns: {missing}")

    X = df_out[expected_cols].copy()

    # Preprocess
    X_trans = preprocessor.transform(X)

    # Predict Probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_trans)[:, 1]
    elif hasattr(model, "decision_function"):
        # For classifiers without predict_proba, calibrate via sigmoid
        scores = model.decision_function(X_trans)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
    else:
        raise AttributeError(f"Model {type(model).__name__} does not support probability predictions.")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    df_out["Churn_Probability"] = np.round(probabilities, 4)

    # Risk Segmentation
    df_out["Risk_Level"] = segmenter.segment(df_out["Churn_Probability"].values)

    # Retention Enrichment
    df_out = retention_engine.enrich_retention_intelligence(df_out)

    return df_out
