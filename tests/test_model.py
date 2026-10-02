"""Unit tests for ML model training, evaluation, and probability inference."""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic_data
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.train_model import ModelTrainer
from src.predict import predict_churn
from src.evaluate_model import evaluate_predictions


@pytest.fixture(scope="module")
def prepared_sample_data():
    """Generate small dataset and run basic cleaning and featurization for tests."""
    raw = generate_synthetic_data(num_records=500, random_seed=42)
    cleaner = DataCleaner()
    clean_df, _ = cleaner.clean(raw)
    engineer = FeatureEngineer()
    feat_df = engineer.transform(clean_df)
    return feat_df


def test_model_training_and_artifacts(prepared_sample_data):
    """Test model training pipeline, feature transformer, and metric calculation."""
    trainer = ModelTrainer(test_size=0.20, random_state=42)
    results = trainer.train_and_evaluate_all(prepared_sample_data)

    assert results["best_model"] is not None
    assert results["preprocessor"] is not None
    assert len(results["feature_names"]) > 0

    metrics = results["chosen_metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2


def test_prediction_inference(prepared_sample_data):
    """Verify predict_churn attaches bounded probabilities and risk labels."""
    trainer = ModelTrainer(test_size=0.20, random_state=42)
    results = trainer.train_and_evaluate_all(prepared_sample_data)

    predicted_df = predict_churn(
        prepared_sample_data,
        results["best_model"],
        results["preprocessor"]
    )

    assert "Churn_Probability" in predicted_df.columns
    assert "Risk_Level" in predicted_df.columns
    assert "Priority" in predicted_df.columns

    # Probabilities strictly in [0, 1]
    assert (predicted_df["Churn_Probability"] >= 0.0).all()
    assert (predicted_df["Churn_Probability"] <= 1.0).all()

    # Risk tiers
    valid_risks = {"Low", "Medium", "High"}
    assert set(predicted_df["Risk_Level"].unique()).issubset(valid_risks)
