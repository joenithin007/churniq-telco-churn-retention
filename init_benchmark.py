"""Initialize sample benchmark dataset (12,000 records) and pre-train models."""

import os
from pathlib import Path
from src.data_loader import load_dataset, SAMPLE_DATA_PATH
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.train_model import ModelTrainer


def main():
    print(f"Generating or loading sample dataset at: {SAMPLE_DATA_PATH}")
    df_raw = load_dataset()
    print(f"Dataset ready with {len(df_raw)} records and {len(df_raw.columns)} columns.")

    print("Cleaning dataset...")
    cleaner = DataCleaner()
    df_clean, summary = cleaner.clean(df_raw)
    print(f"Cleaned dataset: {len(df_clean)} records. Duplicates removed: {summary['duplicates_removed']}.")

    print("Engineering features...")
    engineer = FeatureEngineer()
    df_feat = engineer.transform(df_clean)
    print(f"Features engineered. Total columns: {len(df_feat.columns)}.")

    print("Training candidate models and selecting optimal model...")
    trainer = ModelTrainer(test_size=0.20, random_state=42)
    results = trainer.train_and_evaluate_all(df_feat)
    print(f"Optimal model selected: {results['best_model_name']}")
    for m in results["all_metrics"]:
        print(f"  - {m['model_name']}: Accuracy={m['accuracy']:.3f}, Recall={m['recall']:.3f}, Precision={m['precision']:.3f}, ROC-AUC={m['roc_auc']:.3f}")

    print("Benchmark artifacts successfully generated and persisted!")


if __name__ == "__main__":
    main()
