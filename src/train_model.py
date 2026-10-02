"""Model training and persistence module for Customer Churn Prediction."""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.evaluate_model import evaluate_predictions


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelTrainer:
    """Trains, compares, and persists churn prediction models."""

    def __init__(self, test_size: float = 0.20, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor: Optional[ColumnTransformer] = None
        self.best_model: Optional[Any] = None
        self.best_model_name: str = ""
        self.feature_names: List[str] = []
        self.all_model_metrics: List[Dict[str, Any]] = []

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """Split features and target, identifying numerical and categorical feature sets."""
        if "Churn" not in df.columns:
            raise ValueError("Target variable 'Churn' not found in dataset.")

        # Exclude ID and target-derived leakages
        exclude_cols = [
            "Customer_ID", "Churn", "Risk_Level", "Churn_Probability",
            "Priority", "Main_Risk_Factors", "Suggested_Review_Areas",
            "Value_Tier", "Tenure_Group", "Age_Group"
        ]

        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].copy()
        y = df["Churn"].astype(int).copy()

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        return X, y, num_cols, cat_cols

    def train_and_evaluate_all(
        self,
        df: pd.DataFrame,
        selected_model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train candidate models, evaluate on holdout test set, select best, and persist."""
        X, y, num_cols, cat_cols = self.prepare_data(df)

        # 1. Stratified Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        # 2. Strict Preprocessing Pipeline fit only on train
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ]
        )

        X_train_trans = self.preprocessor.fit_transform(X_train)
        X_test_trans = self.preprocessor.transform(X_test)

        # Extract encoded feature names
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        self.feature_names = num_cols + encoded_cat_names

        # 3. Candidate Models
        # Compute class balance ratio
        pos_ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())

        candidates = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=120,
                max_depth=8,
                min_samples_split=10,
                class_weight="balanced",
                random_state=self.random_state
            ),
        }

        if HAS_XGBOOST:
            candidates["XGBoost"] = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                scale_pos_weight=pos_ratio,
                random_state=self.random_state,
                eval_metric="logloss",
            )
        else:
            candidates["Gradient Boosting"] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=self.random_state
            )

        self.all_model_metrics = []
        trained_models = {}

        for name, clf in candidates.items():
            clf.fit(X_train_trans, y_train)
            trained_models[name] = clf

            y_pred = clf.predict(X_test_trans)
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test_trans)[:, 1]
            else:
                y_prob = clf.decision_function(X_test_trans)

            metrics = evaluate_predictions(y_test.values, y_pred, y_prob, model_name=name)
            self.all_model_metrics.append(metrics)

        # 4. Selection Criteria: If specified, use that; otherwise pick highest ROC-AUC
        if selected_model_type and selected_model_type in trained_models:
            chosen_name = selected_model_type
        else:
            # Pick highest ROC-AUC
            sorted_models = sorted(self.all_model_metrics, key=lambda m: m["roc_auc"], reverse=True)
            chosen_name = sorted_models[0]["model_name"]

        self.best_model_name = chosen_name
        self.best_model = trained_models[chosen_name]
        chosen_metrics = next(m for m in self.all_model_metrics if m["model_name"] == chosen_name)

        # 5. Persist artifacts
        self.save_artifacts(chosen_metrics)

        return {
            "best_model_name": self.best_model_name,
            "best_model": self.best_model,
            "preprocessor": self.preprocessor,
            "feature_names": self.feature_names,
            "chosen_metrics": chosen_metrics,
            "all_metrics": self.all_model_metrics,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "test_records": len(y_test),
        }

    def save_artifacts(self, chosen_metrics: Dict[str, Any]) -> None:
        """Persist model, preprocessor, and metadata."""
        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(self.best_model, MODEL_DIR / "churn_model.pkl")
        joblib.dump(self.preprocessor, MODEL_DIR / "preprocessor.pkl")

        metadata = {
            "model_version": "1.0.0",
            "model_type": self.best_model_name,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "metrics": {
                "accuracy": chosen_metrics["accuracy"],
                "precision": chosen_metrics["precision"],
                "recall": chosen_metrics["recall"],
                "f1": chosen_metrics["f1"],
                "roc_auc": chosen_metrics["roc_auc"],
            },
            "comparison": self.all_model_metrics,
        }

        with open(MODEL_DIR / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def load_saved_model() -> Tuple[Optional[Any], Optional[ColumnTransformer], Optional[Dict[str, Any]]]:
    """Load persisted model, preprocessor, and metadata if available."""
    model_path = MODEL_DIR / "churn_model.pkl"
    prep_path = MODEL_DIR / "preprocessor.pkl"
    meta_path = MODEL_DIR / "model_metadata.json"

    if model_path.exists() and prep_path.exists() and meta_path.exists():
        model = joblib.load(model_path)
        prep = joblib.load(prep_path)
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        return model, prep, metadata

    return None, None, None
