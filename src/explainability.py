"""Model explainability and SHAP interpretation module (Dark Theme)."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ModelExplainer:
    """Provides global and local model explanations using SHAP and model feature importance."""

    def __init__(self, model: object, preprocessor: object, feature_names: List[str]):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer: Optional[object] = None
        self._init_explainer()

    def _init_explainer(self) -> None:
        """Initialize SHAP explainer based on underlying model architecture."""
        if not HAS_SHAP or self.model is None:
            return

        model_type = type(self.model).__name__
        try:
            if "Forest" in model_type or "XGB" in model_type or "Gradient" in model_type:
                self.explainer = shap.TreeExplainer(self.model)
            elif "Logistic" in model_type or "Linear" in model_type:
                self.explainer = shap.LinearExplainer(self.model)
        except Exception:
            self.explainer = None

    def get_global_importance(self, df_sample: pd.DataFrame, max_features: int = 12) -> pd.DataFrame:
        """Compute top global features contributing to model predictions."""
        if self.model is None:
            return pd.DataFrame()

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        df_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances,
        })

        df_imp["Display_Feature"] = df_imp["Feature"].apply(self._clean_feature_name)
        df_grouped = df_imp.groupby("Display_Feature", as_index=False)["Importance"].sum()
        df_grouped = df_grouped.sort_values(by="Importance", ascending=False).head(max_features)
        
        total = df_grouped["Importance"].sum()
        if total > 0:
            df_grouped["Relative_Importance"] = (df_grouped["Importance"] / total * 100.0).round(1)
        else:
            df_grouped["Relative_Importance"] = 0.0

        return df_grouped

    def explain_individual(
        self,
        customer_row: pd.Series,
        raw_df: pd.DataFrame,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Explain model prediction for an individual customer."""
        expected_cols = []
        for name, trans, cols in self.preprocessor.transformers_:
            if name != "remainder" and cols:
                expected_cols.extend(cols)

        cust_df = pd.DataFrame([customer_row])[expected_cols]
        X_trans = self.preprocessor.transform(cust_df)

        contributions = []

        if HAS_SHAP and self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(X_trans)
                if isinstance(shap_vals, list) and len(shap_vals) > 1:
                    vals = shap_vals[1][0]
                elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
                    vals = shap_vals[0, :, 1]
                else:
                    vals = shap_vals[0]

                for feat, val in zip(self.feature_names, vals):
                    contributions.append({
                        "Feature": feat,
                        "Display_Feature": self._clean_feature_name(feat),
                        "Shap_Value": float(val),
                        "Direction": "Increases Risk" if val > 0 else "Decreases Risk",
                    })

                contributions.sort(key=lambda x: abs(x["Shap_Value"]), reverse=True)
                return contributions[:top_k]
            except Exception:
                pass

        if hasattr(self.model, "feature_importances_"):
            imps = self.model.feature_importances_
            for feat, imp in zip(self.feature_names, imps):
                contributions.append({
                    "Feature": feat,
                    "Display_Feature": self._clean_feature_name(feat),
                    "Shap_Value": float(imp),
                    "Direction": "Key Factor",
                })
            contributions.sort(key=lambda x: x["Shap_Value"], reverse=True)
            return contributions[:top_k]

        return []

    def _clean_feature_name(self, name: str) -> str:
        """Format column names into clean business labels."""
        clean = name.replace("num__", "").replace("cat__", "").replace("_", " ")
        if "Contract Type" in clean:
            return clean.replace("Contract Type", "Contract")
        if "Service Type" in clean:
            return clean.replace("Service Type", "Service")
        if "Payment Method" in clean:
            return clean.replace("Payment Method", "Payment")
        if "Subscription Plan" in clean:
            return clean.replace("Subscription Plan", "Plan")
        return clean.strip().title()


def plot_global_feature_importance(df_imp: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing top global features in Dark Theme."""
    if df_imp.empty:
        fig = go.Figure()
        fig.add_annotation(text="Feature importance unavailable.", x=0.5, y=0.5, showarrow=False, font=dict(color="#94A3B8"))
        fig.update_layout(plot_bgcolor="#1E293B", paper_bgcolor="#1E293B")
        return fig

    df_sorted = df_imp.sort_values(by="Importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_sorted["Relative_Importance"],
            y=df_sorted["Display_Feature"],
            orientation="h",
            marker=dict(
                color=df_sorted["Relative_Importance"],
                colorscale=[[0, "#818CF8"], [1.0, "#4F46E5"]],
                showscale=False,
            ),
            text=df_sorted["Relative_Importance"].apply(lambda v: f" {v:.1f}%"),
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=12),
            hovertemplate="<b>%{y}</b><br>Relative Influence: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="<b>Factors Influencing Model Predictions</b>", font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(title="Relative Influence (%)", gridcolor="#334155", tickfont=dict(color="#94A3B8")),
        yaxis=dict(title="", tickfont=dict(color="#F8FAFC", size=12)),
        plot_bgcolor="#1E293B",
        paper_bgcolor="#1E293B",
        height=320,
        margin=dict(l=140, r=20, t=40, b=40),
    )
    return fig


def plot_customer_explanation_waterfall(contributions: List[Dict[str, Any]], customer_id: str) -> go.Figure:
    """Horizontal impact chart displaying individual prediction drivers in Dark Theme."""
    if not contributions:
        fig = go.Figure()
        fig.add_annotation(text="Individual explanation not available for this record.", x=0.5, y=0.5, showarrow=False, font=dict(color="#94A3B8"))
        fig.update_layout(plot_bgcolor="#1E293B", paper_bgcolor="#1E293B")
        return fig

    df_c = pd.DataFrame(contributions).iloc[::-1]

    colors = [
        "#EF4444" if "Increase" in d or "Key" in d else "#10B981"
        for d in df_c["Direction"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df_c["Shap_Value"],
            y=df_c["Display_Feature"],
            orientation="h",
            marker=dict(color=colors),
            text=df_c["Direction"],
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=11),
            hovertemplate="<b>%{y}</b><br>Impact Value: %{x:.3f}<br>Direction: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>Contributing Drivers for {customer_id}</b>", font=dict(size=14, color="#F8FAFC")),
        xaxis=dict(title="Model Impact Direction (SHAP)", gridcolor="#334155", tickfont=dict(color="#94A3B8")),
        yaxis=dict(title="", tickfont=dict(color="#F8FAFC", size=12)),
        plot_bgcolor="#1E293B",
        paper_bgcolor="#1E293B",
        height=260,
        margin=dict(l=140, r=20, t=35, b=35),
    )
    return fig
