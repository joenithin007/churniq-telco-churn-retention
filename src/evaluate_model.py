"""Model evaluation metrics, confusion matrix, ROC curve, and model comparison (Dark Theme)."""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
import plotly.graph_objects as go


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics for binary classification."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def plot_confusion_matrix(cm_matrix: List[List[int]], model_name: str = "") -> go.Figure:
    """Plot an interactive annotated confusion matrix in Dark Theme."""
    z = np.array(cm_matrix)
    x = ["Predicted Stay (0)", "Predicted Churn (1)"]
    y = ["Actual Stay (0)", "Actual Churn (1)"]

    tn, fp = z[0][0], z[0][1]
    fn, tp = z[1][0], z[1][1]

    annotations_text = [
        [f"True Neg (Stay)<br><b>{tn:,}</b>", f"False Pos (Alarm)<br><b>{fp:,}</b>"],
        [f"False Neg (Missed)<br><b>{fn:,}</b>", f"True Pos (Caught)<br><b>{tp:,}</b>"]
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0, "#1E293B"], [0.5, "#312E81"], [1.0, "#6366F1"]],
            showscale=False,
            text=annotations_text,
            texttemplate="%{text}",
            textfont=dict(size=13, color="#FFFFFF"),
            hoverinfo="z",
        )
    )

    title = f"<b>Confusion Matrix</b> ({model_name})" if model_name else "<b>Confusion Matrix</b>"
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(title="Model Prediction", side="bottom", tickfont=dict(color="#F8FAFC")),
        yaxis=dict(title="Actual Label", autorange="reversed", tickfont=dict(color="#F8FAFC")),
        plot_bgcolor="#1E293B",
        paper_bgcolor="#1E293B",
        height=320,
        margin=dict(l=60, r=40, t=40, b=40),
    )
    return fig


def plot_roc_curve(fpr: List[float], tpr: List[float], auc_score: float, model_name: str = "") -> go.Figure:
    """Plot ROC Curve in Dark Theme."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{model_name} (AUC = {auc_score:.3f})",
            line=dict(color="#818CF8", width=3),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess (0.50)",
            line=dict(color="#64748B", width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>ROC Curve</b> (AUC = {auc_score:.3f})", font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(title="False Positive Rate", range=[0, 1], gridcolor="#334155", tickfont=dict(color="#94A3B8")),
        yaxis=dict(title="True Positive Rate (Recall)", range=[0, 1.05], gridcolor="#334155", tickfont=dict(color="#94A3B8")),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(color="#F8FAFC")),
        plot_bgcolor="#1E293B",
        paper_bgcolor="#1E293B",
        height=320,
        margin=dict(l=50, r=30, t=40, b=50),
    )
    return fig


def format_model_comparison_table(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format comparative model evaluation metrics for side-by-side display."""
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m["model_name"],
            "Accuracy": f"{m['accuracy'] * 100:.1f}%",
            "Precision": f"{m['precision'] * 100:.1f}%",
            "Recall (Churn)": f"{m['recall'] * 100:.1f}%",
            "F1 Score": f"{m['f1'] * 100:.1f}%",
            "ROC-AUC": f"{m['roc_auc']:.3f}",
        })
    return pd.DataFrame(rows)
