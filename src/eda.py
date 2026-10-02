"""Exploratory Data Analysis (EDA) and interactive Plotly visualization module (Dark Enterprise Theme)."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Cohesive Dark Enterprise Color Palette
COLORS = {
    "primary": "#6366F1",       # Indigo accent
    "churn": "#EF4444",         # Red (High Risk / Churned)
    "retained": "#10B981",      # Emerald Green (Retained / Low Risk)
    "medium": "#F59E0B",        # Amber (Medium Risk)
    "card_bg": "#1E293B",       # Slate 800
    "text": "#F8FAFC",          # Slate 50
    "text_muted": "#94A3B8",    # Slate 400
    "grid": "#334155",          # Slate 700
}


def get_empty_chart(message: str) -> go.Figure:
    """Return an empty figure with an informative message in dark theme."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=COLORS["text_muted"]),
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=COLORS["card_bg"],
        paper_bgcolor=COLORS["card_bg"],
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def plot_churn_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing proportion of Retained vs Churned customers with center churn rate."""
    if "Churn" not in df.columns or df.empty:
        return get_empty_chart("Data unavailable: 'Churn' column not found.")

    counts = df["Churn"].value_counts().rename(index={0: "Retained", 1: "Churned"})
    total = len(df)
    churned_count = counts.get("Churned", 0)
    churn_rate = (churned_count / total * 100.0) if total > 0 else 0.0

    labels = ["Retained", "Churned"]
    values = [counts.get("Retained", 0), counts.get("Churned", 0)]
    colors = [COLORS["retained"], COLORS["churn"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.65,
                marker=dict(colors=colors, line=dict(color="#0F172A", width=2)),
                textinfo="label+percent",
                textfont=dict(size=13, color="#FFFFFF"),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )

    # Center callout: churn rate and label
    fig.add_annotation(
        text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:11px;color:#94A3B8;'>Overall Churn</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="#FFFFFF"),
    )

    fig.update_layout(
        title=dict(text="<b>Churn Distribution</b>", font=dict(size=15, color=COLORS["text"])),
        showlegend=False,
        plot_bgcolor=COLORS["card_bg"],
        paper_bgcolor=COLORS["card_bg"],
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_risk_distribution(df: pd.DataFrame, risk_col: str = "Risk_Level") -> go.Figure:
    """Simple 3-bar chart displaying distribution across Low, Medium, and High Risk."""
    if risk_col not in df.columns or df.empty:
        return get_empty_chart("Data unavailable: Risk segmentation not yet calculated.")

    tier_order = ["Low Risk", "Medium Risk", "High Risk"]
    tier_colors = {"Low Risk": COLORS["retained"], "Medium Risk": COLORS["medium"], "High Risk": COLORS["churn"]}

    # Normalize values in df
    s = df[risk_col].astype(str).str.replace(" Risk", "") + " Risk"
    counts = s.value_counts().reindex(tier_order, fill_value=0)
    total = len(df)
    percentages = (counts / total * 100.0) if total > 0 else counts * 0.0

    fig = go.Figure()
    for tier in tier_order:
        cnt = int(counts.get(tier, 0))
        pct = float(percentages.get(tier, 0.0))
        fig.add_trace(
            go.Bar(
                x=[tier],
                y=[cnt],
                name=tier,
                marker=dict(color=tier_colors[tier], line=dict(color="#0F172A", width=1.5)),
                text=f"<b>{cnt:,}</b> ({pct:.1f}%)",
                textposition="auto",
                textfont=dict(color="#FFFFFF", size=12),
                hovertemplate=f"<b>{tier}</b><br>Count: {cnt:,}<br>Share: {pct:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="<b>Customer Risk Segmentation</b>", font=dict(size=15, color=COLORS["text"])),
        showlegend=False,
        plot_bgcolor=COLORS["card_bg"],
        paper_bgcolor=COLORS["card_bg"],
        yaxis=dict(title="Customers", gridcolor=COLORS["grid"], tickfont=dict(color=COLORS["text_muted"])),
        xaxis=dict(tickfont=dict(color=COLORS["text"])),
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def plot_top_churn_drivers_overview(df_imp: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """Horizontal bar chart showing the TOP 5 most important model-derived churn drivers."""
    if df_imp is None or df_imp.empty:
        return get_empty_chart("Top churn drivers not available.")

    df_top = df_imp.sort_values(by="Importance", ascending=True).tail(top_n)

    fig = go.Figure(
        go.Bar(
            x=df_top["Relative_Importance"],
            y=df_top["Display_Feature"],
            orientation="h",
            marker=dict(
                color=df_top["Relative_Importance"],
                colorscale=[[0, "#818CF8"], [1.0, "#4F46E5"]],
                showscale=False,
            ),
            text=df_top["Relative_Importance"].apply(lambda v: f" {v:.1f}%"),
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=12),
            hovertemplate="<b>%{y}</b><br>Relative Influence: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>Top {top_n} Churn Drivers (Model Influence)</b>", font=dict(size=15, color=COLORS["text"])),
        xaxis=dict(title="Relative Influence (%)", gridcolor=COLORS["grid"], tickfont=dict(color=COLORS["text_muted"])),
        yaxis=dict(title="", tickfont=dict(color=COLORS["text"], size=12)),
        plot_bgcolor=COLORS["card_bg"],
        paper_bgcolor=COLORS["card_bg"],
        height=260,
        margin=dict(l=140, r=20, t=40, b=30),
    )
    return fig


def plot_churn_rate_by_category(df: pd.DataFrame, category_col: str, title: str) -> go.Figure:
    """Interactive bar chart showing empirical Churn Rate (%) grouped by category."""
    if "Churn" not in df.columns or category_col not in df.columns or df.empty:
        return get_empty_chart(f"Data unavailable: '{category_col}' or 'Churn' not found.")

    agg = df.groupby(category_col).agg(
        Total=("Churn", "count"),
        Churned=("Churn", "sum")
    ).reset_index()

    agg["Churn_Rate"] = ((agg["Churned"] / agg["Total"]) * 100.0).round(1)
    agg = agg.sort_values(by="Churn_Rate", ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=agg[category_col],
            y=agg["Churn_Rate"],
            marker=dict(
                color=agg["Churn_Rate"],
                colorscale=[[0, "#38BDF8"], [0.5, "#F59E0B"], [1.0, "#EF4444"]],
                showscale=False,
            ),
            text=agg["Churn_Rate"].apply(lambda v: f"{v}%"),
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=12),
            hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<br>Total: %{customdata[0]:,}<br>Churned: %{customdata[1]:,}<extra></extra>",
            customdata=agg[["Total", "Churned"]].values,
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color=COLORS["text"])),
        xaxis=dict(title=category_col.replace("_", " "), tickfont=dict(color=COLORS["text"]), tickangle=-15),
        yaxis=dict(title="Churn Rate (%)", gridcolor=COLORS["grid"], tickfont=dict(color=COLORS["text_muted"])),
        plot_bgcolor=COLORS["card_bg"],
        paper_bgcolor=COLORS["card_bg"],
        height=320,
        margin=dict(l=40, r=20, t=40, b=50),
    )
    return fig


def plot_churn_distribution_numerical(df: pd.DataFrame, num_col: str, title: str) -> go.Figure:
    """Distribution histogram overlay comparing retained vs churned in dark theme."""
    if "Churn" not in df.columns or num_col not in df.columns or df.empty:
        return get_empty_chart(f"Data unavailable: '{num_col}' or 'Churn' not found.")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df[df["Churn"] == 0][num_col],
            name="Retained (0)",
            opacity=0.65,
            marker_color=COLORS["retained"],
            histnorm="probability density",
            nbinsx=30,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=df[df["Churn"] == 1][num_col],
            name="Churned (1)",
            opacity=0.65,
            marker_color=COLORS["churn"],
            histnorm="probability density",
            nbinsx=30,
        )
    )

    fig.update_layout(
        barmode="overlay",
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color=COLORS["text"])),
        xaxis=dict(title=num_col.replace("_", " "), tickfont=dict(color=COLORS["text"])),
        yaxis=dict(title="Density", gridcolor=COLORS["grid"], tickfont=dict(color=COLORS["text_muted"])),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color=COLORS["text"])),
        plot_bgcolor=COLORS["card_bg"],
        paper_bgcolor=COLORS["card_bg"],
        height=320,
        margin=dict(l=40, r=20, t=40, b=50),
    )
    return fig
