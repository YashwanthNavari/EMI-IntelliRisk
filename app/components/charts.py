import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def plot_probability_distribution(probs: Dict[str, float]) -> go.Figure:
    """Plot an elegant horizontal bar chart for multiclass prediction confidence."""
    classes = list(probs.keys())
    values = [probs[c] * 100 for c in classes]

    color_map = {
        "Eligible": "#10B981",
        "High_Risk": "#F59E0B",
        "Not_Eligible": "#EF4444"
    }
    colors = [color_map.get(c, "#6366F1") for c in classes]

    fig = go.Figure(go.Bar(
        x=values,
        y=classes,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="auto"
    ))

    fig.update_layout(
        title="Class Probability Distribution",
        xaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F3F4F6", family="Plus Jakarta Sans"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=220
    )
    return fig

def plot_gauge_meter(value: float, title: str, max_val: float = 100.0, threshold: float = 50.0, suffix: str = "%") -> go.Figure:
    """Plot a speedometer gauge for FOIR / Risk indicators."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix=suffix, font=dict(size=24, color="#FFFFFF", family="Outfit")),
        title=dict(text=title, font=dict(size=14, color="#94A3B8")),
        gauge=dict(
            axis=dict(range=[0, max_val], tickwidth=1, tickcolor="#94A3B8"),
            bar=dict(color="#6366F1"),
            bgcolor="rgba(255,255,255,0.05)",
            borderwidth=1,
            bordercolor="rgba(255,255,255,0.1)",
            steps=[
                dict(range=[0, threshold], color="rgba(16, 185, 129, 0.2)"),
                dict(range=[threshold, max_val], color="rgba(239, 68, 68, 0.25)")
            ],
            threshold=dict(
                line=dict(color="#EF4444", width=3),
                thickness=0.75,
                value=threshold
            )
        )
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F3F4F6"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=180
    )
    return fig

def plot_budget_breakdown(monthly_salary: float, expenses: Dict[str, float], current_emi: float, disposable_income: float) -> go.Figure:
    """Plot an interactive donut chart of applicant monthly cash flow."""
    labels = list(expenses.keys()) + ["Current EMI", "Disposable Surplus"]
    values = list(expenses.values()) + [current_emi, max(0.0, disposable_income)]

    colors = px.colors.qualitative.Prism[:len(expenses)] + ["#F59E0B", "#10B981"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0B0F19", width=2)),
        textinfo="label+percent",
        hoverinfo="label+value"
    )])

    fig.update_layout(
        title="Monthly Cash Flow Allocation",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F3F4F6", family="Plus Jakarta Sans"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=40),
        height=340
    )
    return fig

def plot_confusion_matrix_heatmap(cm: List[List[int]], classes: List[str]) -> go.Figure:
    """Plot an annotated confusion matrix heatmap."""
    z = np.array(cm)
    fig = px.imshow(
        z,
        x=classes,
        y=classes,
        color_continuous_scale="Viridis",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        text_auto=True
    )
    fig.update_layout(
        title="Multiclass Confusion Matrix",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F3F4F6"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=380
    )
    return fig

def plot_feature_importance_bar(df_imp: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Plot horizontal bar chart of global feature importances."""
    df_top = df_imp.head(top_n).sort_values(by="importance_pct", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_top["importance_pct"],
        y=df_top["feature"],
        orientation="h",
        marker=dict(
            color=df_top["importance_pct"],
            colorscale="Purples",
            showscale=False
        ),
        text=[f"{v:.1f}%" for v in df_top["importance_pct"]],
        textposition="auto"
    ))

    fig.update_layout(
        title=f"Top {top_n} Driving Features (Global Attribution)",
        xaxis=dict(title="Relative Importance (%)", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F3F4F6", family="Plus Jakarta Sans"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    return fig
