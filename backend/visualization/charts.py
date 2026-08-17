"""
charts.py
=========
Builds every Plotly figure used by the Streamlit frontend:
- Class distribution bar chart (Dashboard)
- Correlation heatmap (Visualization)
- Feature distribution histogram (Visualization)
- Explained variance chart (Visualization)
- PCA scatter 2D / 3D (Visualization)

All charts are built with Plotly only (no static matplotlib images), as
required by the specification. Every function accepts a ``lang``
argument ("en" or "vi") so titles, axis labels and legends can be
rendered in the language currently selected in the app.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from backend.utils.i18n import DEFAULT_LANGUAGE, t

# Professional blue/white palette
PRIMARY_BLUE = "#1f4e8c"
SECONDARY_BLUE = "#4a90d9"
ACCENT = "#e74c3c"
BACKGROUND = "#ffffff"


def _diagnosis_labels(lang: str) -> Dict[str, str]:
    return {
        "malignant": t("diagnosis_malignant", lang),
        "benign": t("diagnosis_benign", lang),
    }


def class_distribution_chart(
    class_distribution: Dict[str, int], lang: str = DEFAULT_LANGUAGE
) -> go.Figure:
    """Bar chart of benign vs malignant sample counts."""
    dl = _diagnosis_labels(lang)
    labels = [dl.get(k, k) for k in class_distribution]
    values = list(class_distribution.values())
    fig = px.bar(
        x=labels,
        y=values,
        text=values,
        color=labels,
        color_discrete_sequence=[ACCENT, PRIMARY_BLUE],
        labels={"x": t("axis_diagnosis", lang), "y": t("axis_num_samples", lang)},
        title=t("chart_class_distribution", lang),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor=BACKGROUND, paper_bgcolor=BACKGROUND)
    return fig


def correlation_heatmap(
    X: pd.DataFrame, top_n: int = 15, lang: str = DEFAULT_LANGUAGE
) -> go.Figure:
    """Correlation heatmap for the top-N most variable features (readability)."""
    top_features = X.var().sort_values(ascending=False).head(top_n).index.tolist()
    corr = X[top_features].corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=t("chart_correlation_heatmap", lang, n=top_n),
        aspect="auto",
    )
    fig.update_layout(paper_bgcolor=BACKGROUND)
    return fig


def feature_distribution_chart(
    X: pd.DataFrame, y: pd.Series, feature: str, lang: str = DEFAULT_LANGUAGE
) -> go.Figure:
    """Histogram of a single feature, split by diagnosis class."""
    dl = _diagnosis_labels(lang)
    df = X[[feature]].copy()
    diagnosis_col = t("axis_diagnosis", lang)
    df[diagnosis_col] = y.map({0: dl["malignant"], 1: dl["benign"]})
    fig = px.histogram(
        df,
        x=feature,
        color=diagnosis_col,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={dl["malignant"]: ACCENT, dl["benign"]: PRIMARY_BLUE},
        title=t("chart_feature_distribution", lang, feature=feature),
    )
    fig.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND)
    return fig


def explained_variance_chart(pca_full: PCA, lang: str = DEFAULT_LANGUAGE) -> go.Figure:
    """Explained variance ratio (bar) + cumulative explained variance (line)."""
    ratio = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    n = len(ratio)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(1, n + 1)),
            y=ratio,
            name=t("legend_explained_variance_ratio", lang),
            marker_color=SECONDARY_BLUE,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, n + 1)),
            y=cumulative,
            name=t("legend_cumulative_variance", lang),
            mode="lines+markers",
            marker_color=ACCENT,
            yaxis="y",
        )
    )
    fig.add_hline(y=0.95, line_dash="dash", line_color="gray", annotation_text="95%")
    fig.add_hline(y=0.99, line_dash="dot", line_color="gray", annotation_text="99%")
    fig.update_layout(
        title=t("chart_explained_variance", lang),
        xaxis_title=t("axis_principal_component", lang),
        yaxis_title=t("axis_explained_variance_ratio", lang),
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def pca_scatter_2d(X_pca: np.ndarray, y: np.ndarray, lang: str = DEFAULT_LANGUAGE) -> go.Figure:
    """2D scatter plot of the first two principal components."""
    dl = _diagnosis_labels(lang)
    diagnosis_col = t("axis_diagnosis", lang)
    df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1]})
    df[diagnosis_col] = [dl["malignant"] if v == 0 else dl["benign"] for v in y]
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color=diagnosis_col,
        color_discrete_map={dl["malignant"]: ACCENT, dl["benign"]: PRIMARY_BLUE},
        title=t("chart_pca_scatter_2d", lang),
        opacity=0.75,
    )
    fig.update_layout(paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND)
    return fig


def pca_scatter_3d(X_pca: np.ndarray, y: np.ndarray, lang: str = DEFAULT_LANGUAGE) -> go.Figure:
    """3D scatter plot of the first three principal components."""
    dl = _diagnosis_labels(lang)
    diagnosis_col = t("axis_diagnosis", lang)
    df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "PC3": X_pca[:, 2]})
    df[diagnosis_col] = [dl["malignant"] if v == 0 else dl["benign"] for v in y]
    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color=diagnosis_col,
        color_discrete_map={dl["malignant"]: ACCENT, dl["benign"]: PRIMARY_BLUE},
        title=t("chart_pca_scatter_3d", lang),
        opacity=0.75,
    )
    fig.update_layout(paper_bgcolor=BACKGROUND)
    return fig



