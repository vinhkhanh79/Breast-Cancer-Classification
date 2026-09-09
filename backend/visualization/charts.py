"""
charts.py
=========
Builds every Plotly figure used by the Streamlit frontend:
- Class distribution bar chart (Dashboard)
- Correlation heatmap (Visualization)
- Feature distribution histogram (Visualization)
- Explained variance chart (Visualization)
- PCA scatter 2D / 3D (Visualization)
- ROC curve (Visualization)
- Confusion matrix heatmap (Visualization / Prediction)
- Model comparison bar chart & radar chart (Model Comparison)

All charts are built with Plotly only (no static matplotlib images), as
required by the specification. Every function accepts a ``lang``
argument ("en" or "vi") so titles, axis labels and legends can be
rendered in the language currently selected in the app.
"""

from __future__ import annotations

from typing import Dict, List

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


def roc_curve_chart(
    fpr: List[float], tpr: List[float], roc_auc: float, label: str, lang: str = DEFAULT_LANGUAGE
) -> go.Figure:
    """ROC curve for a single model/PCA configuration."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{label} (AUC = {roc_auc:.3f})",
            line=dict(color=PRIMARY_BLUE, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name=t("legend_random_classifier", lang),
            line=dict(color="gray", dash="dash"),
        )
    )
    fig.update_layout(
        title=t("chart_roc_curve", lang),
        xaxis_title=t("axis_false_positive_rate", lang),
        yaxis_title=t("axis_true_positive_rate", lang),
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
    )
    return fig


def confusion_matrix_chart(
    cm: List[List[int]], target_names: List[str], lang: str = DEFAULT_LANGUAGE
) -> go.Figure:
    """Confusion matrix heatmap."""
    dl = _diagnosis_labels(lang)
    display_names = [dl.get(name, name) for name in target_names]
    fig = px.imshow(
        cm,
        x=display_names,
        y=display_names,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(
            x=t("axis_predicted", lang), y=t("axis_actual", lang), color=t("axis_count", lang)
        ),
        title=t("chart_confusion_matrix", lang),
    )
    fig.update_layout(paper_bgcolor=BACKGROUND)
    return fig


def comparison_bar_chart(
    comparison_df: pd.DataFrame, metric: str = "accuracy", lang: str = DEFAULT_LANGUAGE
) -> go.Figure:
    """Grouped bar chart comparing a metric across models and PCA configs."""
    metric_label = t(f"col_{metric}", lang)
    fig = px.bar(
        comparison_df,
        x="model",
        y=metric,
        color="pca_config",
        barmode="group",
        title=t("chart_comparison_bar", lang, metric=metric_label),
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig.update_layout(
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        yaxis_title=metric_label,
        xaxis_title=t("col_model", lang),
    )
    return fig


def comparison_radar_chart(
    comparison_df: pd.DataFrame, pca_config: str, lang: str = DEFAULT_LANGUAGE
) -> go.Figure:
    """Radar chart comparing Accuracy/Precision/Recall/F1/ROC-AUC across models
    for one selected PCA configuration."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = [t(f"col_{m}", lang) for m in metrics]
    subset = comparison_df[comparison_df["pca_config"] == pca_config]
    pca_label = subset["pca_label"].iloc[0] if not subset.empty else pca_config

    fig = go.Figure()
    for _, row in subset.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=metric_labels,
                fill="toself",
                name=row["model"],
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=t("chart_comparison_radar", lang, pca_config=pca_label),
        paper_bgcolor=BACKGROUND,
    )
    return fig
