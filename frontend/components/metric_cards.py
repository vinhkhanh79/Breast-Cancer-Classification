"""
metric_cards.py
================
Reusable metric-card layouts built on top of st.metric() / st.columns().
Pure UI component — receives already-computed values, no ML logic.
Text is localized via backend/utils/i18n.py.
"""

from __future__ import annotations

from typing import Dict

import streamlit as st

from backend.utils.i18n import DEFAULT_LANGUAGE, t


def render_dataset_metrics(
    n_samples: int, n_features: int, n_classes: int = 2, lang: str = DEFAULT_LANGUAGE
) -> None:
    """Render the top-row metric cards on the Dashboard page."""
    col1, col2, col3 = st.columns(3)
    col1.metric(t("metric_total_samples", lang), f"{n_samples}")
    col2.metric(t("metric_num_features", lang), f"{n_features}")
    col3.metric(t("metric_num_classes", lang), f"{n_classes}")


def render_prediction_result_metrics(
    result: Dict[str, object], lang: str = DEFAULT_LANGUAGE
) -> None:
    """Render the prediction outcome as metric cards on the Prediction page."""
    col1, col2, col3 = st.columns(3)

    label = result["prediction_label"]
    is_malignant = result["prediction"] == "malignant"
    delta_color = "inverse" if is_malignant else "normal"
    col1.metric(
        t("prediction_metric_label", lang),
        label,
        delta=t("review_delta", lang) if is_malignant else t("normal_delta", lang),
        delta_color=delta_color,
    )
    col2.metric(t("prob_benign_metric", lang), f"{result['probability_benign'] * 100:.2f}%")
    col3.metric(
        t("prediction_time_metric", lang), f"{result['prediction_time'] * 1000:.2f} ms"
    )


def render_comparison_summary_metrics(
    best_accuracy_row: Dict[str, object],
    best_f1_row: Dict[str, object],
    lang: str = DEFAULT_LANGUAGE,
) -> None:
    """Render best-model highlight cards on the Model Comparison page."""
    col1, col2 = st.columns(2)
    col1.metric(
        t("best_accuracy_metric", lang),
        f"{best_accuracy_row['accuracy'] * 100:.2f}%",
        help=f"{best_accuracy_row['model']} — {best_accuracy_row['pca_label']}",
    )
    col2.metric(
        t("best_f1_metric", lang),
        f"{best_f1_row['f1'] * 100:.2f}%",
        help=f"{best_f1_row['model']} — {best_f1_row['pca_label']}",
    )
