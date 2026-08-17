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


def render_transform_result_metrics(
    result: Dict[str, object], lang: str = DEFAULT_LANGUAGE
) -> None:
    """Render the PCA (Dimensionality Reduction) transform outcome as metric
    cards on the Prediction page."""
    col1, col2, col3 = st.columns(3)

    col1.metric(
        t("dimensionality_metric", lang),
        f"{result['n_features_before']} → {result['n_features_after']}",
    )
    col2.metric(
        t("variance_retained_metric", lang),
        f"{result['cumulative_variance_explained'] * 100:.2f}%",
    )
    col3.metric(
        t("transform_time_metric", lang), f"{result['transform_time'] * 1000:.2f} ms"
    )
