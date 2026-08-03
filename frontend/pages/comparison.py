"""
comparison.py
=============
Model Comparison page: full metrics table across every (model, PCA
config) pair, plus bar chart and radar chart. Data comes from
results/tables/comparison_table.csv via backend/api.py.
"""

from __future__ import annotations

import streamlit as st

from backend import api
from backend.utils.i18n import DEFAULT_LANGUAGE, t
from frontend.components.metric_cards import render_comparison_summary_metrics


def render(lang: str = DEFAULT_LANGUAGE) -> None:
    """Render the Model Comparison page."""
    st.title(t("comparison_title", lang))

    try:
        df = api.get_comparison_table()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    best_accuracy_row = df.loc[df["accuracy"].idxmax()]
    best_f1_row = df.loc[df["f1"].idxmax()]
    render_comparison_summary_metrics(best_accuracy_row, best_f1_row, lang)

    st.markdown(f"### {t('full_table_header', lang)}")
    display_cols = [
        "model", "pca_label", "accuracy", "precision", "recall", "f1", "roc_auc",
        "training_time", "prediction_time", "n_features_before_pca", "n_features_after_pca",
    ]
    column_labels = {
        "model": t("col_model", lang),
        "pca_label": t("col_pca_config", lang),
        "accuracy": t("col_accuracy", lang),
        "precision": t("col_precision", lang),
        "recall": t("col_recall", lang),
        "f1": t("col_f1", lang),
        "roc_auc": t("col_roc_auc", lang),
        "training_time": t("col_training_time", lang),
        "prediction_time": t("col_prediction_time", lang),
        "n_features_before_pca": t("col_dim_before", lang),
        "n_features_after_pca": t("col_dim_after", lang),
    }
    st.dataframe(
        df[display_cols].rename(columns=column_labels),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(f"### {t('bar_chart_header', lang)}")
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric = st.selectbox(
        t("metric_selector_label", lang),
        options=metric_keys,
        format_func=lambda k: t(f"col_{k}", lang),
        index=0,
    )
    st.plotly_chart(api.get_comparison_bar_chart(metric, lang), use_container_width=True)

    st.markdown(f"### {t('radar_chart_header', lang)}")
    pca_options = api.get_pca_options(lang)
    pca_key = st.selectbox(
        t("pca_configuration_label", lang), options=list(pca_options.keys()),
        format_func=lambda k: pca_options[k], key="cmp_radar_pca_key",
    )
    st.plotly_chart(api.get_comparison_radar_chart(pca_key, lang), use_container_width=True)
