"""
visualization.py
=================
Visualization page: explained variance, PCA scatter 2D/3D,
correlation heatmap, feature distribution — all via Plotly, all built
by backend/api.py.
"""

from __future__ import annotations

import streamlit as st

from backend import api
from backend.utils.i18n import DEFAULT_LANGUAGE, t


def render(lang: str = DEFAULT_LANGUAGE) -> None:
    """Render the Visualization page."""
    st.title(t("visualization_title", lang))

    tabs = st.tabs(
        [
            t("tab_pca_analysis", lang),
            t("tab_correlation", lang),
            t("tab_feature_distribution", lang),
        ]
    )

    with tabs[0]:
        _render_pca_analysis(lang)

    with tabs[1]:
        st.plotly_chart(api.get_correlation_heatmap(lang), use_container_width=True)

    with tabs[2]:
        _render_feature_distribution(lang)


def _render_pca_analysis(lang: str) -> None:
    try:
        st.plotly_chart(api.get_explained_variance_chart(lang), use_container_width=True)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    pca_options = api.get_pca_options(lang)
    pca_key = st.selectbox(
        t("pca_scatter_selector_label", lang),
        options=list(pca_options.keys()),
        format_func=lambda k: pca_options[k],
        key="viz_pca_key",
    )

    summary = api.get_pca_config_summary(pca_key)
    col1, col2 = st.columns(2)
    col1.metric(t("dimensionality_metric", lang), f"30 → {summary['n_components']}")
    col2.metric(
        t("variance_retained_metric", lang),
        f"{summary['cumulative_variance_explained'] * 100:.2f}%",
    )
    st.caption(t("pca_scatter_invariance_note", lang))

    fig_2d, fig_3d = api.get_pca_scatter_charts(pca_key, lang)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_2d, use_container_width=True)
    with col2:
        st.plotly_chart(fig_3d, use_container_width=True)


def _render_feature_distribution(lang: str) -> None:
    feature_names = api.get_all_feature_names()
    feature = st.selectbox(t("feature_selector_label", lang), options=feature_names)
    st.plotly_chart(api.get_feature_distribution_chart(feature, lang), use_container_width=True)
