"""
dashboard.py
============
Dashboard page: project title, objective, dataset overview and class
distribution chart. Read-only — calls backend/api.py only.
"""

from __future__ import annotations

import streamlit as st

from backend import api
from backend.utils.i18n import DEFAULT_LANGUAGE, t
from frontend.components.metric_cards import render_dataset_metrics


def render(lang: str = DEFAULT_LANGUAGE) -> None:
    """Render the Dashboard page."""
    data = api.get_dashboard_data(lang)

    st.title(t("dashboard_title", lang))
    st.subheader(data["title"])

    with st.expander(t("project_objective_expander", lang), expanded=True):
        st.write(data["objective"])

    st.markdown(f"### {t('dataset_overview_header', lang)}")
    st.write(f"{t('dataset_label', lang)} {data['dataset_name']}")

    render_dataset_metrics(data["n_samples"], data["n_features"], lang=lang)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f"#### {t('class_distribution_header', lang)}")
        st.plotly_chart(api.get_class_distribution_chart(lang), use_container_width=True)

    with col_right:
        st.markdown(f"#### {t('feature_names_header', lang)}")
        with st.container(height=380):
            for i, name in enumerate(data["feature_names"], start=1):
                st.write(f"{i}. {name}")

    st.info(t("missing_values_info", lang, n=data["missing_values"]), icon="ℹ️")
