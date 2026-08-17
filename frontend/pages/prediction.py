"""
prediction.py
=============
Prediction page: single-patient manual dimensionality-reduction form +
batch transform via CSV upload. Every transform goes through
backend/api.py only — this file never touches sklearn, joblib, or
.pkl files directly. There is no classification model in this
application; only the PCA (Dimensionality Reduction) transform is
applied and its resulting components are shown.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backend import api
from backend.utils.i18n import DEFAULT_LANGUAGE, t
from frontend.components.metric_cards import render_transform_result_metrics
from frontend.components.patient_form import render_patient_form


def render(lang: str = DEFAULT_LANGUAGE) -> None:
    """Render the Prediction (Dimensionality Reduction) page."""
    st.title(t("prediction_title", lang))
    st.caption(t("prediction_page_caption", lang))

    pca_options = api.get_pca_options(lang)
    pca_key = st.selectbox(
        t("pca_components_label", lang),
        options=list(pca_options.keys()),
        format_func=lambda k: pca_options[k],
    )

    tab_manual, tab_upload = st.tabs([t("tab_manual", lang), t("tab_upload", lang)])

    with tab_manual:
        _render_manual_transform(pca_key, lang)

    with tab_upload:
        _render_batch_transform(pca_key, lang)


def _render_manual_transform(pca_key: str, lang: str) -> None:
    feature_defaults = api.get_feature_defaults()
    st.caption(t("manual_caption", lang))

    values = render_patient_form(feature_defaults, lang)

    if st.button(t("transform_button", lang), type="primary", use_container_width=True):
        try:
            result = api.run_single_transform(pca_key, values, lang)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return

        st.divider()
        render_transform_result_metrics(result, lang)

        st.markdown(f"#### {t('components_header', lang)}")
        components_df = pd.DataFrame(
            {
                t("component_col", lang): [
                    f"PC{i + 1}" for i in range(len(result["components"]))
                ],
                t("component_value_col", lang): result["components"],
            }
        )
        st.dataframe(components_df, use_container_width=True, hide_index=True)


def _render_batch_transform(pca_key: str, lang: str) -> None:
    st.caption(t("upload_caption", lang))
    uploaded_file = st.file_uploader(t("upload_prompt", lang), type=["csv"])

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)
    st.write(t("loaded_rows", lang, n=len(df)))

    if st.button(t("transform_all_button", lang), type="primary"):
        try:
            result_df = api.run_batch_transform(pca_key, df, lang)
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
            return

        st.success(t("transformed_success", lang, n=len(result_df)))
        st.dataframe(result_df, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            t("download_button", lang),
            data=csv_bytes,
            file_name="pca_transform_result.csv",
            mime="text/csv",
            use_container_width=True,
        )
