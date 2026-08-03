"""
prediction.py
=============
Prediction page: single-patient manual prediction form + batch prediction
via CSV upload. Every prediction goes through backend/api.py only — this
file never touches sklearn, joblib, or .pkl files directly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backend import api
from backend.utils.i18n import DEFAULT_LANGUAGE, t
from frontend.components.metric_cards import render_prediction_result_metrics
from frontend.components.patient_form import render_patient_form


def render(lang: str = DEFAULT_LANGUAGE) -> None:
    """Render the Prediction page."""
    st.title(t("prediction_title", lang))

    model_options = api.get_model_options()
    pca_options = api.get_pca_options(lang)

    col1, col2 = st.columns(2)
    with col1:
        model_key = st.selectbox(
            t("model_label", lang),
            options=list(model_options.keys()),
            format_func=lambda k: model_options[k],
        )
    with col2:
        pca_key = st.selectbox(
            t("pca_components_label", lang),
            options=list(pca_options.keys()),
            format_func=lambda k: pca_options[k],
        )

    tab_manual, tab_upload = st.tabs([t("tab_manual", lang), t("tab_upload", lang)])

    with tab_manual:
        _render_manual_prediction(model_key, pca_key, lang)

    with tab_upload:
        _render_batch_prediction(model_key, pca_key, lang)


def _render_manual_prediction(model_key: str, pca_key: str, lang: str) -> None:
    feature_defaults = api.get_feature_defaults()
    st.caption(t("manual_caption", lang))

    values = render_patient_form(feature_defaults, lang)

    if st.button(t("predict_button", lang), type="primary", use_container_width=True):
        try:
            result = api.run_single_prediction(model_key, pca_key, values, lang)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return

        st.divider()
        if result["prediction"] == "malignant":
            st.error(f"### {t('prediction_metric_label', lang)}: {result['prediction_label']} 🔴")
        else:
            st.success(f"### {t('prediction_metric_label', lang)}: {result['prediction_label']} 🟢")

        render_prediction_result_metrics(result, lang)

        st.markdown(f"#### {t('probability_breakdown_header', lang)}")
        prob_df = pd.DataFrame(
            {
                t("prob_table_class_col", lang): [
                    t("diagnosis_malignant", lang), t("diagnosis_benign", lang),
                ],
                t("prob_table_prob_col", lang): [
                    result["probability_malignant"],
                    result["probability_benign"],
                ],
            }
        )
        st.dataframe(prob_df, use_container_width=True, hide_index=True)


def _render_batch_prediction(model_key: str, pca_key: str, lang: str) -> None:
    st.caption(t("upload_caption", lang))
    uploaded_file = st.file_uploader(t("upload_prompt", lang), type=["csv"])

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)
    st.write(t("loaded_rows", lang, n=len(df)))

    if st.button(t("predict_all_button", lang), type="primary"):
        try:
            result_df = api.run_batch_prediction(model_key, pca_key, df, lang)
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
            return

        st.success(t("predicted_success", lang, n=len(result_df)))
        st.dataframe(result_df, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            t("download_button", lang),
            data=csv_bytes,
            file_name="prediction_result.csv",
            mime="text/csv",
            use_container_width=True,
        )
