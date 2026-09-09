"""
patient_form.py
================
Renders the 30-feature patient input form used on the Prediction page.
Each field defaults to the dataset mean for that feature (provided by
the backend). Pure UI component — no ML logic, no direct model access.

Feature names themselves (radius_mean, texture_se, ...) are kept in
English as standard scientific/dataset terms; only the group tab
labels (Mean / Standard Error / Worst) are localized.
"""

from __future__ import annotations

from typing import Dict

import streamlit as st

from backend.utils.i18n import DEFAULT_LANGUAGE, t

# Group the 30 WDBC features into Mean / SE / Worst, matching how they
# are named in the dataset (e.g. radius_mean, radius_se, radius_worst).
FEATURE_GROUPS = {
    "feature_group_mean": [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    ],
    "feature_group_se": [
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave points_se", "symmetry_se", "fractal_dimension_se",
    ],
    "feature_group_worst": [
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
    ],
}


def render_patient_form(
    feature_defaults: Dict[str, float], lang: str = DEFAULT_LANGUAGE
) -> Dict[str, float]:
    """Render 30 number inputs (grouped into tabs) and return the entered values.

    Args:
        feature_defaults: dict of feature_name -> default value (dataset mean).
        lang: current UI language ("en" or "vi") for the tab labels.

    Returns:
        Dict of feature_name -> value currently entered by the user.
    """
    values: Dict[str, float] = {}
    group_keys = list(FEATURE_GROUPS.keys())
    tabs = st.tabs([t(key, lang) for key in group_keys])

    for tab, group_key in zip(tabs, group_keys):
        with tab:
            features = FEATURE_GROUPS[group_key]
            cols = st.columns(2)
            for i, feature in enumerate(features):
                with cols[i % 2]:
                    default_value = float(feature_defaults.get(feature, 0.0))
                    values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        value=default_value,
                        format="%.4f",
                        key=f"input_{feature}",
                    )
    return values
