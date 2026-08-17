"""
api.py
======
The ONE entry point the Streamlit frontend is allowed to call.

Architecture:

    Frontend  --calls-->  backend/api.py  --calls-->  backend/services/*
                                                             |
                                                  PCA (Dimensionality Reduction)

The frontend never imports sklearn, never loads a .pkl file, and never
touches backend/services or backend/preprocessing directly — every such
need is exposed here as a small, well-documented function.

Every function that returns user-facing text or a chart accepts a
``lang`` argument ("en" or "vi", default "en") so the whole application
can be shown in English or Vietnamese.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go

from backend.preprocessing.preprocess import (
    get_dataset_summary,
    get_feature_means,
    get_feature_names,
    load_dataset,
)
from backend.services.prediction_service import (
    available_pca_keys,
    transform_batch,
    transform_single,
)
from backend.utils.helpers import load_artifact
from backend.utils.i18n import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, t
from backend.visualization import charts

# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

def get_supported_languages() -> Dict[str, str]:
    """Return {lang_code: display_name}, e.g. {"en": "English", "vi": "Tiếng Việt"}."""
    return dict(SUPPORTED_LANGUAGES)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def get_dashboard_data(lang: str = DEFAULT_LANGUAGE) -> Dict[str, Any]:
    """Return everything the Dashboard page needs: project info + dataset summary."""
    summary = get_dataset_summary()
    project_info = {
        "title": t("project_title", lang),
        "objective": t("project_objective", lang),
        "dataset_name": t("dataset_name", lang),
    }
    return {**project_info, **summary}


def get_class_distribution_chart(lang: str = DEFAULT_LANGUAGE) -> go.Figure:
    summary = get_dataset_summary()
    return charts.class_distribution_chart(summary["class_distribution"], lang)


# ---------------------------------------------------------------------------
# Prediction (Dimensionality Reduction transform)
# ---------------------------------------------------------------------------

def get_pca_options(lang: str = DEFAULT_LANGUAGE) -> Dict[str, str]:
    """Return {pca_key: display_name} for the PCA components selector.

    Only configurations that actually reduce dimensionality are offered
    (the "no_pca" configuration is not a Dimensionality Reduction result).
    """
    labels = {
        "pca_10": {"en": "PCA (10 components)", "vi": "PCA (10 thành phần)"},
        "pca_15": {"en": "PCA (15 components)", "vi": "PCA (15 thành phần)"},
        "pca_20": {"en": "PCA (20 components)", "vi": "PCA (20 thành phần)"},
        "pca_95": {"en": "PCA (95% variance)", "vi": "PCA (95% phương sai)"},
        "pca_99": {"en": "PCA (99% variance)", "vi": "PCA (99% phương sai)"},
    }
    return {key: labels[key].get(lang, labels[key][DEFAULT_LANGUAGE]) for key in available_pca_keys()}


def get_feature_defaults() -> Dict[str, float]:
    """Return the 30 feature names mapped to their dataset mean (form defaults)."""
    return get_feature_means()


def run_single_transform(
    pca_key: str, feature_values: Dict[str, float], lang: str = DEFAULT_LANGUAGE
) -> Dict[str, Any]:
    """Reduce the dimensionality of one patient record entered manually in the form."""
    return transform_single(pca_key, feature_values)


def run_batch_transform(
    pca_key: str, df: pd.DataFrame, lang: str = DEFAULT_LANGUAGE
) -> pd.DataFrame:
    """Reduce the dimensionality of every row of an uploaded patient CSV file."""
    return transform_batch(pca_key, df)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def get_correlation_heatmap(lang: str = DEFAULT_LANGUAGE) -> go.Figure:
    X, _ = load_dataset()
    return charts.correlation_heatmap(X, lang=lang)


def get_feature_distribution_chart(feature: str, lang: str = DEFAULT_LANGUAGE) -> go.Figure:
    X, y = load_dataset()
    return charts.feature_distribution_chart(X, y, feature, lang)


def get_all_feature_names() -> List[str]:
    return get_feature_names()


def get_explained_variance_chart(lang: str = DEFAULT_LANGUAGE) -> go.Figure:
    pca_full = load_artifact("pca_full.pkl")
    return charts.explained_variance_chart(pca_full, lang)


def get_pca_scatter_charts(
    pca_key: str, lang: str = DEFAULT_LANGUAGE
) -> Tuple[go.Figure, go.Figure]:
    """Return (2D scatter, 3D scatter) figures for a given PCA configuration.

    Falls back to fitting a fresh 3-component PCA on the fly for the 3D
    view when the persisted PCA has fewer than 3 components.

    Note: PCA components are nested, so PC1/PC2/PC3 are identical no
    matter how many total components a configuration keeps — only the
    dimensionality and variance retained (see get_pca_config_summary)
    actually differ between configurations.
    """
    _, y = load_dataset()
    scaler = load_artifact("scaler.pkl")
    X, _ = load_dataset()
    X_scaled = scaler.transform(X)

    pca = load_artifact(f"pca_{pca_key}.pkl")
    if pca.n_components_ < 3:
        from sklearn.decomposition import PCA as _PCA

        pca_viz = _PCA(n_components=3, random_state=42)
        X_pca_viz = pca_viz.fit_transform(X_scaled)
    else:
        X_pca_viz = pca.transform(X_scaled)

    fig_2d = charts.pca_scatter_2d(X_pca_viz, y.to_numpy(), lang)
    fig_3d = charts.pca_scatter_3d(X_pca_viz, y.to_numpy(), lang)
    return fig_2d, fig_3d


def get_pca_config_summary(pca_key: str) -> Dict[str, Any]:
    """Return {n_components, cumulative_variance_explained} for a PCA
    configuration, so the UI can show what actually changes between
    configurations (PC1/PC2/PC3 themselves do not)."""
    pca = load_artifact(f"pca_{pca_key}.pkl")
    return {
        "n_components": int(pca.n_components_),
        "cumulative_variance_explained": float(pca.explained_variance_ratio_.sum()),
    }
