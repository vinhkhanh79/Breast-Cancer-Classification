"""
api.py
======
The ONE entry point the Streamlit frontend is allowed to call.

Architecture:

    Frontend  --calls-->  backend/api.py  --calls-->  backend/services/*
                                                             |
                                                      Machine Learning Models

The frontend never imports sklearn, never loads a .pkl file, and never
touches backend/services or backend/preprocessing directly — every such
need is exposed here as a small, well-documented function.

Every function that returns user-facing text or a chart accepts a
``lang`` argument ("en" or "vi", default "en") so the whole application
can be shown in English or Vietnamese.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go

from backend.preprocessing.preprocess import (
    PCA_TAGS,
    TARGET_NAMES,
    get_dataset_summary,
    get_feature_means,
    get_feature_names,
    load_dataset,
)
from backend.services.prediction_service import (
    MODEL_FILES,
    predict_batch,
    predict_single,
)
from backend.utils.helpers import (
    RESULTS_METRICS_DIR,
    RESULTS_TABLES_DIR,
    load_artifact,
)
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
# Prediction
# ---------------------------------------------------------------------------

def get_model_options() -> Dict[str, str]:
    """Return {model_key: display_name} for the model selector.

    The app intentionally keeps a single classifier: Logistic Regression.
    """
    return dict(MODEL_FILES)


def get_pca_options(lang: str = DEFAULT_LANGUAGE) -> Dict[str, str]:
    """Return {pca_key: display_name} for the PCA components selector."""
    labels = {
        "no_pca": {"en": "No PCA", "vi": "Không dùng PCA"},
        "pca_10": {"en": "PCA (10 components)", "vi": "PCA (10 thành phần)"},
        "pca_15": {"en": "PCA (15 components)", "vi": "PCA (15 thành phần)"},
        "pca_20": {"en": "PCA (20 components)", "vi": "PCA (20 thành phần)"},
        "pca_95": {"en": "PCA (95% variance)", "vi": "PCA (95% phương sai)"},
        "pca_99": {"en": "PCA (99% variance)", "vi": "PCA (99% phương sai)"},
    }
    return {key: labels[key].get(lang, labels[key][DEFAULT_LANGUAGE]) for key in PCA_TAGS}


def get_feature_defaults() -> Dict[str, float]:
    """Return the 30 feature names mapped to their dataset mean (form defaults)."""
    return get_feature_means()


def run_single_prediction(
    model_key: str, pca_key: str, feature_values: Dict[str, float], lang: str = DEFAULT_LANGUAGE
) -> Dict[str, Any]:
    """Predict one patient record entered manually in the form."""
    result = predict_single(model_key, pca_key, feature_values)
    result["prediction_label"] = (
        t("diagnosis_malignant", lang) if result["prediction"] == "malignant"
        else t("diagnosis_benign", lang)
    )
    return result


def run_batch_prediction(
    model_key: str, pca_key: str, df: pd.DataFrame, lang: str = DEFAULT_LANGUAGE
) -> pd.DataFrame:
    """Predict every row of an uploaded patient CSV file."""
    result_df = predict_batch(model_key, pca_key, df)
    mapping = {"Malignant": t("diagnosis_malignant", lang), "Benign": t("diagnosis_benign", lang)}
    result_df["prediction"] = result_df["prediction"].map(mapping)
    return result_df


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
    """
    _, y = load_dataset()
    scaler = load_artifact("scaler.pkl")
    X, _ = load_dataset()
    X_scaled = scaler.transform(X)

    pca = load_artifact(f"pca_{pca_key}.pkl") if pca_key != "no_pca" else None
    if pca is None or pca.n_components_ < 3:
        from sklearn.decomposition import PCA as _PCA

        pca_viz = _PCA(n_components=3, random_state=42)
        X_pca_viz = pca_viz.fit_transform(X_scaled)
    else:
        X_pca_viz = pca.transform(X_scaled)

    fig_2d = charts.pca_scatter_2d(X_pca_viz, y.to_numpy(), lang)
    fig_3d = charts.pca_scatter_3d(X_pca_viz, y.to_numpy(), lang)
    return fig_2d, fig_3d


def get_roc_curve_chart(model_key: str, pca_key: str, lang: str = DEFAULT_LANGUAGE) -> go.Figure:
    metrics = _load_metrics(model_key, pca_key)
    label = f"{MODEL_FILES[model_key]} / {get_pca_options(lang)[pca_key]}"
    return charts.roc_curve_chart(
        metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], metrics["roc_auc"], label, lang
    )


def get_confusion_matrix_chart(
    model_key: str, pca_key: str, lang: str = DEFAULT_LANGUAGE
) -> go.Figure:
    metrics = _load_metrics(model_key, pca_key)
    return charts.confusion_matrix_chart(metrics["confusion_matrix"], TARGET_NAMES, lang)


def _load_metrics(model_key: str, pca_key: str) -> Dict[str, Any]:
    path = RESULTS_METRICS_DIR / f"{model_key}_{pca_key}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path.name}. Please run 'python train.py' first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def get_comparison_table() -> pd.DataFrame:
    """Load the full model-comparison table produced by train.py."""
    path = RESULTS_TABLES_DIR / "comparison_table.csv"
    if not path.exists():
        raise FileNotFoundError(
            "comparison_table.csv not found. Please run 'python train.py' first."
        )
    return pd.read_csv(path)


def get_comparison_bar_chart(metric: str = "accuracy", lang: str = DEFAULT_LANGUAGE) -> go.Figure:
    df = get_comparison_table()
    return charts.comparison_bar_chart(df, metric, lang)


def get_comparison_radar_chart(pca_key: str, lang: str = DEFAULT_LANGUAGE) -> go.Figure:
    df = get_comparison_table()
    return charts.comparison_radar_chart(df, pca_key, lang)
