"""
prediction_service.py
======================
Runtime dimensionality-reduction path used by the Streamlit frontend
(via backend/api.py):

    Frontend  -->  backend/api.py  -->  prediction_service.py
                                              |
                                   load scaler -> load PCA -> transform

The web application NEVER re-fits PCA here. It only loads the
``.pkl`` artifacts produced by ``train.py`` and applies the PCA
(Dimensionality Reduction) transform to the input features.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.preprocessing.preprocess import get_feature_names
from backend.utils.helpers import load_artifact


def _transform_features(raw_features: pd.DataFrame, pca_key: str) -> np.ndarray:
    """Load the scaler + PCA for the given configuration and transform raw
    30-feature input into the reduced PCA component space."""
    scaler = load_artifact("scaler.pkl")
    scaled = scaler.transform(raw_features)

    pca = load_artifact(f"pca_{pca_key}.pkl")
    return pca.transform(scaled)


def transform_single(pca_key: str, feature_values: Dict[str, float]) -> Dict[str, Any]:
    """Apply dimensionality reduction to a single patient record entered
    manually in the Prediction page.

    Args:
        pca_key: one of {"pca_10", "pca_15", "pca_20", "pca_95", "pca_99"}.
        feature_values: dict of the 30 original feature values.

    Returns:
        Dict with the reduced components, dimensionality before/after,
        the explained variance ratio retained and the transform time.
    """
    start = time.perf_counter()

    feature_names = get_feature_names()
    raw = pd.DataFrame([[feature_values[name] for name in feature_names]], columns=feature_names)

    pca = load_artifact(f"pca_{pca_key}.pkl")
    X_transformed = _transform_features(raw, pca_key)

    elapsed = time.perf_counter() - start

    return {
        "components": X_transformed[0].tolist(),
        "n_features_before": len(feature_names),
        "n_features_after": X_transformed.shape[1],
        "cumulative_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
        "pca_config": pca_key,
        "transform_time": elapsed,
    }


def transform_batch(pca_key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply dimensionality reduction to every row of an uploaded CSV.

    The CSV must contain the 30 original feature columns (extra columns
    such as 'id' or 'diagnosis' are ignored if present).

    Returns:
        A copy of the input DataFrame with the reduced PCA components
        (PC1, PC2, ...) appended as new columns.
    """
    feature_names = get_feature_names()
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing}")

    raw = df[feature_names].apply(pd.to_numeric, errors="coerce")
    if raw.isna().any().any():
        raise ValueError("Uploaded CSV contains non-numeric or missing values.")

    X_transformed = _transform_features(raw, pca_key)

    result = df.copy()
    for i in range(X_transformed.shape[1]):
        result[f"PC{i + 1}"] = X_transformed[:, i].round(4)
    return result


def available_pca_keys() -> List[str]:
    """Return the list of supported PCA configuration keys, in display order."""
    return ["pca_10", "pca_15", "pca_20", "pca_95", "pca_99"]
