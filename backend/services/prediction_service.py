"""
prediction_service.py
======================
Runtime prediction path used by the Streamlit frontend (via backend/api.py):

    Frontend  -->  backend/api.py  -->  prediction_service.py
                                              |
                                   load scaler -> load PCA (if any) -> load model -> predict

The web application NEVER re-trains a model here. It only loads the
``.pkl`` artifacts produced by ``train.py``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.preprocessing.preprocess import TARGET_NAMES, get_feature_names
from backend.utils.helpers import load_artifact

MODEL_FILES = {
    "logistic": "Logistic Regression",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "knn": "KNN",
}


def _artifact_name(model_key: str, pca_key: str, kind: str) -> str:
    """Build the artifact filename for a given model/pca configuration.

    kind is one of {"model", "pca"}. The scaler is shared and does not
    depend on model_key/pca_key.
    """
    if kind == "pca":
        return "scaler.pkl" if pca_key == "no_pca" else f"pca_{pca_key}.pkl"
    return f"{model_key}_{pca_key}.pkl"


def _transform_features(raw_features: pd.DataFrame, pca_key: str) -> np.ndarray:
    """Load scaler (+ PCA if applicable) and transform raw 30-feature input."""
    scaler = load_artifact("scaler.pkl")
    scaled = scaler.transform(raw_features)

    if pca_key == "no_pca":
        return scaled

    pca = load_artifact(f"pca_{pca_key}.pkl")
    return pca.transform(scaled)


def predict_single(
    model_key: str, pca_key: str, feature_values: Dict[str, float]
) -> Dict[str, Any]:
    """Predict a single patient record entered manually in the Prediction page.

    Args:
        model_key: one of {"logistic", "svm", "random_forest", "knn"}.
        pca_key: one of {"no_pca", "pca_10", "pca_15", "pca_20", "pca_95", "pca_99"}.
        feature_values: dict of the 30 original feature values.

    Returns:
        Dict with prediction label, probability, model/PCA info and
        prediction time in seconds.
    """
    start = time.perf_counter()

    feature_names = get_feature_names()
    raw = pd.DataFrame([[feature_values[name] for name in feature_names]], columns=feature_names)

    X_transformed = _transform_features(raw, pca_key)
    model = load_artifact(_artifact_name(model_key, pca_key, "model"))

    pred = int(model.predict(X_transformed)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_transformed)[0]
    else:
        # Fallback for models without predict_proba: derive a pseudo-probability
        score = model.decision_function(X_transformed)[0]
        proba_pos = 1 / (1 + np.exp(-score))
        proba = np.array([1 - proba_pos, proba_pos])

    elapsed = time.perf_counter() - start

    return {
        "prediction": TARGET_NAMES[pred],
        "prediction_label": "Malignant" if pred == 0 else "Benign",
        "probability_malignant": float(proba[0]),
        "probability_benign": float(proba[1]),
        "model_name": MODEL_FILES[model_key],
        "pca_config": pca_key,
        "prediction_time": elapsed,
    }


def predict_batch(
    model_key: str, pca_key: str, df: pd.DataFrame
) -> pd.DataFrame:
    """Predict every row of an uploaded CSV (patient.csv).

    The CSV must contain the 30 original feature columns (extra columns
    such as 'id' or 'diagnosis' are ignored if present).

    Returns:
        A copy of the input DataFrame with 'prediction', 'probability_benign'
        and 'probability_malignant' columns appended.
    """
    feature_names = get_feature_names()
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing}")

    raw = df[feature_names].apply(pd.to_numeric, errors="coerce")
    if raw.isna().any().any():
        raise ValueError("Uploaded CSV contains non-numeric or missing values.")

    X_transformed = _transform_features(raw, pca_key)
    model = load_artifact(_artifact_name(model_key, pca_key, "model"))

    preds = model.predict(X_transformed)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_transformed)
    else:
        scores = model.decision_function(X_transformed)
        proba_pos = 1 / (1 + np.exp(-scores))
        probas = np.column_stack([1 - proba_pos, proba_pos])

    result = df.copy()
    result["prediction"] = ["Malignant" if p == 0 else "Benign" for p in preds]
    result["probability_malignant"] = probas[:, 0].round(4)
    result["probability_benign"] = probas[:, 1].round(4)
    return result


def available_model_keys() -> List[str]:
    """Return the list of supported model keys, in display order."""
    return list(MODEL_FILES.keys())
