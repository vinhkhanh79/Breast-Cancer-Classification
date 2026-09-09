"""
training_service.py
====================
Responsible for training the single classification model
(Logistic Regression) for every PCA configuration
("no_pca", "pca_10", "pca_15", "pca_20", "pca_95", "pca_99").

Called only from train.py — never from the Streamlit frontend directly.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from backend.preprocessing.preprocess import RANDOM_STATE
from backend.utils.helpers import print_section, timed

MODEL_FACTORIES: Dict[str, Any] = {
    "logistic": lambda: LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
}


def transform_with_pca(
    X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, n_components: int
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA on the (scaled) train split and transform both splits.

    PCA is fit on train data only to avoid data leakage.
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca, pca


def train_one_model(
    model_key: str, X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[Any, float]:
    """Instantiate and fit a single model, returning it with the training time (s)."""
    factory = MODEL_FACTORIES[model_key]
    model, elapsed = timed(lambda: factory().fit(X_train, y_train))
    return model, elapsed


def train_all_models_for_config(
    X_train: np.ndarray, y_train: np.ndarray
) -> Dict[str, Tuple[Any, float]]:
    """Train the single Logistic Regression model on the given data.

    Args:
        X_train: training features (already scaled, and PCA-transformed
            unless the config is "no_pca").
        y_train: training labels.

    Returns:
        Dict mapping model_key -> (fitted_model, training_time_seconds)
    """
    print_section(f"Training model on {X_train.shape[1]}-dimensional input")
    results: Dict[str, Tuple[Any, float]] = {}
    for model_key in MODEL_FACTORIES:
        model, elapsed = train_one_model(model_key, X_train, y_train)
        print(f"  - {model_key:15s} trained in {elapsed * 1000:.2f} ms")
        results[model_key] = (model, elapsed)
    return results
