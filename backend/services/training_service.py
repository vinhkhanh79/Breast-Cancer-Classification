"""
training_service.py
====================
Responsible for fitting the PCA (Dimensionality Reduction) transformer
for every configuration ("pca_10", "pca_15", "pca_20", "pca_95", "pca_99").

Called only from train.py — never from the Streamlit frontend directly.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA

from backend.preprocessing.preprocess import RANDOM_STATE


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
