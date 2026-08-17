"""
preprocess.py
=============
Data-layer responsibilities for the breast cancer classification system:
- Loading the dataset (data/breast_cancer.csv)
- Basic EDA (shape, class distribution, descriptive statistics)
- Train/test split
- StandardScaler fitting/transforming
- Resolving PCA component counts for every supported configuration
  (10, 15, 20 components, and 95% / 99% explained variance)

This module contains NO model training and NO Streamlit code — it is
pure data preparation, reusable by both train.py and the backend services.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.utils.helpers import DATASET_PATH

TARGET_NAMES: List[str] = ["malignant", "benign"]
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# Human-readable PCA configuration tags used throughout the whole system
# (file names, dropdown labels, results tables, chart titles).
PCA_TAGS = {
    "no_pca": "No PCA",
    "pca_10": "PCA (10 components)",
    "pca_15": "PCA (15 components)",
    "pca_20": "PCA (20 components)",
    "pca_95": "PCA (95% variance)",
    "pca_99": "PCA (99% variance)",
}


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load features (X) and target (y) from data/breast_cancer.csv.

    The raw dataset uses ``diagnosis`` in {"M", "B"}; it is mapped to
    0 = malignant, 1 = benign.

    Returns:
        X: DataFrame with the 30 numeric features.
        y: Series with the binary target.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Please place breast_cancer.csv inside the data/ folder."
        )

    df = pd.read_csv(DATASET_PATH)

    if "diagnosis" not in df.columns:
        raise ValueError("breast_cancer.csv must contain a 'diagnosis' column.")

    mapping = {"M": 0, "MALIGNANT": 0, "B": 1, "BENIGN": 1}
    diagnosis = df["diagnosis"].astype(str).str.strip().str.upper()
    y = diagnosis.map(mapping)

    if y.isna().any():
        unknown = sorted(set(diagnosis[y.isna()]))
        raise ValueError(f"Unsupported diagnosis values found: {unknown}")

    feature_cols = [c for c in df.columns if c not in {"id", "diagnosis"}]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        raise ValueError("Non-numeric values found in feature columns.")

    return X.reset_index(drop=True), y.rename("target").astype(int).reset_index(drop=True)


def get_feature_names() -> List[str]:
    """Return the ordered list of the 30 original feature names."""
    X, _ = load_dataset()
    return list(X.columns)


def get_feature_means() -> Dict[str, float]:
    """Return mean value of every feature, used to pre-fill the Prediction form."""
    X, _ = load_dataset()
    return X.mean().round(4).to_dict()


def get_dataset_summary() -> Dict[str, object]:
    """Return dataset-level info shown on the Dashboard page."""
    X, y = load_dataset()
    class_counts = y.value_counts().sort_index()
    class_distribution = {
        TARGET_NAMES[cls]: int(count) for cls, count in class_counts.items()
    }
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "target_names": TARGET_NAMES,
        "class_distribution": class_distribution,
        "missing_values": int(X.isna().sum().sum()),
    }


def split_and_scale() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split the dataset into train/test and fit a StandardScaler on the train split.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, fitted_scaler
    """
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.to_numpy(),
        y_test.to_numpy(),
        scaler,
    )


def resolve_pca_component_counts(X_train_scaled: np.ndarray) -> Dict[str, int]:
    """Determine how many PCA components each configuration corresponds to.

    Fits a full-dimensional PCA once to read the cumulative explained
    variance curve, then derives the component count needed for the
    95% and 99% variance targets. Fixed configs (10/15/20) are capped
    at the number of original features.

    Returns:
        Dict mapping config key ("pca_10", "pca_95", ...) -> n_components.
    """
    n_features = X_train_scaled.shape[1]
    pca_full = PCA(n_components=None, random_state=RANDOM_STATE)
    pca_full.fit(X_train_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)

    n_95 = int(np.argmax(cumulative >= 0.95) + 1)
    n_99 = int(np.argmax(cumulative >= 0.99) + 1)

    return {
        "pca_10": min(10, n_features),
        "pca_15": min(15, n_features),
        "pca_20": min(20, n_features),
        "pca_95": n_95,
        "pca_99": n_99,
    }


def fit_full_pca(X_train_scaled: np.ndarray) -> PCA:
    """Fit a full-dimensional PCA (used for the Explained Variance chart)."""
    pca_full = PCA(n_components=None, random_state=RANDOM_STATE)
    pca_full.fit(X_train_scaled)
    return pca_full
