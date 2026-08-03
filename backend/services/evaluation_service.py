"""
evaluation_service.py
======================
Computes evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC,
Confusion Matrix, ROC curve points) for a trained model on the test set.

Used by train.py (to build results/tables/comparison_table.csv) and by
the Visualization / Model Comparison pages through backend/api.py.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from backend.utils.helpers import timed


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluate a fitted classifier on the test split.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, confusion
        matrix, ROC curve points (fpr/tpr) and prediction time (seconds
        for the whole test split, used for the Model Comparison table).
    """
    y_pred, predict_time = timed(model.predict, X_test)

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "prediction_time": float(predict_time),
    }
