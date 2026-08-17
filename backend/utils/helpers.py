"""
helpers.py
==========
Shared low-level utility functions used across the backend layer:
- Path resolution (data / models / results directories)
- Joblib save / load wrappers with clear error messages
- Simple execution timer used for Training Time / Prediction Time metrics

No business logic lives here — only generic, reusable helpers.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Tuple

import joblib

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "backend" / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
RESULTS_FIGURES_DIR: Path = RESULTS_DIR / "figures"
RESULTS_METRICS_DIR: Path = RESULTS_DIR / "metrics"
RESULTS_TABLES_DIR: Path = RESULTS_DIR / "tables"
DATASET_PATH: Path = DATA_DIR / "breast_cancer.csv"

for directory in (MODELS_DIR, RESULTS_FIGURES_DIR, RESULTS_METRICS_DIR, RESULTS_TABLES_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Joblib persistence helpers
# ---------------------------------------------------------------------------

def save_artifact(obj: Any, filename: str) -> Path:
    """Save any picklable object (model, scaler, PCA) into backend/models/.

    Args:
        obj: object to persist (e.g. a fitted estimator).
        filename: file name, e.g. ``"logistic.pkl"``.

    Returns:
        Full path where the artifact was written.
    """
    path = MODELS_DIR / filename
    joblib.dump(obj, path)
    return path


def load_artifact(filename: str) -> Any:
    """Load a previously saved artifact from backend/models/.

    Args:
        filename: file name, e.g. ``"scaler.pkl"``.

    Raises:
        FileNotFoundError: if the artifact does not exist. This normally
            means ``python train.py`` has not been run yet.
    """
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact '{filename}' not found in {MODELS_DIR}. "
            "Please run 'python train.py' first to train and save the models."
        )
    return joblib.load(path)


def artifact_exists(filename: str) -> bool:
    """Check whether a given model artifact has already been trained."""
    return (MODELS_DIR / filename).exists()


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def timed(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Run ``func`` and return ``(result, elapsed_seconds)``.

    Used to measure Training Time and Prediction Time for the
    Model Comparison and Prediction pages.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def print_section(title: str) -> None:
    """Print a formatted section header to the console (used by train.py)."""
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")
