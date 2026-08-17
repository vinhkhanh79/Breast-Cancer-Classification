"""
train.py
========
Run this script once (or whenever the dataset changes) BEFORE launching
the Streamlit web app:

    python train.py

Pipeline:
    1. Load dataset (data/breast_cancer.csv) + EDA
    2. Train/test split + StandardScaler
    3. Fit PCA for every configuration: No PCA, 10, 15, 20 components,
       95% and 99% explained variance
    4. Persist:
        - backend/models/scaler.pkl
        - backend/models/pca_full.pkl              (for explained-variance chart)
        - backend/models/pca_<config>.pkl           (per PCA configuration)
        - results/tables/pca_summary.csv            (n_components / variance retained per config)
        - results/tables/descriptive_statistics.csv
        - results/tables/class_distribution.csv

The web application (frontend/) never trains anything itself — it only
loads these artifacts through backend/api.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.preprocessing.preprocess import (
    PCA_TAGS,
    fit_full_pca,
    get_dataset_summary,
    load_dataset,
    resolve_pca_component_counts,
    split_and_scale,
)
from backend.services.training_service import transform_with_pca
from backend.utils.helpers import (
    RESULTS_TABLES_DIR,
    print_section,
    save_artifact,
)


def run_eda_and_save_tables() -> None:
    """Print EDA summary to console and persist descriptive tables."""
    print_section("1. DATA LOADING & EXPLORATORY DATA ANALYSIS (EDA)")
    X, y = load_dataset()
    summary = get_dataset_summary()

    print(f"Samples: {summary['n_samples']}, Features: {summary['n_features']}")
    print(f"Class distribution: {summary['class_distribution']}")
    print(f"Missing values: {summary['missing_values']}")

    desc = X.describe().T.reset_index().rename(columns={"index": "feature"})
    desc.to_csv(RESULTS_TABLES_DIR / "descriptive_statistics.csv", index=False)

    class_dist_df = pd.DataFrame(
        {
            "class": list(summary["class_distribution"].keys()),
            "count": list(summary["class_distribution"].values()),
        }
    )
    class_dist_df["percentage"] = 100 * class_dist_df["count"] / summary["n_samples"]
    class_dist_df.to_csv(RESULTS_TABLES_DIR / "class_distribution.csv", index=False)


def main() -> None:
    run_eda_and_save_tables()

    print_section("2. TRAIN / TEST SPLIT + STANDARD SCALING")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale()
    save_artifact(scaler, "scaler.pkl")
    print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

    print_section("3. FITTING FULL PCA (for Explained Variance chart)")
    pca_full = fit_full_pca(X_train_scaled)
    save_artifact(pca_full, "pca_full.pkl")

    n_components_map = resolve_pca_component_counts(X_train_scaled)
    print("Resolved PCA component counts:")
    for key, n in n_components_map.items():
        print(f"  - {key}: {n} components ({PCA_TAGS[key]})")

    n_features_original = X_train_scaled.shape[1]
    summary_rows = []

    print_section("4. FITTING PCA FOR EVERY CONFIGURATION")
    for pca_key, n_components in n_components_map.items():
        label = PCA_TAGS[pca_key]
        print(f"  - Fitting {label} ...")
        _, _, pca_obj = transform_with_pca(X_train_scaled, X_test_scaled, n_components)
        save_artifact(pca_obj, f"pca_{pca_key}.pkl")

        cumulative_variance = float(np.sum(pca_obj.explained_variance_ratio_))
        summary_rows.append(
            {
                "pca_config": pca_key,
                "pca_label": label,
                "n_features_before_pca": n_features_original,
                "n_components": n_components,
                "cumulative_variance_explained": round(cumulative_variance, 4),
            }
        )

    print_section("5. SAVING PCA SUMMARY TABLE")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_TABLES_DIR / "pca_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    print_section("DONE")
    print("Scaler and every PCA transformer were saved to backend/models/.")
    print("Run the web app with:  streamlit run frontend/app.py")


if __name__ == "__main__":
    main()
