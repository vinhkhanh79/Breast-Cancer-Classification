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
    4. Train the single Logistic Regression model for EVERY PCA configuration
    5. Evaluate every (model, PCA config) pair on the test set
    6. Persist:
        - backend/models/scaler.pkl
        - backend/models/pca_full.pkl              (for explained-variance chart)
        - backend/models/pca_<config>.pkl           (per PCA configuration)
        - backend/models/<model>_<config>.pkl       (per model x PCA configuration)
        - results/metrics/<model>_<config>.json     (per-pair metrics)
        - results/tables/comparison_table.csv       (all PCA variants for the single model)
        - results/tables/descriptive_statistics.csv
        - results/tables/class_distribution.csv

The web application (frontend/) never trains anything itself — it only
loads these artifacts through backend/api.py.
"""

from __future__ import annotations

import json

import pandas as pd

from backend.preprocessing.preprocess import (
    PCA_TAGS,
    fit_full_pca,
    get_dataset_summary,
    load_dataset,
    resolve_pca_component_counts,
    split_and_scale,
)
from backend.services.evaluation_service import evaluate_model
from backend.services.training_service import (
    train_all_models_for_config,
    transform_with_pca,
)
from backend.utils.helpers import (
    RESULTS_METRICS_DIR,
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

    comparison_rows = []
    n_features_original = X_train_scaled.shape[1]

    # ---- Config: No PCA -----------------------------------------------
    print_section(f"4. TRAINING — {PCA_TAGS['no_pca']}")
    trained = train_all_models_for_config(X_train_scaled, y_train)
    for model_key, (model, train_time) in trained.items():
        save_artifact(model, f"{model_key}_no_pca.pkl")
        metrics = evaluate_model(model, X_test_scaled, y_test)
        metrics["training_time"] = train_time
        _persist_metrics_and_row(
            comparison_rows, model_key, "no_pca", metrics,
            n_features_original, n_features_original,
        )

    # ---- Configs: PCA_10 / PCA_15 / PCA_20 / PCA_95 / PCA_99 -----------
    for pca_key, n_components in n_components_map.items():
        label = PCA_TAGS[pca_key]
        print_section(f"4. TRAINING — {label}")
        X_train_pca, X_test_pca, pca_obj = transform_with_pca(
            X_train_scaled, X_test_scaled, n_components
        )
        save_artifact(pca_obj, f"pca_{pca_key}.pkl")

        trained = train_all_models_for_config(X_train_pca, y_train)
        for model_key, (model, train_time) in trained.items():
            save_artifact(model, f"{model_key}_{pca_key}.pkl")
            metrics = evaluate_model(model, X_test_pca, y_test)
            metrics["training_time"] = train_time
            _persist_metrics_and_row(
                comparison_rows, model_key, pca_key, metrics,
                n_features_original, n_components,
            )

    print_section("5. SAVING MODEL COMPARISON TABLE")
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULTS_TABLES_DIR / "comparison_table.csv", index=False)
    print(comparison_df.to_string(index=False))

    print_section("DONE")
    print("All models, PCA transformers and the scaler were saved to backend/models/.")
    print("Run the web app with:  streamlit run frontend/app.py")


def _persist_metrics_and_row(
    comparison_rows: list,
    model_key: str,
    pca_key: str,
    metrics: dict,
    n_features_before: int,
    n_features_after: int,
) -> None:
    """Save a per-pair metrics JSON file and append a summary row for the
    comparison table."""
    with open(RESULTS_METRICS_DIR / f"{model_key}_{pca_key}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    from backend.services.prediction_service import MODEL_FILES
    from backend.preprocessing.preprocess import PCA_TAGS

    comparison_rows.append(
        {
            "model": MODEL_FILES[model_key],
            "model_key": model_key,
            "pca_config": pca_key,
            "pca_label": PCA_TAGS[pca_key],
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "roc_auc": round(metrics["roc_auc"], 4),
            "training_time": round(metrics["training_time"], 6),
            "prediction_time": round(metrics["prediction_time"], 6),
            "n_features_before_pca": n_features_before,
            "n_features_after_pca": n_features_after,
        }
    )


if __name__ == "__main__":
    main()
