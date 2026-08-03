"""
i18n.py
=======
Single source of truth for all user-facing text in English and Vietnamese.

Used by:
- frontend/* (page titles, labels, buttons, table headers)
- backend/visualization/charts.py (Plotly chart titles, axis labels, legends)

Kept here (backend/utils) rather than under frontend/ because chart text
is produced by the backend visualization layer, which must not depend on
Streamlit — this module has zero UI framework dependencies.

Usage:
    from backend.utils.i18n import t
    t("nav_dashboard", lang)                  # -> "Dashboard" / "Tổng quan"
    t("loaded_rows", lang, n=42)               # -> "Loaded **42** rows."
"""

from __future__ import annotations

SUPPORTED_LANGUAGES = {"en": "English", "vi": "Tiếng Việt"}
DEFAULT_LANGUAGE = "en"

_TEXTS = {
    # --- App / Sidebar --------------------------------------------------
    "app_name": {"en": "🩺 Breast Cancer AI", "vi": "🩺 AI Ung thư vú"},
    "sidebar_caption": {
        "en": "PCA-based Dimensionality Reduction for Classification",
        "vi": "Giảm chiều dữ liệu bằng PCA phục vụ Phân loại",
    },
    "language_label": {"en": "Language", "vi": "Ngôn ngữ"},
    "nav_dashboard": {"en": "Dashboard", "vi": "Tổng quan"},
    "nav_prediction": {"en": "Prediction", "vi": "Dự đoán"},
    "nav_visualization": {"en": "Visualization", "vi": "Trực quan hóa"},
    "nav_comparison": {"en": "Model Comparison", "vi": "So sánh mô hình"},
    "nav_about": {"en": "About", "vi": "Giới thiệu"},
    "sidebar_footer1": {
        "en": "Tech stack: Streamlit · Scikit-learn · Plotly",
        "vi": "Công nghệ: Streamlit · Scikit-learn · Plotly",
    },
    "sidebar_footer2": {
        "en": "Models are pre-trained — run `python train.py` to (re)train.",
        "vi": "Mô hình đã được huấn luyện sẵn — chạy `python train.py` để (huấn luyện lại).",
    },

    # --- Dashboard --------------------------------------------------------
    "dashboard_title": {
        "en": "🩺 Breast Cancer Classification Dashboard",
        "vi": "🩺 Bảng điều khiển Phân loại Ung thư vú",
    },
    "project_title": {
        "en": "Application of Dimensionality Reduction for Breast Cancer "
        "Classification Using Machine Learning",
        "vi": "Ứng dụng Giảm chiều Dữ liệu trong Phân loại Ung thư vú "
        "bằng Học máy",
    },
    "project_objective": {
        "en": "Compare the impact of PCA-based dimensionality reduction on the "
        "performance of several classification models (Logistic Regression, "
        "SVM, Random Forest, KNN) applied to breast cancer diagnosis.",
        "vi": "So sánh tác động của việc giảm chiều dữ liệu bằng PCA đến hiệu "
        "năng của nhiều mô hình phân loại (Logistic Regression, SVM, Random "
        "Forest, KNN) áp dụng cho chẩn đoán ung thư vú.",
    },
    "dataset_name": {
        "en": "Breast Cancer Wisconsin (Diagnostic) Dataset",
        "vi": "Bộ dữ liệu Breast Cancer Wisconsin (Diagnostic)",
    },
    "project_objective_expander": {"en": "📌 Project Objective", "vi": "📌 Mục tiêu Đề tài"},
    "dataset_overview_header": {"en": "Dataset Overview", "vi": "Tổng quan Dữ liệu"},
    "dataset_label": {"en": "**Dataset:**", "vi": "**Bộ dữ liệu:**"},
    "class_distribution_header": {"en": "Class Distribution", "vi": "Phân bố Lớp"},
    "feature_names_header": {"en": "Feature Names", "vi": "Danh sách Đặc trưng"},
    "missing_values_info": {
        "en": "Missing values in the dataset: **{n}** (0 expected for this clean dataset).",
        "vi": "Số giá trị thiếu trong bộ dữ liệu: **{n}** (dự kiến bằng 0 với bộ dữ liệu sạch này).",
    },
    "metric_total_samples": {"en": "Total Samples", "vi": "Tổng số mẫu"},
    "metric_num_features": {"en": "Number of Features", "vi": "Số lượng Đặc trưng"},
    "metric_num_classes": {"en": "Number of Classes", "vi": "Số lượng Lớp"},

    # --- Prediction ---------------------------------------------------
    "prediction_title": {"en": "🔬 Breast Cancer Prediction", "vi": "🔬 Dự đoán Ung thư vú"},
    "model_label": {"en": "Model", "vi": "Mô hình"},
    "pca_components_label": {"en": "PCA Components", "vi": "Số thành phần PCA"},
    "tab_manual": {"en": "📝 Manual Entry", "vi": "📝 Nhập thủ công"},
    "tab_upload": {"en": "📁 Upload CSV", "vi": "📁 Tải file CSV"},
    "manual_caption": {
        "en": "Every field is pre-filled with the dataset mean — adjust as needed.",
        "vi": "Mỗi ô đã được điền sẵn giá trị trung bình của bộ dữ liệu — có thể chỉnh sửa.",
    },
    "predict_button": {"en": "🔍 Predict", "vi": "🔍 Dự đoán"},
    "probability_breakdown_header": {"en": "Probability Breakdown", "vi": "Chi tiết Xác suất"},
    "prob_table_class_col": {"en": "Class", "vi": "Lớp"},
    "prob_table_prob_col": {"en": "Probability", "vi": "Xác suất"},
    "upload_caption": {
        "en": "Upload a CSV file containing the 30 original feature columns "
        "(e.g. exported from the dataset) to predict multiple patients at once.",
        "vi": "Tải lên file CSV chứa đủ 30 cột đặc trưng gốc (ví dụ xuất từ bộ "
        "dữ liệu) để dự đoán nhiều bệnh nhân cùng lúc.",
    },
    "upload_prompt": {"en": "Upload patient.csv", "vi": "Tải lên patient.csv"},
    "loaded_rows": {"en": "Loaded **{n}** rows.", "vi": "Đã tải **{n}** dòng."},
    "predict_all_button": {"en": "🔍 Predict All", "vi": "🔍 Dự đoán tất cả"},
    "predicted_success": {
        "en": "Predicted {n} patients.", "vi": "Đã dự đoán xong {n} bệnh nhân.",
    },
    "download_button": {
        "en": "⬇️ Download prediction_result.csv",
        "vi": "⬇️ Tải xuống prediction_result.csv",
    },
    "diagnosis_malignant": {"en": "Malignant", "vi": "Ác tính"},
    "diagnosis_benign": {"en": "Benign", "vi": "Lành tính"},
    "review_delta": {"en": "⚠️ Review", "vi": "⚠️ Cần xem xét"},
    "normal_delta": {"en": "✅ Normal", "vi": "✅ Bình thường"},

    # --- Metric cards ------------------------------------------------------
    "prediction_metric_label": {"en": "Prediction", "vi": "Kết quả dự đoán"},
    "prob_benign_metric": {"en": "Probability (Benign)", "vi": "Xác suất (Lành tính)"},
    "prediction_time_metric": {"en": "Prediction Time", "vi": "Thời gian dự đoán"},
    "best_accuracy_metric": {"en": "Best Accuracy", "vi": "Độ chính xác tốt nhất"},
    "best_f1_metric": {"en": "Best F1-Score", "vi": "F1-Score tốt nhất"},

    # --- Patient form (feature groups) --------------------------------------
    "feature_group_mean": {"en": "Mean", "vi": "Trung bình"},
    "feature_group_se": {"en": "Standard Error", "vi": "Sai số Chuẩn"},
    "feature_group_worst": {"en": "Worst", "vi": "Giá trị Xấu nhất"},

    # --- Visualization ---------------------------------------------------
    "visualization_title": {
        "en": "📊 Data & Model Visualization", "vi": "📊 Trực quan hóa Dữ liệu & Mô hình",
    },
    "tab_pca_analysis": {"en": "PCA Analysis", "vi": "Phân tích PCA"},
    "tab_roc_cm": {
        "en": "ROC & Confusion Matrix", "vi": "Đường cong ROC & Ma trận nhầm lẫn",
    },
    "tab_correlation": {"en": "Correlation Heatmap", "vi": "Bản đồ nhiệt Tương quan"},
    "tab_feature_distribution": {"en": "Feature Distribution", "vi": "Phân bố Đặc trưng"},
    "pca_scatter_selector_label": {
        "en": "PCA configuration for scatter plots",
        "vi": "Cấu hình PCA cho biểu đồ phân tán",
    },
    "pca_configuration_label": {"en": "PCA Configuration", "vi": "Cấu hình PCA"},
    "feature_selector_label": {"en": "Feature", "vi": "Đặc trưng"},

    # --- Model Comparison --------------------------------------------------
    "comparison_title": {"en": "⚖️ Model Comparison", "vi": "⚖️ So sánh Mô hình"},
    "full_table_header": {"en": "Full Comparison Table", "vi": "Bảng So sánh Đầy đủ"},
    "bar_chart_header": {"en": "Bar Chart", "vi": "Biểu đồ Cột"},
    "metric_selector_label": {"en": "Metric", "vi": "Chỉ số"},
    "radar_chart_header": {"en": "Radar Chart", "vi": "Biểu đồ Radar"},
    "col_model": {"en": "Model", "vi": "Mô hình"},
    "col_pca_config": {"en": "PCA Config", "vi": "Cấu hình PCA"},
    "col_accuracy": {"en": "Accuracy", "vi": "Độ chính xác"},
    "col_precision": {"en": "Precision", "vi": "Độ chuẩn xác"},
    "col_recall": {"en": "Recall", "vi": "Độ bao phủ"},
    "col_f1": {"en": "F1", "vi": "F1"},
    "col_roc_auc": {"en": "ROC AUC", "vi": "ROC AUC"},
    "col_training_time": {"en": "Training Time (s)", "vi": "Thời gian huấn luyện (s)"},
    "col_prediction_time": {"en": "Prediction Time (s)", "vi": "Thời gian dự đoán (s)"},
    "col_dim_before": {"en": "Dim. Before PCA", "vi": "Số chiều trước PCA"},
    "col_dim_after": {"en": "Dim. After PCA", "vi": "Số chiều sau PCA"},

    # --- About ---------------------------------------------------------
    "about_title": {"en": "ℹ️ About This Project", "vi": "ℹ️ Giới thiệu Đề tài"},
    "about_project_header": {"en": "Project", "vi": "Đề tài"},
    "about_student_header": {"en": "Student", "vi": "Sinh viên"},
    "about_student_placeholder": {
        "en": "_Add your name / student ID here._", "vi": "_Điền tên / MSSV của bạn tại đây._",
    },
    "about_lecturer_header": {"en": "Lecturer", "vi": "Giảng viên"},
    "about_lecturer_placeholder": {
        "en": "_Add supervising lecturer's name here._",
        "vi": "_Điền tên giảng viên hướng dẫn tại đây._",
    },
    "about_dataset_header": {"en": "Dataset", "vi": "Bộ dữ liệu"},
    "about_dataset_text": {
        "en": "Breast Cancer Wisconsin (Diagnostic) Dataset — 569 samples, 30 "
        "numeric features, binary diagnosis (Malignant / Benign).",
        "vi": "Bộ dữ liệu Breast Cancer Wisconsin (Diagnostic) — 569 mẫu, 30 "
        "đặc trưng số, chẩn đoán nhị phân (Ác tính / Lành tính).",
    },
    "about_github_header": {"en": "GitHub", "vi": "GitHub"},
    "about_github_placeholder": {
        "en": "_Add your repository link here._", "vi": "_Điền liên kết repository tại đây._",
    },
    "about_architecture_header": {"en": "Architecture", "vi": "Kiến trúc"},
    "about_architecture_text": {
        "en": "This application follows a Clean Architecture with a strict "
        "separation between the Frontend (Streamlit UI only) and the "
        "Backend (all Machine Learning logic), communicating through a "
        "single facade module: `backend/api.py`.",
        "vi": "Ứng dụng tuân theo kiến trúc Clean Architecture với sự tách "
        "biệt rõ ràng giữa Frontend (chỉ giao diện Streamlit) và Backend "
        "(toàn bộ logic Học máy), giao tiếp qua một module facade duy "
        "nhất: `backend/api.py`.",
    },

    # --- Charts (backend/visualization/charts.py) ---------------------
    "chart_class_distribution": {"en": "Class Distribution", "vi": "Phân bố Lớp"},
    "axis_diagnosis": {"en": "Diagnosis", "vi": "Chẩn đoán"},
    "axis_num_samples": {"en": "Number of Samples", "vi": "Số lượng mẫu"},
    "chart_correlation_heatmap": {
        "en": "Correlation Heatmap (Top {n} Features by Variance)",
        "vi": "Bản đồ nhiệt Tương quan (Top {n} Đặc trưng theo Phương sai)",
    },
    "chart_feature_distribution": {
        "en": "Feature Distribution: {feature}", "vi": "Phân bố Đặc trưng: {feature}",
    },
    "chart_explained_variance": {
        "en": "Explained Variance by Principal Component",
        "vi": "Phương sai Giải thích theo Thành phần Chính",
    },
    "legend_explained_variance_ratio": {
        "en": "Explained Variance Ratio", "vi": "Tỷ lệ Phương sai Giải thích",
    },
    "legend_cumulative_variance": {
        "en": "Cumulative Explained Variance", "vi": "Phương sai Giải thích Tích lũy",
    },
    "axis_principal_component": {"en": "Principal Component", "vi": "Thành phần Chính"},
    "axis_explained_variance_ratio": {
        "en": "Explained Variance Ratio", "vi": "Tỷ lệ Phương sai Giải thích",
    },
    "chart_pca_scatter_2d": {"en": "PCA Scatter Plot (2D)", "vi": "Biểu đồ Phân tán PCA (2D)"},
    "chart_pca_scatter_3d": {"en": "PCA Scatter Plot (3D)", "vi": "Biểu đồ Phân tán PCA (3D)"},
    "chart_roc_curve": {"en": "ROC Curve", "vi": "Đường cong ROC"},
    "axis_false_positive_rate": {"en": "False Positive Rate", "vi": "Tỷ lệ Dương tính giả"},
    "axis_true_positive_rate": {"en": "True Positive Rate", "vi": "Tỷ lệ Dương tính thật"},
    "legend_random_classifier": {"en": "Random Classifier", "vi": "Bộ phân loại Ngẫu nhiên"},
    "chart_confusion_matrix": {"en": "Confusion Matrix", "vi": "Ma trận Nhầm lẫn"},
    "axis_predicted": {"en": "Predicted", "vi": "Dự đoán"},
    "axis_actual": {"en": "Actual", "vi": "Thực tế"},
    "axis_count": {"en": "Count", "vi": "Số lượng"},
    "chart_comparison_bar": {
        "en": "Model Comparison — {metric}", "vi": "So sánh Mô hình — {metric}",
    },
    "chart_comparison_radar": {
        "en": "Model Comparison Radar — {pca_config}",
        "vi": "Biểu đồ Radar So sánh Mô hình — {pca_config}",
    },
}


def t(key: str, lang: str = DEFAULT_LANGUAGE, **kwargs: object) -> str:
    """Translate a UI/chart text key into the requested language.

    Falls back to English if the language or the key is missing, and
    falls back to the raw key (so missing translations are obvious
    during development rather than crashing the app).
    """
    entry = _TEXTS.get(key)
    if entry is None:
        return key
    text = entry.get(lang) or entry.get(DEFAULT_LANGUAGE) or key
    return text.format(**kwargs) if kwargs else text
