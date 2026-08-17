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
        "en": "PCA-based Dimensionality Reduction",
        "vi": "Giảm chiều dữ liệu bằng PCA",
    },
    "language_label": {"en": "Language", "vi": "Ngôn ngữ"},
    "nav_dashboard": {"en": "Dashboard", "vi": "Tổng quan"},
    "nav_prediction": {"en": "Prediction", "vi": "Dự đoán"},
    "nav_visualization": {"en": "Visualization", "vi": "Trực quan hóa"},
    "nav_about": {"en": "About", "vi": "Giới thiệu"},
    "sidebar_footer1": {
        "en": "Tech stack: Streamlit · Scikit-learn · Plotly",
        "vi": "Công nghệ: Streamlit · Scikit-learn · Plotly",
    },
    "sidebar_footer2": {
        "en": "The scaler and PCA transformers are pre-fitted — run "
        "`python train.py` to (re)fit.",
        "vi": "Bộ chuẩn hóa và PCA đã được huấn luyện sẵn — chạy `python train.py` "
        "để (huấn luyện lại).",
    },

    # --- Dashboard --------------------------------------------------------
    "dashboard_title": {
        "en": "🩺 Breast Cancer Classification Dashboard",
        "vi": "🩺 Bảng điều khiển Phân loại Ung thư vú",
    },
    "project_title": {
        "en": "Application of Dimensionality Reduction (PCA) for Breast "
        "Cancer Data Analysis",
        "vi": "Ứng dụng Giảm chiều Dữ liệu (PCA) trong Phân tích Dữ liệu "
        "Ung thư vú",
    },
    "project_objective": {
        "en": "Explore how PCA-based dimensionality reduction summarizes the "
        "30 original features of the breast cancer dataset into a small "
        "number of principal components while retaining most of the "
        "variance.",
        "vi": "Khám phá cách giảm chiều dữ liệu bằng PCA tóm tắt 30 đặc "
        "trưng gốc của bộ dữ liệu ung thư vú thành một số ít thành phần "
        "chính trong khi vẫn giữ lại phần lớn phương sai.",
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

    # --- Prediction (Dimensionality Reduction transform) ------------------
    "prediction_title": {
        "en": "🔬 Dimensionality Reduction (PCA) Transform",
        "vi": "🔬 Giảm chiều Dữ liệu (PCA)",
    },
    "prediction_page_caption": {
        "en": "Enter a patient's 30 features (or upload a CSV) to see how PCA "
        "reduces them to fewer principal components.",
        "vi": "Nhập 30 đặc trưng của bệnh nhân (hoặc tải lên CSV) để xem PCA "
        "giảm chúng xuống còn bao nhiêu thành phần chính.",
    },
    "pca_components_label": {"en": "PCA Components", "vi": "Số thành phần PCA"},
    "tab_manual": {"en": "📝 Manual Entry", "vi": "📝 Nhập thủ công"},
    "tab_upload": {"en": "📁 Upload CSV", "vi": "📁 Tải file CSV"},
    "manual_caption": {
        "en": "Every field is pre-filled with the dataset mean — adjust as needed.",
        "vi": "Mỗi ô đã được điền sẵn giá trị trung bình của bộ dữ liệu — có thể chỉnh sửa.",
    },
    "transform_button": {"en": "🔍 Reduce Dimensions", "vi": "🔍 Giảm chiều"},
    "components_header": {"en": "Principal Components", "vi": "Các Thành phần Chính"},
    "component_col": {"en": "Component", "vi": "Thành phần"},
    "component_value_col": {"en": "Value", "vi": "Giá trị"},
    "upload_caption": {
        "en": "Upload a CSV file containing the 30 original feature columns "
        "(e.g. exported from the dataset) to reduce the dimensionality of "
        "multiple patients at once.",
        "vi": "Tải lên file CSV chứa đủ 30 cột đặc trưng gốc (ví dụ xuất từ bộ "
        "dữ liệu) để giảm chiều nhiều bệnh nhân cùng lúc.",
    },
    "upload_prompt": {"en": "Upload patient.csv", "vi": "Tải lên patient.csv"},
    "loaded_rows": {"en": "Loaded **{n}** rows.", "vi": "Đã tải **{n}** dòng."},
    "transform_all_button": {"en": "🔍 Reduce All", "vi": "🔍 Giảm chiều tất cả"},
    "transformed_success": {
        "en": "Reduced dimensionality for {n} patients.",
        "vi": "Đã giảm chiều xong {n} bệnh nhân.",
    },
    "download_button": {
        "en": "⬇️ Download pca_transform_result.csv",
        "vi": "⬇️ Tải xuống pca_transform_result.csv",
    },
    "diagnosis_malignant": {"en": "Malignant", "vi": "Ác tính"},
    "diagnosis_benign": {"en": "Benign", "vi": "Lành tính"},

    # --- Metric cards ------------------------------------------------------
    "dimensionality_metric": {
        "en": "Dimensionality (before → after)", "vi": "Số chiều (trước → sau)",
    },
    "variance_retained_metric": {
        "en": "Variance Retained", "vi": "Phương sai Giữ lại",
    },
    "transform_time_metric": {"en": "Transform Time", "vi": "Thời gian giảm chiều"},

    # --- Patient form (feature groups) --------------------------------------
    "feature_group_mean": {"en": "Mean", "vi": "Trung bình"},
    "feature_group_se": {"en": "Standard Error", "vi": "Sai số Chuẩn"},
    "feature_group_worst": {"en": "Worst", "vi": "Giá trị Xấu nhất"},

    # --- Visualization ---------------------------------------------------
    "visualization_title": {
        "en": "📊 Data & Model Visualization", "vi": "📊 Trực quan hóa Dữ liệu & Mô hình",
    },
    "tab_pca_analysis": {"en": "PCA Analysis", "vi": "Phân tích PCA"},
    "tab_correlation": {"en": "Correlation Heatmap", "vi": "Bản đồ nhiệt Tương quan"},
    "tab_feature_distribution": {"en": "Feature Distribution", "vi": "Phân bố Đặc trưng"},
    "pca_scatter_selector_label": {
        "en": "PCA configuration for scatter plots",
        "vi": "Cấu hình PCA cho biểu đồ phân tán",
    },
    "pca_scatter_invariance_note": {
        "en": "Note: PC1/PC2/PC3 are mathematically identical across "
        "configurations (PCA components are nested) — only the "
        "dimensionality and variance retained above change.",
        "vi": "Lưu ý: PC1/PC2/PC3 giống hệt nhau giữa các cấu hình do bản "
        "chất lồng nhau của PCA — chỉ số chiều và % phương sai giữ lại "
        "ở trên là thay đổi theo lựa chọn.",
    },
    "pca_configuration_label": {"en": "PCA Configuration", "vi": "Cấu hình PCA"},
    "feature_selector_label": {"en": "Feature", "vi": "Đặc trưng"},

    # --- About ---------------------------------------------------------
    "about_title": {"en": "ℹ️ About This Project", "vi": "ℹ️ Giới thiệu Đề tài"},
    "about_project_header": {"en": "Project", "vi": "Đề tài"},
    "about_student_header": {"en": "Student", "vi": "Sinh viên"},
    "about_student_placeholder": {
        "en": "**Group 5** — 4 members: 26410326 Nguyễn Hữu Vinh, 26410266 Nguyễn Hoài Phú, 26410286 Nguyễn Ngọc Thanh, 26410267 Nguyễn Trọng Phúc",
        "vi": "**Nhóm 5** — 4 thành viên: 26410326 Nguyễn Hữu Vinh, 26410266 Nguyễn Hoài Phú, 26410286 Nguyễn Ngọc Thanh, 26410267 Nguyễn Trọng Phúc",
    },
    "about_lecturer_header": {"en": "Lecturer", "vi": "Giảng viên"},
    "about_lecturer_placeholder": {
        "en": "TS. Dương Việt Hằng | [elearning](https://elearning.hcmute.edu.vn/)",
        "vi": "TS. Dương Việt Hằng | [elearning](https://elearning.hcmute.edu.vn/)",
    },
    "about_dataset_header": {"en": "Dataset", "vi": "Bộ dữ liệu"},
    "about_dataset_text": {
        "en": "Breast Cancer Wisconsin (Diagnostic) Dataset — 569 samples, 30 "
        "numeric features, used here to demonstrate PCA-based "
        "dimensionality reduction.",
        "vi": "Bộ dữ liệu Breast Cancer Wisconsin (Diagnostic) — 569 mẫu, 30 "
        "đặc trưng số, dùng để minh họa việc giảm chiều dữ liệu bằng PCA.",
    },
    "about_github_header": {"en": "GitHub", "vi": "GitHub"},
    "about_github_placeholder": {
        "en": "[vinhkhanh79/Breast-Cancer-Classification](https://github.com/vinhkhanh79/Breast-Cancer-Classification)",
        "vi": "[vinhkhanh79/Breast-Cancer-Classification](https://github.com/vinhkhanh79/Breast-Cancer-Classification)",
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
