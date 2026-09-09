"""
app.py
======
Main Streamlit entry point. Responsible ONLY for page configuration,
theme, and routing between pages based on the sidebar selection.

No Machine Learning logic lives here — every page module calls
backend/api.py for data and predictions. The currently selected
language ("en" or "vi") is stored in st.session_state and threaded
into every page render function.

Run with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the project root importable (so `backend.*` and `frontend.*` resolve
# regardless of the working directory `streamlit run` is invoked from).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils.i18n import t  # noqa: E402
from frontend.components.sidebar import render_sidebar  # noqa: E402
from frontend.pages import comparison, dashboard, prediction, visualization  # noqa: E402

st.set_page_config(
    page_title="Breast Cancer PCA Classification",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Blue & White theme touches (Streamlit's native theming is
# configured in .streamlit/config.toml; these tweaks add a few extra accents).
st.markdown(
    """
    <style>
        .stButton>button { border-radius: 8px; font-weight: 600; }
        section[data-testid="stSidebar"] { background-color: #f4f8fc; }
        h1, h2, h3 { color: #1f4e8c; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_about(lang: str) -> None:
    """Render the About page (project / student / lecturer / dataset / GitHub info)."""
    st.title(t("about_title", lang))
    st.markdown(
        f"""
        ### {t("about_project_header", lang)}
        **{t("project_title", lang)}**

        ### {t("about_student_header", lang)}
        {t("about_student_placeholder", lang)}

        ### {t("about_lecturer_header", lang)}
        {t("about_lecturer_placeholder", lang)}

        ### {t("about_dataset_header", lang)}
        {t("about_dataset_text", lang)}

        ### {t("about_github_header", lang)}
        {t("about_github_placeholder", lang)}

        ### {t("about_architecture_header", lang)}
        {t("about_architecture_text", lang)}
        """
    )


PAGE_RENDERERS = {
    "nav_dashboard": dashboard.render,
    "nav_prediction": prediction.render,
    "nav_visualization": visualization.render,
    "nav_comparison": comparison.render,
    "nav_about": render_about,
}


def main() -> None:
    selected_page_key, lang = render_sidebar()
    PAGE_RENDERERS[selected_page_key](lang)


if __name__ == "__main__":
    main()
