"""
sidebar.py
==========
Renders the navigation sidebar, including the language selector
(EN/VI) placed above the navigation menu. Pure UI component — no ML
logic; translated text comes from backend/utils/i18n.py.
"""

from __future__ import annotations

from typing import Tuple

import streamlit as st

from backend import api
from backend.utils.i18n import DEFAULT_LANGUAGE, t

PAGE_KEYS = [
    "nav_dashboard",
    "nav_prediction",
    "nav_visualization",
    "nav_about",
]


def render_sidebar() -> Tuple[str, str]:
    """Render the sidebar and return (selected_page_key, lang).

    selected_page_key is one of the PAGE_KEYS values above (language
    independent), so routing in app.py never depends on display text.
    """
    if "lang" not in st.session_state:
        st.session_state["lang"] = DEFAULT_LANGUAGE

    languages = api.get_supported_languages()

    with st.sidebar:
        lang_code = st.selectbox(
            t("language_label", st.session_state["lang"]),
            options=list(languages.keys()),
            format_func=lambda code: languages[code],
            index=list(languages.keys()).index(st.session_state["lang"]),
            key="language_selector",
        )
        st.session_state["lang"] = lang_code
        lang = lang_code

        st.markdown(
            f"<h2 style='color:#1f4e8c;'>{t('app_name', lang)}</h2>",
            unsafe_allow_html=True,
        )
        st.caption(t("sidebar_caption", lang))
        st.divider()

        selection = st.radio(
            "Navigation",
            PAGE_KEYS,
            format_func=lambda key: t(key, lang),
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(t("sidebar_footer1", lang))
        st.caption(t("sidebar_footer2", lang))

    return selection, lang
