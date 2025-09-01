from __future__ import annotations

from pathlib import Path
import streamlit as st

_ASSETS_DIR = Path("assets")
_CSS_PATH = _ASSETS_DIR / "styles.css"
_LOGO_PATHS = [
    _ASSETS_DIR / "logo.png",
    _ASSETS_DIR / "logo.jpg",
    _ASSETS_DIR / "logo.svg",
    _ASSETS_DIR / "favicon.png",
]

_DEFAULT_TITLE = "The Video Editing Mistake Checker"

def _first_existing_logo() -> str | None:
    for p in _LOGO_PATHS:
        if p.exists():
            return str(p)
    return None

def apply_base_theme(
    page_title: str = "Mistake Checker",
    page_icon: str | None = None,
):
    # Prefer an image logo if present; otherwise use an emoji icon.
    if page_icon is None:
        page_icon = _first_existing_logo() or "🎬"

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load base CSS
    if _CSS_PATH.exists():
        css = _CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # Hide default Streamlit chrome (non-destructive)
    st.markdown(
        """
        <style>
        /* Hide the Streamlit menu and footer for a cleaner look */
        [data-testid="stToolbar"] { display: none !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_runtime_theme_controls():
    with st.sidebar.expander("Appearance", expanded=False):
        col1, col2 = st.columns([1, 1])
        theme_mode = col1.selectbox("Theme", ["System", "Light", "Dark"], index=2)
        brand = col2.color_picker("Brand", value="#7C3AED")

        # App title (hero heading)
        default_title = st.session_state.get("hero_title", _DEFAULT_TITLE)
        hero_title = st.text_input("App title", value=default_title, help="Controls the main page title at the top of the app.")
        st.session_state["hero_title"] = hero_title

        # Advanced: tweak page density and content width
        density = st.select_slider("Density", options=["Cozy", "Comfortable", "Compact"], value="Comfortable")
        max_width = st.slider("Content width (px)", min_value=960, max_value=1400, value=1180, step=10)

        # Theme color presets
        light_bg = "#FFFFFF"
        light_panel = "#F7F8FA"
        light_text = "#111827"

        dark_bg = "#0B0F19"
        dark_panel = "#111827"
        dark_text = "#E5E7EB"

        if theme_mode == "Light":
            bg, panel, text = light_bg, light_panel, light_text
        elif theme_mode == "Dark":
            bg, panel, text = dark_bg, dark_panel, dark_text
        else:
            # "System" — keep config defaults but still set brand variable
            bg, panel, text = "inherit", "inherit", "inherit"

        padding_map = {
            "Cozy": ("1.75rem", "1.0rem"),
            "Comfortable": ("1.25rem", "0.75rem"),
            "Compact": ("0.75rem", "0.5rem"),
        }
        pad_main, pad_sidebar = padding_map[density]

        # Inject CSS variables for runtime theming
        st.markdown(
            f"""
            <style>
            :root {{
              --brand-primary: {brand};
              --bg: {bg};
              --panel: {panel};
              --text: {text};
              --content-max-width: {max_width}px;
              --pad-main: {pad_main};
              --pad-sidebar: {pad_sidebar};
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Apply computed density/width
        st.markdown(
            """
            <style>
            .block-container { max-width: var(--content-max-width) !important; padding-top: var(--pad-main) !important; }
            [data-testid="stSidebar"] .block-container { padding-top: var(--pad-sidebar) !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        return {
            "theme_mode": theme_mode,
            "brand": brand,
            "density": density,
            "max_width": max_width,
            "hero_title": hero_title,
        }
