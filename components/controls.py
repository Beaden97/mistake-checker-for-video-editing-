"""Mobile-friendly analysis controls component."""
import streamlit as st
from typing import Dict, List, Optional


def render_analysis_controls() -> Dict:
    """
    Render mobile-friendly inline analysis controls in an expander.
    
    Returns:
        Dict with parsed configuration including:
        - safe_mode: bool
        - deep_ocr: bool
        - frame_sampling_step: int
        - max_ocr_frames: int
        - text_language: str ("US" or "UK")
        - custom_words: List[str]
    """
    # Initialize session state defaults if not present
    if 'controls_safe_mode' not in st.session_state:
        st.session_state.controls_safe_mode = False
    if 'controls_deep_ocr' not in st.session_state:
        st.session_state.controls_deep_ocr = False
    if 'controls_frame_sampling' not in st.session_state:
        st.session_state.controls_frame_sampling = 30
    if 'controls_max_ocr_frames' not in st.session_state:
        st.session_state.controls_max_ocr_frames = 10
    if 'controls_text_language' not in st.session_state:
        st.session_state.controls_text_language = "English (US)"
    if 'controls_custom_dictionary' not in st.session_state:
        st.session_state.controls_custom_dictionary = ""
    
    with st.expander("⚙️ Analysis Settings", expanded=False):
        st.markdown("**Configure analysis options (mobile-friendly controls):**")
        
        # Basic toggles
        col1, col2 = st.columns(2)
        
        with col1:
            safe_mode = st.checkbox(
                "Safe mode",
                value=st.session_state.controls_safe_mode,
                key="controls_safe_mode",
                help="Lightweight analysis with reduced timeouts"
            )
        
        with col2:
            deep_ocr = st.checkbox(
                "Deep OCR analysis",
                value=st.session_state.controls_deep_ocr,
                key="controls_deep_ocr",
                disabled=safe_mode,
                help="Enable text detection and spell checking (disabled in safe mode)"
            )
        
        # Frame sampling settings
        col3, col4 = st.columns(2)
        
        with col3:
            frame_sampling_step = st.number_input(
                "Frame sampling step",
                min_value=10,
                max_value=100,
                value=st.session_state.controls_frame_sampling,
                step=10,
                key="controls_frame_sampling",
                help="Analyze every Nth frame (higher = faster but less thorough)"
            )
        
        with col4:
            max_ocr_frames = st.number_input(
                "Max OCR frames",
                min_value=5,
                max_value=50,
                value=st.session_state.controls_max_ocr_frames,
                step=5,
                key="controls_max_ocr_frames",
                help="Maximum frames to analyze for text (reduces processing time)"
            )
        
        # Text language selector
        text_language = st.selectbox(
            "Text language",
            options=["English (US)", "English (UK)"],
            index=0 if st.session_state.controls_text_language == "English (US)" else 1,
            key="controls_text_language",
            help="Choose language variant for spell checking"
        )
        
        # Custom dictionary
        custom_dictionary = st.text_area(
            "Custom dictionary (brand names, slang, etc.)",
            value=st.session_state.controls_custom_dictionary,
            key="controls_custom_dictionary",
            height=80,
            placeholder="Enter custom words separated by commas or new lines\nExample: TikTok, influencer, slay, periodt",
            help="Add custom words that should not be flagged as misspelled"
        )
    
    # Parse custom words from textarea
    custom_words = []
    if custom_dictionary.strip():
        # Split by both commas and newlines, then clean
        words = []
        for line in custom_dictionary.split('\n'):
            words.extend([word.strip() for word in line.split(',')])
        custom_words = [word.lower() for word in words if word.strip()]
    
    # Determine spell variant
    spell_variant = "UK" if text_language.endswith("(UK)") else "US"
    
    return {
        'safe_mode': safe_mode,
        'deep_ocr': deep_ocr,
        'frame_sampling_step': frame_sampling_step,
        'max_ocr_frames': max_ocr_frames,
        'text_language': text_language,
        'spell_variant': spell_variant,
        'custom_words': custom_words
    }