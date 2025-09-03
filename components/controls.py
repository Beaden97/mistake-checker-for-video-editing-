"""Mobile-friendly analysis controls component."""
import streamlit as st
from typing import Dict, List, Optional


def _detect_mobile_browser() -> bool:
    """
    Detect if the user is likely on a mobile device based on viewport.
    
    Returns:
        True if mobile browser detected
    """
    # This is a heuristic - Streamlit doesn't provide direct mobile detection
    # We'll use session state to track mobile-optimized settings
    return st.session_state.get('mobile_optimized', False)


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
    # Detect if we should use mobile-optimized defaults
    is_mobile = _detect_mobile_browser()
    
    # Initialize session state defaults if not present
    if 'controls_safe_mode' not in st.session_state:
        st.session_state.controls_safe_mode = is_mobile  # Default safe mode on mobile
    if 'controls_deep_ocr' not in st.session_state:
        st.session_state.controls_deep_ocr = False
    if 'controls_frame_sampling' not in st.session_state:
        st.session_state.controls_frame_sampling = 60 if is_mobile else 30  # More conservative on mobile
    if 'controls_max_ocr_frames' not in st.session_state:
        st.session_state.controls_max_ocr_frames = 5 if is_mobile else 10  # Fewer frames on mobile
    if 'controls_text_language' not in st.session_state:
        st.session_state.controls_text_language = "English (US)"
    if 'controls_custom_dictionary' not in st.session_state:
        st.session_state.controls_custom_dictionary = ""
    
    with st.expander("⚙️ Analysis Settings", expanded=False):
        st.markdown("**Configure analysis options (optimized for all devices):**")
        
        # Mobile optimization toggle
        mobile_optimized = st.checkbox(
            "🔥 Mobile/Low-resource mode",
            value=st.session_state.get('mobile_optimized', False),
            key="mobile_optimized",
            help="Optimize settings for mobile devices and limited resources"
        )
        
        if mobile_optimized:
            st.info("📱 Mobile optimizations: Reduced memory usage, faster processing, conservative settings")
        
        # Basic toggles
        col1, col2 = st.columns(2)
        
        with col1:
            safe_mode = st.checkbox(
                "Safe mode",
                value=st.session_state.controls_safe_mode,
                key="controls_safe_mode",
                help="Lightweight analysis with reduced timeouts (recommended for mobile/cloud)"
            )
        
        with col2:
            deep_ocr = st.checkbox(
                "Full scan",
                value=st.session_state.controls_deep_ocr,
                key="controls_deep_ocr",
                disabled=safe_mode,
                help="Enable text detection and spell checking (disabled in safe mode)"
            )
        
        # Frame sampling settings with mobile-aware defaults
        col3, col4 = st.columns(2)
        
        with col3:
            frame_sampling_step = st.number_input(
                "Frame sampling step",
                min_value=10,
                max_value=120,
                value=st.session_state.controls_frame_sampling,
                step=10,
                key="controls_frame_sampling",
                help="Analyze every Nth frame (higher = faster, lower = more thorough)"
            )
        
        with col4:
            max_ocr_frames = st.number_input(
                "Max OCR frames",
                min_value=3,
                max_value=20,
                value=st.session_state.controls_max_ocr_frames,
                step=1,
                key="controls_max_ocr_frames",
                help="Maximum frames to analyze for text (fewer = less memory usage)"
            )
        
        # Text language selector
        text_language = st.selectbox(
            "Text language",
            options=["English (US)", "English (UK)"],
            index=0 if st.session_state.controls_text_language == "English (US)" else 1,
            key="controls_text_language",
            help="Choose language variant for spell checking"
        )
        
        # Custom dictionary with examples
        custom_dictionary = st.text_area(
            "Custom dictionary (brand names, slang, etc.)",
            value=st.session_state.controls_custom_dictionary,
            key="controls_custom_dictionary",
            height=100,
            placeholder="Enter custom words separated by commas or new lines\n\nExamples:\nTikTok, Instagram, YouTube, influencer, periodt, slay, bestie, vibe, lit, fire",
            help="Add custom words that should not be flagged as misspelled"
        )
        
        # Mobile-specific guidance
        if mobile_optimized or _detect_mobile_browser():
            st.markdown("**📱 Mobile Tips:**")
            st.markdown("- Use Safe mode for faster processing")
            st.markdown("- Keep Max OCR frames ≤ 5 for best performance")
            st.markdown("- Frame sampling ≥ 60 reduces memory usage")
            st.markdown("- Close other apps while analyzing videos")
    
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
    
    # Apply mobile optimizations if enabled
    if mobile_optimized:
        # Override with mobile-friendly settings
        frame_sampling_step = max(frame_sampling_step, 60)
        max_ocr_frames = min(max_ocr_frames, 5)
    
    return {
        'safe_mode': safe_mode,
        'deep_ocr': deep_ocr,
        'frame_sampling_step': frame_sampling_step,
        'max_ocr_frames': max_ocr_frames,
        'text_language': text_language,
        'spell_variant': spell_variant,
        'custom_words': custom_words,
        'mobile_optimized': mobile_optimized
    }