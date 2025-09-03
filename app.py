import streamlit as st
import tempfile
import os
import json
import time
import re
import requests
from pathlib import Path
from urllib.parse import urlparse
import yt_dlp

from app_theme import apply_base_theme, apply_runtime_theme_controls
from components.feedback import render_feedback_widget
from components.checklist import render_corrections_checklist
from components.controls import render_analysis_controls
from analyzers.runner import AnalyzerRunner, AnalysisConfig
from analyzers.common import is_cloud_environment, get_memory_usage

# Apply theme before any Streamlit output
apply_base_theme(page_title="The Video Editing Mistake Checker", page_icon=None)
_appearance = apply_runtime_theme_controls()

# Add feedback widget
render_feedback_widget()

# Initialize session state for button control
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# --- INSTRUCTIONS SECTION ---
_default_hero = "The Video Editing Mistake Checker"
hero_title = (_appearance or {}).get("hero_title", _default_hero)
st.title(hero_title)
st.markdown("""
Hi! Are you a clumsy video editor who often misses small mistakes on projects? I have inattentive adhd, and I created this tool to help spot the little, clumsy mistakes for our work before it gets sent out to that big boss or client. I often feel rubbish about myself because I send out projects with mistakes I've missed. I'm hoping this little app will help us all feel more confident in the work we produce.
""")

st.markdown("""
#### How to Use This App
1. **Configure analysis settings** in the sidebar (Safe mode is recommended for Streamlit Cloud)
2. **Upload your TikTok video** or provide a URL for analysis
3. **Describe exactly what your video should be** in the notes box  
4. **Click Analyze Video** to get AI-powered analysis with fail-soft execution
5. **Download the detailed JSON report** for comprehensive results and debugging info

The app now features modular analyzers with timeout protection and graceful error handling.
""")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Analysis Configuration")
    
    # Detect cloud environment
    is_cloud = is_cloud_environment()
    
    # Safe mode toggle (default on for cloud)
    safe_mode = st.toggle(
        "Safe mode (recommended on Streamlit Cloud)", 
        value=is_cloud,
        help="Enables lightweight analysis with reduced timeouts and disabled heavy operations"
    )
    
    if safe_mode:
        st.info("🛡️ Safe mode: Heavy analyzers disabled, reduced timeouts")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        deep_ocr = st.toggle(
            "Deep OCR Analysis", 
            value=False,
            disabled=safe_mode,
            help="Enable text detection and spell checking (disabled in safe mode)"
        )
        
        use_scenedetect = st.toggle(
            "Use SceneDetect Library", 
            value=False,
            disabled=safe_mode,
            help="Use advanced scene detection (disabled in safe mode)"
        )
        
        pre_transcode = st.toggle(
            "Pre-transcode video", 
            value=not safe_mode,
            help="Normalize video format before analysis for better compatibility"
        )
        
        frame_sampling = st.selectbox(
            "Frame sampling step",
            options=[1, 2, 3, 5, 10],
            index=0 if not safe_mode else 2,
            help="Analyze every Nth frame (higher = faster but less thorough)"
        )
        
        max_ocr_frames = st.number_input(
            "Max OCR frames",
            min_value=1,
            max_value=50,
            value=5 if safe_mode else 10,
            help="Maximum frames to analyze for text (reduces processing time)"
        )
    
    # Environment info
    with st.expander("🌍 Environment Info"):
        st.write(f"**Cloud detected:** {is_cloud}")
        st.write(f"**Safe mode:** {safe_mode}")
        memory = get_memory_usage()
        if memory:
            st.write(f"**Memory usage:** {memory:.1f} MB")

# --- FILE UPLOAD & DESCRIPTION ---
st.subheader("📹 Video Input")
uploaded_file = st.file_uploader(
    "Upload your TikTok video", 
    type=["mp4", "mov", "avi", "mkv"],
    help="Supported formats: MP4, MOV, AVI, MKV"
)

description = st.text_area(
    "Describe in detail what the video is supposed to be:",
    value="A short dancing clip with text captions. Example: First 5 seconds - dancer enters from left, caption 'Welcome!' appears. Transition to close-up. Audio: upbeat pop, no silence. End with logo.",
    height=100,
    help="Detailed descriptions help with better analysis and Notes Check comparison"
)

# --- URL-BASED ANALYSIS ALTERNATIVE ---
st.markdown("#### Alternative: Analyze from URL")
video_url_for_analysis = st.text_input(
    "Video URL (YouTube, TikTok, Google Drive, or direct link)",
    placeholder="https://www.youtube.com/watch?v=... or https://drive.google.com/file/d/.../view or https://example.com/video.mp4",
    help="Supports YouTube, TikTok, Google Drive shared links, and direct video URLs. For Google Drive, make sure the link is set to 'Anyone with the link can view'"
)

# --- INLINE CONTROLS (MOBILE-FRIENDLY) ---
controls_config = render_analysis_controls()

# --- SUBMIT LOGIC ---
can_submit = (
    (uploaded_file is not None or video_url_for_analysis.strip()) 
    and description.strip() 
    and description.strip() != "A short dancing clip with text captions. Example: First 5 seconds - dancer enters from left, caption 'Welcome!' appears. Transition to close-up. Audio: upbeat pop, no silence. End with logo."
    and not st.session_state.analysis_running
)

submit_button = st.button(
    "🔍 Analyze Video", 
    type="primary",
    disabled=not can_submit,
    help="Provide video and description to enable analysis" if not can_submit else "Start comprehensive video analysis"
)

# URL analysis button
analyze_url_button = st.button(
    "🔍 Analyze Video from URL",
    type="secondary",
    disabled=not (video_url_for_analysis.strip() and description.strip() and not st.session_state.analysis_running)
)


def extract_google_drive_id(url: str) -> str:
    """Extract file ID from Google Drive URL."""
    patterns = [
        r'/file/d/([a-zA-Z0-9-_]+)',
        r'[?&]id=([a-zA-Z0-9-_]+)',
        r'/open\?id=([a-zA-Z0-9-_]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    raise ValueError("Could not extract Google Drive file ID from URL")


def download_from_google_drive(file_id: str, destination: str) -> str:
    """Download file from Google Drive using the file ID."""
    session = requests.Session()
    
    # First request to get the download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = session.get(url, stream=True)
    
    # Check if download warning is present (for large files)
    if "download_warning" in response.text:
        # Extract confirm token for large files
        for line in response.text.split('\n'):
            if 'confirm=' in line:
                token_match = re.search(r'confirm=([^&]+)', line)
                if token_match:
                    token = token_match.group(1)
                    url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
                    response = session.get(url, stream=True)
                    break
    
    # Download the file
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    
    return destination


def is_google_drive_url(url: str) -> bool:
    """Check if URL is a Google Drive link."""
    drive_domains = ['drive.google.com', 'docs.google.com']
    parsed = urlparse(url)
    return any(domain in parsed.netloc for domain in drive_domains)


def download_video_from_url(url: str) -> str:
    """Download video from URL using appropriate method."""
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Check if it's a Google Drive URL
        if is_google_drive_url(url):
            st.write("📁 Detected Google Drive link, processing...")
            file_id = extract_google_drive_id(url)
            
            # Create temporary file with video extension
            temp_file = os.path.join(temp_dir, f"gdrive_video_{file_id}.mp4")
            download_from_google_drive(file_id, temp_file)
            
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                return temp_file
            else:
                raise Exception("Downloaded file is empty or not found")
        
        else:
            # Use yt-dlp for other URLs (YouTube, TikTok, direct links, etc.)
            st.write("🌐 Using yt-dlp for video download...")
            ydl_opts = {
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'format': 'best[height<=720]/best',  # Limit to 720p for processing
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                
                if os.path.exists(filename):
                    return filename
                
                # Try to find the downloaded file
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.isfile(file_path):
                        return file_path
                        
                raise Exception("Downloaded file not found")
            
    except Exception as e:
        raise Exception(f"Failed to download video: {str(e)}")


def run_analysis(video_path: str, description: str, controls_config: dict) -> dict:
    """Run the complete analysis pipeline."""
    # Determine min_confidence_for_spell based on Deep OCR setting
    min_confidence = 0.35 if controls_config.get('deep_ocr', False) else 0.4
    
    # Create analysis configuration
    config = AnalysisConfig(
        safe_mode=controls_config.get('safe_mode', safe_mode),
        deep_ocr=controls_config.get('deep_ocr', deep_ocr),
        use_scenedetect=use_scenedetect,  # Keep sidebar value for compatibility
        pre_transcode=pre_transcode,  # Keep sidebar value for compatibility
        frame_sampling_step=controls_config.get('frame_sampling_step', frame_sampling),
        max_ocr_frames=controls_config.get('max_ocr_frames', max_ocr_frames),
        spell_variant=controls_config.get('spell_variant', 'US'),
        custom_words=controls_config.get('custom_words', []),
        min_confidence_for_spell=min_confidence
    )
    
    # Run analysis
    runner = AnalyzerRunner(config)
    return runner.analyze_video(video_path, description)


def display_results(results: dict):
    """Display analysis results with proper formatting."""
    summary = results['summary']
    
    # Main results header
    st.subheader("📊 Analysis Results")
    
    # Success state check
    if not summary.get('has_critical_issues', True):
        st.success("🎉 All good to go!! 👍")
        st.balloons()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Issues Found", summary.get('critical_issues_count', 0))
    with col2:
        st.metric("Analyzers Run", summary.get('analyzers_run', 0))
    with col3:
        st.metric("Success Rate", f"{summary.get('success_rate', 0):.0%}")
    with col4:
        st.metric("Total Time", f"{summary.get('total_duration', 0):.1f}s")
    
    # Issues list
    all_issues = []
    for analyzer_name, analyzer_data in results['analyzers'].items():
        if analyzer_data['success']:
            issues = analyzer_data['result'].get('issues', [])
            for issue in issues:
                issue['analyzer'] = analyzer_name
                all_issues.append(issue)
    
    # Filter for critical issues (warnings and errors) for the checklist
    critical_issues = [
        issue for issue in all_issues 
        if issue.get('severity') in ['warning', 'error']
    ]
    
    if critical_issues:
        # Use interactive checklist for critical issues
        all_checked = render_corrections_checklist(critical_issues)
        
        # Show informational issues separately if any
        info_issues = [issue for issue in all_issues if issue.get('severity') == 'info']
        if info_issues:
            with st.expander("ℹ️ Additional Information"):
                for issue in info_issues:
                    st.write(f"🔵 **[{issue['timestamp']}]** {issue['message']}")
    elif all_issues:
        # Only informational issues
        st.subheader("ℹ️ Information")
        for issue in all_issues:
            st.write(f"🔵 **[{issue['timestamp']}]** {issue['message']}")
    else:
        st.success("No issues detected in the analysis!")
    
    # Analysis details
    with st.expander("📋 Detailed Analysis Results"):
        for analyzer_name, analyzer_data in results['analyzers'].items():
            with st.expander(f"🔍 {analyzer_name.replace('_', ' ').title()}"):
                if analyzer_data['success']:
                    result = analyzer_data['result']
                    
                    # Analyzer metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration", f"{analyzer_data['duration']:.2f}s")
                    with col2:
                        st.metric("Issues", analyzer_data['issues_count'])
                    with col3:
                        memory_used = analyzer_data.get('memory_used', 0)
                        st.metric("Memory", f"{memory_used:.1f}MB" if memory_used else "N/A")
                    
                    # Metadata
                    if result.get('metadata'):
                        st.json(result['metadata'])
                    
                    # Issues for this analyzer
                    if result.get('issues'):
                        st.write("**Issues:**")
                        for issue in result['issues']:
                            st.write(f"- {issue['message']}")
                else:
                    st.error(f"Analyzer failed: {analyzer_data['result'].get('error', 'Unknown error')}")
                    # Show full traceback if available
                    if 'error' in analyzer_data['result'] and analyzer_data['result']['error']:
                        with st.expander("🔍 Error Traceback"):
                            st.code(analyzer_data['result']['error'], language="python")
    
    # Debug panel
    with st.expander("🐛 Debug Panel"):
        # Show any error traces from issues
        error_traces = []
        for analyzer_name, analyzer_data in results['analyzers'].items():
            if analyzer_data['success']:
                issues = analyzer_data['result'].get('issues', [])
                for issue in issues:
                    if issue.get('type') == 'error_trace':
                        error_traces.append({
                            'analyzer': analyzer_name,
                            'timestamp': issue.get('timestamp', 'Unknown'),
                            'trace': issue.get('message', '')
                        })
        
        if error_traces:
            st.write("**Error Tracebacks:**")
            for trace in error_traces:
                with st.expander(f"Error in {trace['analyzer']} at {trace['timestamp']}"):
                    st.code(trace['trace'], language="python")
        
        st.write("**Environment:**")
        st.json(results['metadata']['environment'])
        
        st.write("**Configuration:**")
        st.json(results['metadata']['config'])
        
        st.write("**Raw Results:**")
        st.json(results)
    
    # Download report
    json_report = json.dumps(results, indent=2, default=str)
    st.download_button(
        label="📥 Download JSON Report",
        data=json_report,
        file_name=f"video_analysis_report_{int(time.time())}.json",
        mime="application/json",
        help="Download complete analysis results and debug information"
    )


# --- MAIN ANALYSIS LOGIC ---
if (submit_button and can_submit) or (analyze_url_button and video_url_for_analysis.strip()):
    st.session_state.analysis_running = True
    temp_video_path = None
    
    try:
        # Determine video source
        if analyze_url_button and video_url_for_analysis.strip():
            # URL-based analysis
            with st.status("🌐 Downloading video from URL...", expanded=True) as status:
                st.write("Fetching video using yt-dlp...")
                temp_video_path = download_video_from_url(video_url_for_analysis.strip())
                status.update(label="✅ Video downloaded successfully!", state="complete")
        
        elif submit_button and uploaded_file:
            # File upload analysis
            with st.status("📁 Processing uploaded video...", expanded=True) as status:
                st.write("Saving uploaded file...")
                suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
                    uploaded_file.seek(0)
                    # Read in chunks to avoid memory issues
                    while True:
                        chunk = uploaded_file.read(4 * 1024 * 1024)  # 4MB chunks
                        if not chunk:
                            break
                        temp_video.write(chunk)
                    temp_video_path = temp_video.name
                status.update(label="✅ Video file ready for analysis!", state="complete")
        
        if temp_video_path:
            # Display video preview
            st.video(temp_video_path)
            st.write(f"**Your Notes:** {description}")
            
            # Run analysis with progress tracking
            with st.status("🤖 Running AI analysis pipeline...", expanded=True) as status:
                phases = [
                    "🔍 Initializing analyzers...",
                    "📐 Checking aspect ratio...",
                    "🎬 Detecting scenes...",
                    "⚫ Scanning for black frames...",
                    "⚡ Analyzing flicker...",
                    "🧊 Detecting freezes...",
                    "📝 Processing text (if enabled)...",
                    "🏷️ Checking credits...",
                    "📊 Compiling results..."
                ]
                
                progress_bar = st.progress(0)
                for i, phase in enumerate(phases):
                    st.write(phase)
                    progress_bar.progress((i + 1) / len(phases))
                    if i < len(phases) - 1:  # Don't sleep on last iteration
                        time.sleep(0.5)
                
                # Run actual analysis
                results = run_analysis(temp_video_path, description, controls_config)
                
                status.update(label="✅ Analysis complete!", state="complete")
            
            # Display results
            display_results(results)
    
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.error("Please try a different video file or check the URL. For large files, try the URL analysis option.")
        
        # Show error details in debug
        with st.expander("🐛 Error Details"):
            st.code(str(e))
    
    finally:
        # Clean up
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass
        
        st.session_state.analysis_running = False

elif not can_submit and not st.session_state.analysis_running:
    st.info("📋 Please upload a video (or provide URL) and add a detailed description to enable analysis.")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ for clumsy video editors who want to catch their mistakes before clients do!*")