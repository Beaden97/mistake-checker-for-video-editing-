import streamlit as st
import tempfile
import os
import json
import time
import re
import requests
import cv2
import base64
from pathlib import Path
from urllib.parse import urlparse
from io import BytesIO
import yt_dlp

from app_theme import apply_base_theme, apply_runtime_theme_controls
from components.feedback import render_feedback_widget
from components.checklist import render_corrections_checklist
from components.controls import render_analysis_controls
from analyzers.runner import AnalyzerRunner, AnalysisConfig
from analyzers.common import is_cloud_environment, get_memory_usage, format_timestamp, safe_capture

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
4. **Click Analyze Video** to get AI-powered analysis
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
            "Full scan", 
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

# Example section above the text area
st.markdown("**💡 Example Description:**")
st.markdown("> A short dancing clip with text captions. Expected text: \"Welcome!\" : \"Follow us for more\". Look for: proper timing, text visibility, smooth transitions. Skip: freeze detection.")

description = st.text_area(
    "Describe in detail what the video is supposed to be:",
    value="",
    height=100,
    placeholder="Enter your video description here...\n\nTip: Use quotes with colons for expected text: \"Step 1: Start\" : \"Step 2: Continue\"\nUse phrases like 'Look for:' or 'Skip:' for analysis instructions.",
    help="Detailed descriptions help with better analysis and Notes Check comparison"
)

# Show description parsing preview if user has entered content
if description and description.strip():
    with st.expander("🤖 AI Prompt Analysis Preview", expanded=False):
        # Import the parser here to avoid circular imports
        from analyzers.description_parser import DescriptionParser
        
        parser = DescriptionParser()
        parsed_preview = parser.parse(description)
        preview_text = parser.format_parsing_preview(parsed_preview)
        
        if preview_text != "No structured content detected.":
            st.markdown(preview_text)
            st.info("💡 **Tip:** Use quotes with colons for expected text: `\"Step 1: Start\" : \"Step 2: Continue\"` and phrases like 'Look for:' or 'Check:' for analysis instructions.")
        else:
            st.markdown("📝 **General description detected** - no specific instructions or expected text found.")
            st.info("💡 **Enhanced Analysis:** Try adding expected text in quotes with colons or analysis instructions like 'Look for: audio sync, text clarity'")

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
    """Run the complete analysis pipeline with mobile optimization."""
    # Determine min_confidence_for_spell based on Full scan setting
    min_confidence = 0.35 if controls_config.get('deep_ocr', False) else 0.4
    
    # Apply mobile optimizations to configuration
    mobile_optimized = controls_config.get('mobile_optimized', False)
    
    # Create analysis configuration
    config = AnalysisConfig.from_dict({
        'safe_mode': controls_config.get('safe_mode', True) or mobile_optimized,  # Force safe mode if mobile optimized
        'deep_ocr': controls_config.get('deep_ocr', False) and not mobile_optimized,  # Disable full scan on mobile optimization
        'use_scenedetect': controls_config.get('use_scenedetect', False) and not mobile_optimized,  # Keep sidebar value for compatibility
        'pre_transcode': controls_config.get('pre_transcode', True) and not mobile_optimized,  # Skip transcoding on mobile to save time/memory
        'frame_sampling_step': controls_config.get('frame_sampling_step', 30),
        'max_ocr_frames': controls_config.get('max_ocr_frames', 10),
        'spell_variant': controls_config.get('spell_variant', 'US'),
        'custom_words': controls_config.get('custom_words', []),
        'min_confidence_for_spell': min_confidence,
        # Pass through the entire controls_config to test backward compatibility
        **controls_config
    })
    
    # Run analysis
    runner = AnalyzerRunner(config)
    return runner.analyze_video(video_path, description)


def generate_plain_language_report(results: dict) -> str:
    """Generate a plain-language report containing only editing errors with human-readable timecodes."""
    
    # Extract all critical issues (warnings and errors only)
    all_issues = []
    for analyzer_name, analyzer_data in results['analyzers'].items():
        if analyzer_data['success']:
            issues = analyzer_data['result'].get('issues', [])
            for issue in issues:
                if issue.get('severity') in ['warning', 'error']:
                    issue['analyzer'] = analyzer_name
                    all_issues.append(issue)
    
    # Generate report content
    report_lines = []
    report_lines.append("VIDEO EDITING MISTAKE CHECKER - ANALYSIS REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Summary
    report_lines.append(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Critical Issues Found: {len(all_issues)}")
    report_lines.append("")
    
    if not all_issues:
        report_lines.append("🎉 GREAT NEWS! No critical editing errors were detected in your video.")
        report_lines.append("Your video appears to be ready for publication!")
    else:
        report_lines.append("EDITING ERRORS REQUIRING ATTENTION:")
        report_lines.append("-" * 40)
        report_lines.append("")
        
        # Group by severity
        errors = [issue for issue in all_issues if issue.get('severity') == 'error']
        warnings = [issue for issue in all_issues if issue.get('severity') == 'warning']
        
        if errors:
            report_lines.append("🔴 CRITICAL ERRORS (Must Fix):")
            for i, issue in enumerate(errors, 1):
                timestamp = issue.get('timestamp', 'Unknown')
                message = issue.get('message', '')
                # Convert timestamp to human-readable format if it's in seconds
                try:
                    if isinstance(timestamp, (int, float)) and timestamp > 0:
                        timestamp = format_timestamp(timestamp)
                except:
                    pass
                report_lines.append(f"  {i}. [{timestamp}] {message}")
            report_lines.append("")
        
        if warnings:
            report_lines.append("🟡 WARNINGS (Recommended to Fix):")
            for i, issue in enumerate(warnings, 1):
                timestamp = issue.get('timestamp', 'Unknown')
                message = issue.get('message', '')
                # Convert timestamp to human-readable format if it's in seconds
                try:
                    if isinstance(timestamp, (int, float)) and timestamp > 0:
                        timestamp = format_timestamp(timestamp)
                except:
                    pass
                report_lines.append(f"  {i}. [{timestamp}] {message}")
            report_lines.append("")
    
    report_lines.append("=" * 50)
    report_lines.append("Report generated by Video Editing Mistake Checker")
    report_lines.append("Built with ❤️ for clumsy video editors!")
    
    return "\n".join(report_lines)


def extract_video_thumbnail(video_path: str, timestamp_seconds: float, max_size: tuple = (300, 200)) -> str:
    """Extract a thumbnail from video at specified timestamp and return as base64 string."""
    try:
        cap = safe_capture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        
        # Calculate frame number
        frame_number = int(timestamp_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Resize frame to max_size while maintaining aspect ratio
        height, width = frame.shape[:2]
        if width > max_size[0] or height > max_size[1]:
            scale = min(max_size[0]/width, max_size[1]/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert BGR to RGB for proper display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Encode as JPEG and convert to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode()
        
        return img_base64
    except Exception as e:
        print(f"Error extracting thumbnail at {timestamp_seconds}s: {e}")
        return None


def render_thumbnail_snapshots(all_issues: list, video_path: str):
    """Render thumbnail snapshots section for critical issues."""
    
    # Filter for critical issues with valid timestamps
    critical_issues_with_timestamps = []
    for issue in all_issues:
        if issue.get('severity') in ['warning', 'error']:
            timestamp = issue.get('timestamp', '')
            # Try to extract numeric timestamp
            if isinstance(timestamp, str) and ':' in timestamp:
                try:
                    # Convert MM:SS to seconds
                    parts = timestamp.split(':')
                    if len(parts) == 2:
                        minutes, seconds = int(parts[0]), int(parts[1])
                        timestamp_seconds = minutes * 60 + seconds
                        critical_issues_with_timestamps.append({
                            'issue': issue,
                            'timestamp_seconds': timestamp_seconds,
                            'timestamp_display': timestamp
                        })
                except:
                    pass
            elif isinstance(timestamp, (int, float)) and timestamp > 0:
                critical_issues_with_timestamps.append({
                    'issue': issue,
                    'timestamp_seconds': timestamp,
                    'timestamp_display': format_timestamp(timestamp)
                })
    
    if not critical_issues_with_timestamps:
        return
    
    st.subheader("📸 Thumbnail Snapshots")
    st.markdown("*Visual previews of detected issues:*")
    
    # Limit to first 5 critical issues to avoid overloading
    max_thumbnails = min(5, len(critical_issues_with_timestamps))
    
    for i, item in enumerate(critical_issues_with_timestamps[:max_thumbnails]):
        issue = item['issue']
        timestamp_seconds = item['timestamp_seconds']
        timestamp_display = item['timestamp_display']
        
        with st.expander(f"🔍 Issue at {timestamp_display}: {issue['message'][:50]}..."):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Extract and display thumbnail
                thumbnail_b64 = extract_video_thumbnail(video_path, timestamp_seconds)
                if thumbnail_b64:
                    st.markdown(f'<img src="data:image/jpeg;base64,{thumbnail_b64}" style="width:100%; border-radius:8px;">', unsafe_allow_html=True)
                else:
                    st.info("Could not extract thumbnail")
            
            with col2:
                st.write(f"**Timestamp:** {timestamp_display}")
                st.write(f"**Severity:** {'🔴 Critical' if issue.get('severity') == 'error' else '🟡 Warning'}")
                st.write(f"**Issue:** {issue['message']}")
                st.write(f"**Source:** {issue.get('analyzer', 'Unknown').replace('_', ' ').title()}")


def display_results(results: dict, video_path: str = None):
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
    
    # Add thumbnail snapshots section for critical issues
    if critical_issues and video_path:
        render_thumbnail_snapshots(all_issues, video_path)
    
    # Download Reports Section
    st.markdown("---")
    st.subheader("📥 Download Reports")
    
    # Plain-language report download button
    plain_report = generate_plain_language_report(results)
    st.download_button(
        label="📄 Download Report",
        data=plain_report,
        file_name=f"video_editing_report_{int(time.time())}.txt",
        mime="text/plain",
        help="Download a human-readable report containing only editing errors with clear timestamps"
    )
    
    # Analysis details
    with st.expander("📋 Detailed Analysis Results"):
        # Show parsed description if available
        parsed_desc = results['metadata'].get('parsed_description')
        if parsed_desc and (parsed_desc.get('expected_text') or parsed_desc.get('analysis_instructions')):
            with st.expander("🤖 AI Prompt Analysis"):
                st.write("**General Description:**")
                st.write(parsed_desc.get('general_description', 'None'))
                
                if parsed_desc.get('expected_text'):
                    st.write(f"**Expected Text ({len(parsed_desc['expected_text'])} items):**")
                    for i, text in enumerate(parsed_desc['expected_text'], 1):
                        st.write(f"{i}. \"{text}\"")
                
                if parsed_desc.get('analysis_instructions'):
                    st.write(f"**Analysis Instructions ({len(parsed_desc['analysis_instructions'])} items):**")
                    for i, instruction in enumerate(parsed_desc['analysis_instructions'], 1):
                        st.write(f"{i}. {instruction}")
                
                if parsed_desc.get('look_for_keywords'):
                    st.write(f"**Focus Keywords:** {', '.join(parsed_desc['look_for_keywords'])}")
        
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
    
    # Debug panel with enhanced OCR error reporting
    with st.expander("🐛 Debug Panel"):
        # Show any error traces from issues
        error_traces = []
        ocr_issues = []
        mobile_warnings = []
        
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
                    elif issue.get('type') in ['ocr_error', 'spelling']:
                        ocr_issues.append({
                            'analyzer': analyzer_name,
                            'timestamp': issue.get('timestamp', 'Unknown'),
                            'type': issue.get('type'),
                            'message': issue.get('message', '')
                        })
                    elif issue.get('type') == 'memory_warning':
                        mobile_warnings.append({
                            'analyzer': analyzer_name,
                            'timestamp': issue.get('timestamp', 'Unknown'),
                            'message': issue.get('message', '')
                        })
        
        # Show OCR-specific issues
        if ocr_issues:
            st.write("**📝 OCR Analysis Results:**")
            for issue in ocr_issues:
                if issue['type'] == 'spelling':
                    st.success(f"✅ **[{issue['timestamp']}]** {issue['message']}")
                else:
                    st.warning(f"⚠️ **[{issue['timestamp']}]** {issue['message']}")
        
        # Show mobile/memory warnings
        if mobile_warnings:
            st.write("**📱 Mobile/Resource Warnings:**")
            for warning in mobile_warnings:
                st.info(f"📱 **[{warning['timestamp']}]** {warning['message']}")
        
        # Show error tracebacks (if any)
        if error_traces:
            st.write("**🔍 Error Tracebacks:**")
            for trace in error_traces:
                with st.expander(f"Error in {trace['analyzer']} at {trace['timestamp']}"):
                    st.code(trace['trace'], language="python")
        
        # OCR Performance Metrics
        ocr_data = results['analyzers'].get('ocr', {})
        if ocr_data.get('success'):
            ocr_metadata = ocr_data['result'].get('metadata', {})
            st.write("**📊 OCR Performance:**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frames Analyzed", ocr_metadata.get('frames_analyzed', 0))
            with col2:
                st.metric("Text Elements Found", ocr_metadata.get('total_text_elements', 0))
            with col3:
                st.metric("Processing Time", f"{ocr_data.get('duration', 0):.1f}s")
            
            # Mobile optimization status
            if ocr_metadata.get('mobile_environment_detected'):
                st.info("📱 Mobile environment detected - optimizations applied")
            if ocr_metadata.get('preprocessing_applied'):
                st.success("🔧 Frame preprocessing applied for better OCR accuracy")
        
        st.write("**🌍 Environment:**")
        st.json(results['metadata']['environment'])
        
        st.write("**⚙️ Configuration:**")
        st.json(results['metadata']['config'])
        
        st.write("**📄 Raw Results:**")
        st.json(results)
    
    # Advanced downloads in expander
    with st.expander("🔧 Advanced"):
        st.markdown("**Technical Downloads & Debug Information**")
        
        # JSON report download
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
            display_results(results, temp_video_path)
    
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