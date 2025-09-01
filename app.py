import streamlit as st
import tempfile
import cv2
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from paddleocr import PaddleOCR
from spellchecker import SpellChecker
import os
import requests
from pytube import YouTube
from urllib.parse import urlparse
import re

# --- INSTRUCTIONS SECTION ---
st.title("AI TikTok Video QA (Deep Learning, Web-Based, No Install)")
st.markdown("""
#### How to Use This App
1. **Upload your TikTok video below.**
2. **Describe exactly what your video should be** in the notes box.  
   - Please include as much detail as possible!  
   - For example: *What scenes should appear? What text should be shown? What actions, transitions, or effects are expected? What should the audio sound like? What timing is important?*
3. **Optional: Enter a reference video URL** to compare your video against a reference video (YouTube or direct video link).
4. The app will analyze your video for common editing mistakes **and compare it to your notes** to find any mismatches.
5. If a reference video is provided, it will also show a detailed comparison between your video and the reference.
""")

# --- FILE UPLOAD & DESCRIPTION ---
uploaded_file = st.file_uploader("Upload your TikTok video", type=["mp4", "mov"])
description = st.text_area(
    "Describe in detail what the video is supposed to be:",
    "A short dancing clip with text captions. Example: First 5 seconds - dancer enters from left, caption 'Welcome!' appears. Transition to close-up. Audio: upbeat pop, no silence. End with logo."
)

# --- REFERENCE VIDEO URL ---
st.markdown("---")
st.subheader("Optional: Reference Video Comparison")
reference_url = st.text_input(
    "Enter a reference video URL (YouTube or direct video link):",
    placeholder="https://www.youtube.com/watch?v=... or https://example.com/video.mp4",
    help="Provide a URL to a reference video to compare formatting and features with your uploaded video."
)

# --- AI/DEEP LEARNING VIDEO ANALYSIS FUNCTIONS ---
@st.cache_resource
def get_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

@st.cache_resource
def get_spell():
    return SpellChecker()

ocr = get_ocr()
spell = get_spell()

def analyze_aspect_ratio(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    mistake = None
    if ret:
        h, w = frame.shape[:2]
        ratio = w / h
        if not (0.55 < ratio < 0.6):  # 9:16 is ~0.5625
            mistake = f"Aspect ratio is not TikTok vertical (9:16), got {w}:{h}"
    cap.release()
    return mistake

def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    return [(int(start.get_seconds()), int(end.get_seconds())) for start, end in scene_list]

def detect_black_frames(video_path, threshold=15, min_duration=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    black_timestamps = []
    current_black = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < threshold:
            if current_black is None:
                current_black = frame_idx
        else:
            if current_black is not None:
                duration = (frame_idx - current_black) / fps
                if duration >= min_duration:
                    black_timestamps.append(
                        f"{int(current_black // fps):02d}:{int(current_black % fps):02d}"
                    )
                current_black = None
        frame_idx += 1
    cap.release()
    return black_timestamps

def detect_flicker(video_path, threshold=40):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_mean = None
    flicker_timestamps = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if prev_mean is not None:
            if abs(mean_brightness - prev_mean) > threshold:
                timestamp = int(frame_idx // fps)
                flicker_timestamps.append(f"{timestamp//60:02d}:{timestamp%60:02d}")
        prev_mean = mean_brightness
        frame_idx += 1
    cap.release()
    return list(set(flicker_timestamps))

def detect_freeze(video_path, freeze_threshold=0.99, min_freeze_duration=20):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_frame = None
    freeze_start = None
    freeze_timestamps = []
    frame_idx = 0
    freeze_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            similarity = np.corrcoef(prev_frame.flatten(), gray.flatten())[0,1]
            if similarity > freeze_threshold:
                if freeze_start is None:
                    freeze_start = frame_idx
                    freeze_count = 1
                else:
                    freeze_count += 1
            else:
                if freeze_start is not None and freeze_count >= min_freeze_duration:
                    timestamp = int(freeze_start // fps)
                    freeze_timestamps.append(f"{timestamp//60:02d}:{timestamp%60:02d}")
                freeze_start = None
                freeze_count = 0
        prev_frame = gray
        frame_idx += 1
    if freeze_start is not None and freeze_count >= min_freeze_duration:
        timestamp = int(freeze_start // fps)
        freeze_timestamps.append(f"{timestamp//60:02d}:{timestamp%60:02d}")
    cap.release()
    return list(set(freeze_timestamps))

def spell_check_texts(video_path, ocr, spell, n_samples=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    mistakes = []
    all_texts = []
    for idx in np.linspace(0, frame_count-1, num=n_samples, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img_path = f"frame_{idx}.jpg"
        cv2.imwrite(img_path, frame)
        ocr_results = ocr.ocr(img_path)
        texts = [l[1][0] for line in ocr_results for l in line]
        all_texts.extend(texts)
        if texts:
            words = []
            for text in texts:
                words.extend([w for w in text.split() if w.isalpha()])
            misspelled = spell.unknown(words)
            for word in misspelled:
                timestamp = int(idx // fps)
                mistakes.append((f"{timestamp//60:02d}:{timestamp%60:02d}", f"Potential typo: '{word}'"))
        os.remove(img_path)
    cap.release()
    return mistakes, all_texts

def compare_to_notes(all_texts, description):
    # Very basic notes comparison: check if all main keywords from notes appear in video text
    notes_words = [w.lower() for w in description.split() if w.isalpha() and len(w) > 3]
    video_words = set([w.lower() for t in all_texts for w in t.split() if w.isalpha()])
    missing = [w for w in notes_words if w not in video_words]
    if missing:
        return f"Content Mismatch: The following expected keywords from your notes were NOT found in the video text: {', '.join(missing)}"
    return "No obvious content mismatches between your notes and the video text detected."

def download_video_from_url(url):
    """Download video from URL (YouTube or direct link) and return local path."""
    try:
        # Check if it's a YouTube URL
        if "youtube.com" in url or "youtu.be" in url:
            # Use pytube for YouTube videos
            yt = YouTube(url)
            # Get the highest quality mp4 stream
            stream = yt.streams.filter(file_extension='mp4').get_highest_resolution()
            if not stream:
                # Fallback to any available stream
                stream = yt.streams.first()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                stream.download(output_path=os.path.dirname(temp_file.name), filename=os.path.basename(temp_file.name))
                return temp_file.name
        else:
            # Direct video link - use requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Check if it's actually a video file
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('video/'):
                # Try to guess from URL extension
                parsed_url = urlparse(url)
                if not any(parsed_url.path.lower().endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.mkv']):
                    raise ValueError("URL does not appear to point to a video file")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                return temp_file.name
                
    except Exception as e:
        raise Exception(f"Failed to download video from URL: {str(e)}")

def get_video_analysis_results(video_path, description):
    """Get detailed analysis results for a video (used for comparison)."""
    results = {}
    
    # Aspect ratio
    results['aspect_ratio'] = analyze_aspect_ratio(video_path)
    
    # Scene detection
    try:
        scenes = detect_scenes(video_path)
        results['scenes'] = scenes
        results['short_scenes'] = [s for s in scenes if s[1] - s[0] < 1]
    except Exception:
        results['scenes'] = []
        results['short_scenes'] = []
    
    # Black frames
    results['black_frames'] = detect_black_frames(video_path)
    
    # Flicker detection
    results['flicker_frames'] = detect_flicker(video_path)
    
    # Freeze detection
    results['freeze_frames'] = detect_freeze(video_path)
    
    # OCR & spell check
    text_mistakes, all_texts = spell_check_texts(video_path, ocr, spell)
    results['text_mistakes'] = text_mistakes
    results['all_texts'] = all_texts
    
    return results

def compare_videos(video1_path, video2_path, description):
    """Compare two videos and return detailed comparison results."""
    st.write("Analyzing uploaded video...")
    uploaded_results = get_video_analysis_results(video1_path, description)
    
    st.write("Analyzing reference video...")
    reference_results = get_video_analysis_results(video2_path, description)
    
    comparison = {
        'uploaded': uploaded_results,
        'reference': reference_results,
        'differences': [],
        'similarities': []
    }
    
    # Compare aspect ratios
    if uploaded_results['aspect_ratio'] and reference_results['aspect_ratio']:
        comparison['differences'].append(f"Both videos have aspect ratio issues: Uploaded - {uploaded_results['aspect_ratio']} | Reference - {reference_results['aspect_ratio']}")
    elif uploaded_results['aspect_ratio'] and not reference_results['aspect_ratio']:
        comparison['differences'].append(f"Uploaded video has aspect ratio issue: {uploaded_results['aspect_ratio']} | Reference video has correct aspect ratio")
    elif not uploaded_results['aspect_ratio'] and reference_results['aspect_ratio']:
        comparison['differences'].append(f"Uploaded video has correct aspect ratio | Reference video has aspect ratio issue: {reference_results['aspect_ratio']}")
    else:
        comparison['similarities'].append("Both videos have appropriate aspect ratios")
    
    # Compare scene counts
    uploaded_scene_count = len(uploaded_results['scenes'])
    reference_scene_count = len(reference_results['scenes'])
    if abs(uploaded_scene_count - reference_scene_count) > 2:
        comparison['differences'].append(f"Scene count differs significantly: Uploaded has {uploaded_scene_count} scenes, Reference has {reference_scene_count} scenes")
    else:
        comparison['similarities'].append(f"Similar scene structure: Uploaded has {uploaded_scene_count} scenes, Reference has {reference_scene_count} scenes")
    
    # Compare issues
    issues_comparison = [
        ('black_frames', 'Black frames'),
        ('flicker_frames', 'Flicker/flash'),
        ('freeze_frames', 'Frozen frames')
    ]
    
    for key, label in issues_comparison:
        uploaded_issues = len(uploaded_results[key])
        reference_issues = len(reference_results[key])
        
        if uploaded_issues > 0 and reference_issues == 0:
            comparison['differences'].append(f"{label}: Uploaded video has {uploaded_issues} instances, Reference video has none")
        elif uploaded_issues == 0 and reference_issues > 0:
            comparison['differences'].append(f"{label}: Uploaded video has none, Reference video has {reference_issues} instances")
        elif uploaded_issues > 0 and reference_issues > 0:
            if abs(uploaded_issues - reference_issues) > 1:
                comparison['differences'].append(f"{label}: Uploaded video has {uploaded_issues} instances, Reference video has {reference_issues} instances")
            else:
                comparison['similarities'].append(f"{label}: Both videos have similar issues ({uploaded_issues} vs {reference_issues} instances)")
        else:
            comparison['similarities'].append(f"{label}: Neither video has {label.lower()}")
    
    # Compare text content
    uploaded_text_count = len(uploaded_results['all_texts'])
    reference_text_count = len(reference_results['all_texts'])
    if abs(uploaded_text_count - reference_text_count) > 2:
        comparison['differences'].append(f"Text overlay count differs: Uploaded has {uploaded_text_count} text elements, Reference has {reference_text_count} text elements")
    else:
        comparison['similarities'].append(f"Similar text overlay usage: Uploaded has {uploaded_text_count} text elements, Reference has {reference_text_count} text elements")
    
    return comparison

def analyze_video(video_path, description):
    mistakes = []
    # Aspect ratio
    aspect_mistake = analyze_aspect_ratio(video_path)
    if aspect_mistake:
        mistakes.append(("00:00", aspect_mistake))
    # Scene/cut detection (short scenes)
    try:
        scenes = detect_scenes(video_path)
        for start, end in scenes:
            if end - start < 1:
                mistakes.append((f"{start//60:02d}:{start%60:02d}", "Detected very short clip segment (<1s)"))
    except Exception:
        pass
    # Black frame detection
    black_ts = detect_black_frames(video_path)
    for ts in black_ts:
        mistakes.append((ts, "Black frame detected"))
    # Flicker detection
    flicker_ts = detect_flicker(video_path)
    for ts in flicker_ts:
        mistakes.append((ts, "Flicker/flash detected"))
    # Freeze detection
    freeze_ts = detect_freeze(video_path)
    for ts in freeze_ts:
        mistakes.append((ts, "Frozen frame or video freeze detected"))
    # OCR & spell check
    text_mistakes, all_texts = spell_check_texts(video_path, ocr, spell)
    for ts, msg in text_mistakes:
        mistakes.append((ts, msg))
    # Compare to notes
    notes_mismatch = compare_to_notes(all_texts, description)
    mistakes.append(("Notes Check", notes_mismatch))
    return mistakes if mistakes else [("00:00", "No obvious mistakes detected.")]

# --- MAIN LOGIC ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name
    
    st.video(temp_video_path)
    st.write(f"**Your Notes:** {description}")
    
    # Check if reference URL is provided
    reference_video_path = None
    if reference_url and reference_url.strip():
        try:
            with st.spinner("Downloading reference video..."):
                reference_video_path = download_video_from_url(reference_url.strip())
            st.success("Reference video downloaded successfully!")
            st.video(reference_video_path)
        except Exception as e:
            st.error(f"Failed to download reference video: {str(e)}")
            reference_video_path = None
    
    # Analyze uploaded video
    with st.spinner("Analyzing uploaded video with AI... (may take up to 2 minutes)"):
        mistakes = analyze_video(temp_video_path, description)
    
    st.subheader("Detected Mistakes & Content Mismatches (with Timestamps)")
    for ts, mistake in mistakes:
        st.write(f"**[{ts}]** {mistake}")
    
    # If reference video is available, do comparison
    if reference_video_path:
        st.markdown("---")
        st.subheader("Reference Video Comparison")
        
        with st.spinner("Comparing videos... (may take up to 3 minutes)"):
            comparison = compare_videos(temp_video_path, reference_video_path, description)
        
        # Display comparison results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 **Differences Found**")
            if comparison['differences']:
                for diff in comparison['differences']:
                    st.write(f"• {diff}")
            else:
                st.write("No significant differences detected!")
        
        with col2:
            st.markdown("### ✅ **Similarities**")
            if comparison['similarities']:
                for sim in comparison['similarities']:
                    st.write(f"• {sim}")
            else:
                st.write("No notable similarities found.")
        
        # Detailed analysis section
        with st.expander("📊 Detailed Analysis Results"):
            st.markdown("#### Uploaded Video Analysis:")
            uploaded_results = comparison['uploaded']
            st.write(f"- **Scenes detected:** {len(uploaded_results['scenes'])}")
            st.write(f"- **Black frames:** {len(uploaded_results['black_frames'])}")
            st.write(f"- **Flicker instances:** {len(uploaded_results['flicker_frames'])}")
            st.write(f"- **Freeze instances:** {len(uploaded_results['freeze_frames'])}")
            st.write(f"- **Text elements:** {len(uploaded_results['all_texts'])}")
            st.write(f"- **Text errors:** {len(uploaded_results['text_mistakes'])}")
            
            st.markdown("#### Reference Video Analysis:")
            reference_results = comparison['reference']
            st.write(f"- **Scenes detected:** {len(reference_results['scenes'])}")
            st.write(f"- **Black frames:** {len(reference_results['black_frames'])}")
            st.write(f"- **Flicker instances:** {len(reference_results['flicker_frames'])}")
            st.write(f"- **Freeze instances:** {len(reference_results['freeze_frames'])}")
            st.write(f"- **Text elements:** {len(reference_results['all_texts'])}")
            st.write(f"- **Text errors:** {len(reference_results['text_mistakes'])}")
        
        # Clean up reference video
        os.remove(reference_video_path)
    
    # Clean up uploaded video
    os.remove(temp_video_path)
else:
    st.info("Please upload a video and provide detailed notes to see analysis results.")