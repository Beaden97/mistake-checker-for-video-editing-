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
import yt_dlp
from urllib.parse import urlparse
import re
from pathlib import Path

from app_theme import apply_base_theme, apply_runtime_theme_controls
from components.feedback import render_feedback_widget

# Apply theme before any Streamlit output
apply_base_theme(page_title="The Video Editing Mistake Checker", page_icon=None)
_appearance = apply_runtime_theme_controls()

# Add feedback widget
render_feedback_widget()

# --- INSTRUCTIONS SECTION ---
_default_hero = "The Video Editing Mistake Checker"
hero_title = (_appearance or {}).get("hero_title", _default_hero)
st.title(hero_title)
st.markdown("""
Hi! Are you a clumsy video editor who often misses small mistakes on projects? I have inattentive adhd, and I created this tool to help spot the little, clumsy mistakes for our work before it gets sent out to that big boss or client. I often feel rubbish about myself because I send out projects with mistakes I've missed. I'm hoping this little app will help us all feel more confident in the work we produce.
""")

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

# --- URL-BASED ANALYSIS ALTERNATIVE ---
st.markdown("#### Alternative: Analyze from a video URL (for large files)")
video_url_for_analysis = st.text_input(
    "Paste a direct video link or YouTube URL to analyze instead of uploading a file",
    placeholder="https://www.youtube.com/watch?v=... or https://example.com/bigvideo.mp4"
)

analyze_url_button = st.button(
    "🔍 Analyze Video from URL",
    type="secondary",
    help="Use this for large files to avoid uploading"
)

# --- SUBMIT BUTTON ---
st.markdown("---")
# Check if both required fields are filled
can_submit = uploaded_file is not None and description.strip() != "" and description.strip() != "A short dancing clip with text captions. Example: First 5 seconds - dancer enters from left, caption 'Welcome!' appears. Transition to close-up. Audio: upbeat pop, no silence. End with logo."

submit_button = st.button(
    "🔍 Analyze Video", 
    type="primary",
    disabled=not can_submit,
    help="Upload a video and provide a description to enable analysis" if not can_submit else "Click to start video analysis"
)

# --- AI/DEEP LEARNING VIDEO ANALYSIS FUNCTIONS ---
@st.cache_resource
def get_ocr():
    try:
        return PaddleOCR(use_angle_cls=True, lang='en')
    except Exception as e:
        st.warning(f"OCR initialization failed: {str(e)}. Text analysis will be limited.")
        return None

@st.cache_resource
def get_spell():
    try:
        return SpellChecker()
    except Exception as e:
        st.warning(f"Spell checker initialization failed: {str(e)}. Spell checking will be disabled.")
        return None

ocr = get_ocr()
spell = get_spell()

def _safe_capture(video_path):
    """Safely open video capture with error handling."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video. Try MP4 (H.264) or re-encode your file.")
    return cap

def analyze_aspect_ratio(video_path):
    try:
        cap = _safe_capture(video_path)
        ret, frame = cap.read()
        mistake = None
        if ret:
            h, w = frame.shape[:2]
            ratio = w / h
            if not (0.55 < ratio < 0.6):  # 9:16 is ~0.5625
                mistake = f"Aspect ratio is not TikTok vertical (9:16), got {w}:{h}"
        cap.release()
        return mistake
    except Exception as e:
        return f"Could not analyze aspect ratio: {str(e)}"

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
    try:
        cap = _safe_capture(video_path)
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
    except Exception as e:
        print(f"[detect_black_frames] error: {e}")
        return []

def detect_flicker(video_path, threshold=40):
    try:
        cap = _safe_capture(video_path)
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
    except Exception as e:
        print(f"[detect_flicker] error: {e}")
        return []

def detect_freeze(video_path, freeze_threshold=1.0, min_freeze_duration=20):
    try:
        cap = _safe_capture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default fallback to prevent division by zero
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
                # Use absolute difference instead of correlation to avoid NaN issues
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                if mean_diff < freeze_threshold:  # Low difference indicates freeze
                    if freeze_start is None:
                        freeze_start = frame_idx
                        freeze_count = 1
                    else:
                        freeze_count += 1
                else:
                    if freeze_start is not None and freeze_count >= min_freeze_duration:
                        timestamp = int(freeze_start // fps) if fps > 0 else 0
                        freeze_timestamps.append(f"{timestamp//60:02d}:{timestamp%60:02d}")
                    freeze_start = None
                    freeze_count = 0
            prev_frame = gray
            frame_idx += 1
        if freeze_start is not None and freeze_count >= min_freeze_duration:
            timestamp = int(freeze_start // fps) if fps > 0 else 0
            freeze_timestamps.append(f"{timestamp//60:02d}:{timestamp%60:02d}")
        cap.release()
        return list(set(freeze_timestamps))
    except Exception as e:
        print(f"[detect_freeze] error: {e}")
        return []

def spell_check_texts(video_path, ocr, spell, n_samples=5):
    try:
        cap = _safe_capture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        mistakes = []
        all_texts = []
        
        if ocr is None:
            cap.release()
            return mistakes, all_texts
        
        for idx in np.linspace(0, frame_count-1, num=n_samples, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            img_path = f"frame_{idx}.jpg"
            cv2.imwrite(img_path, frame)
            try:
                ocr_results = ocr.ocr(img_path)
                if ocr_results:
                    texts = [l[1][0] for line in ocr_results if line for l in line if l]
                    all_texts.extend(texts)
                    if texts and spell:
                        words = []
                        for text in texts:
                            words.extend([w for w in text.split() if w.isalpha()])
                        misspelled = spell.unknown(words)
                        for word in misspelled:
                            timestamp = int(idx // fps) if fps > 0 else 0
                            mistakes.append((f"{timestamp//60:02d}:{timestamp%60:02d}", f"Potential typo: '{word}'"))
            except Exception:
                pass  # Skip OCR errors
            
            try:
                os.remove(img_path)
            except:
                pass
        cap.release()
        return mistakes, all_texts
    except Exception as e:
        print(f"[spell_check_texts] error: {e}")
        return [], []

def compare_to_notes(all_texts, description):
    """Compare video text to notes with stricter verification for explicit text directives."""
    # First check if there are explicit text directives with positional/time cues
    expected_texts = parse_expected_texts(description)
    
    if not expected_texts:
        return "No explicit on-screen text directives to verify."
    
    # Check if expected texts appear in video
    video_text_combined = ' '.join(all_texts).lower()
    missing_texts = []
    
    for expected in expected_texts:
        if expected.lower() not in video_text_combined:
            missing_texts.append(expected)
    
    if missing_texts:
        return f"Content Mismatch: The following expected texts from your notes were NOT found in the video: {', '.join(missing_texts)}"
    
    return "All explicit text directives from your notes were found in the video text."

def is_photo_compilation(description: str) -> bool:
    """Check if the description indicates a photo slideshow/compilation."""
    keywords = ['photo', 'slideshow', 'gallery', 'ctto', 'compilation']
    desc_lower = description.lower()
    return any(keyword in desc_lower for keyword in keywords)

def parse_expected_texts(description: str) -> list:
    """Extract explicit on-screen text directives with positional/time cues."""
    # Look for text with explicit positional or time markers
    patterns = [
        r'at\s+\d+:\d+.*?["\']([^"\']+)["\']',  # "at 00:30 shows 'text'"
        r'["\']([^"\']+)["\'].*?at\s+\d+:\d+',  # "'text' appears at 00:30"
        r'(start|beginning|middle|end).*?["\']([^"\']+)["\']',  # "start shows 'text'"
        r'["\']([^"\']+)["\'].*(start|beginning|middle|end)',  # "'text' at the start"
        r'caption[s]?\s*["\']([^"\']+)["\']',  # "caption 'text'"
        r'text[s]?\s*["\']([^"\']+)["\']',     # "text 'Welcome!'"
        r'displays?\s*["\']([^"\']+)["\']',    # "displays 'Thanks'"
        r'shows?\s*["\']([^"\']+)["\']',       # "shows 'Welcome'"
    ]
    
    expected_texts = []
    for pattern in patterns:
        matches = re.finditer(pattern, description, re.IGNORECASE)
        for match in matches:
            # Extract the text part (group 1 or 2 depending on pattern)
            if len(match.groups()) >= 2:
                text = match.group(1) if match.group(1) else match.group(2)
            else:
                text = match.group(1)
            if text and len(text.strip()) > 2:
                expected_texts.append(text.strip())
    
    return list(set(expected_texts))  # Remove duplicates

def detect_title_text(video_path, ocr, start_seconds=3, sample_frames=8, size_ratio_threshold=1.4, min_height_ratio=0.03):
    """Detect title text in the opening frames of the video."""
    if ocr is None:
        return {'found': False, 'text': '', 'confidence': 0}
    
    try:
        cap = _safe_capture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default fallback
        
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_height <= 0:
            cap.release()
            return {'found': False, 'text': '', 'confidence': 0}
        
        max_frame = int(start_seconds * fps)
        title_candidates = {}
        
        try:
            for i in range(sample_frames):
                frame_idx = int(i * max_frame / sample_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                img_path = f"title_frame_{frame_idx}.jpg"
                cv2.imwrite(img_path, frame)
                
                try:
                    ocr_results = ocr.ocr(img_path)
                    if ocr_results:
                        frame_text_heights = []
                        for line in ocr_results:
                            if line:
                                for text_info in line:
                                    if text_info and len(text_info) >= 2:
                                        bbox = text_info[0]
                                        text = text_info[1][0] if isinstance(text_info[1], tuple) else text_info[1]
                                        
                                        # Calculate text height from bounding box
                                        if len(bbox) >= 4:
                                            y_coords = [point[1] for point in bbox]
                                            text_height = max(y_coords) - min(y_coords)
                                            frame_text_heights.append(text_height)
                                            
                                            # Check if this text is significantly larger
                                            text_height_ratio = text_height / frame_height
                                            if (text_height_ratio > min_height_ratio and 
                                                len(text.strip()) > 2 and 
                                                text_height > np.median(frame_text_heights) * size_ratio_threshold):
                                                
                                                if text in title_candidates:
                                                    title_candidates[text] += 1
                                                else:
                                                    title_candidates[text] = 1
                    
                    os.remove(img_path)
                except Exception:
                    try:
                        os.remove(img_path)
                    except:
                        pass
                    continue
        finally:
            cap.release()
        
        if title_candidates:
            # Return the most voted title
            best_title = max(title_candidates, key=title_candidates.get)
            confidence = title_candidates[best_title] / sample_frames
            return {'found': True, 'text': best_title, 'confidence': confidence}
        
        return {'found': False, 'text': '', 'confidence': 0}
    except Exception as e:
        print(f"[detect_title_text] error: {e}")
        return {'found': False, 'text': '', 'confidence': 0}

def detect_credit_lines_and_typos(video_path, ocr, spell):
    """Detect credit lines and check for typos in them."""
    result = {
        'found': False,
        'lines': [],
        'mistakes': [],
        'best_line': '',
        'timestamps': []
    }
    
    if ocr is None or spell is None:
        return result
    
    try:
        cap = _safe_capture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default fallback
        
        credit_keywords = ['credit', 'credits', 'credit to', 'credit:', 'ctto']
        credit_lines = []
        credit_timestamps = []
        
        try:
            # Sample 10 frames throughout the video
            for idx in np.linspace(0, max(frame_count-1, 0), num=10, dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                img_path = f"credit_frame_{idx}.jpg"
                cv2.imwrite(img_path, frame)
                
                try:
                    ocr_results = ocr.ocr(img_path)
                    if ocr_results:
                        for line in ocr_results:
                            if line:
                                for text_info in line:
                                    if text_info and len(text_info) >= 2:
                                        text = text_info[1][0] if isinstance(text_info[1], tuple) else text_info[1]
                                        text_lower = text.lower()
                                        
                                        # Check if any credit keyword is in this text
                                        if any(keyword in text_lower for keyword in credit_keywords):
                                            credit_lines.append(text)
                                            timestamp = int(idx / fps) if fps > 0 else 0
                                            credit_timestamps.append(f"{timestamp//60:02d}:{timestamp%60:02d}")
                    
                    os.remove(img_path)
                except Exception:
                    try:
                        os.remove(img_path)
                    except:
                        pass
                    continue
        finally:
            cap.release()
        
        if credit_lines:
            result['found'] = True
            result['lines'] = credit_lines
            result['timestamps'] = credit_timestamps
            result['best_line'] = max(credit_lines, key=len) if credit_lines else ''
            
            # Spell check the credit lines
            for i, line in enumerate(credit_lines):
                # Filter out URLs, @handles, hashtags, and alphanumeric tokens
                words = []
                for word in line.split():
                    word_clean = re.sub(r'[^\w]', '', word)
                    if (not word.startswith(('http', 'www', '@', '#')) and 
                        word_clean.isalpha() and 
                        len(word_clean) > 2):
                        words.append(word_clean.lower())
                
                if words:
                    misspelled = spell.unknown(words)
                    for word in misspelled:
                        correction = spell.correction(word)
                        if correction and correction != word:
                            mistake_msg = f"Credit text possible typo: '{word}' → '{correction}'"
                            timestamp = credit_timestamps[i] if i < len(credit_timestamps) else "Credit"
                            result['mistakes'].append((timestamp, mistake_msg))
        
        return result
    except Exception as e:
        print(f"[detect_credit_lines_and_typos] error: {e}")
        return result

def download_video_from_url(url):
    """Download video from URL (YouTube, TikTok, etc.) using yt-dlp and return local path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_path = temp_file.name
        
        # Configure yt-dlp options
        ydl_opts = {
            'outtmpl': temp_path.replace('.mp4', '.%(ext)s'),
            'format': 'best[ext=mp4]/best',  # Prefer mp4, fallback to best available
            'noplaylist': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        # Find the downloaded file (it might have a different extension)
        base_path = temp_path.replace('.mp4', '')
        for ext in ['.mp4', '.webm', '.mkv', '.mov']:
            potential_path = base_path + ext
            if os.path.exists(potential_path):
                return potential_path
                
        # If no file found, fallback to original temp_path
        if os.path.exists(temp_path):
            return temp_path
            
        raise Exception("Downloaded file not found")
                
    except Exception as e:
        raise Exception(f"Failed to download video from URL: {str(e)}")

def get_video_analysis_results_from_url(video_url, description):
    """Get detailed analysis results for a video from URL by downloading it locally first."""
    # Download the video locally
    local_video_path = download_video_from_url(video_url)
    
    try:
        # Run the same analysis pipeline as for uploaded videos
        return get_video_analysis_results(local_video_path, description)
    finally:
        # Clean up the downloaded file
        try:
            os.remove(local_video_path)
        except:
            pass

def compare_videos_with_url(video1_path, video2_url, description):
    """Compare uploaded video (file path) with reference video (URL)."""
    st.write("Analyzing uploaded video...")
    uploaded_results = get_video_analysis_results(video1_path, description)
    
    st.write("Analyzing reference video from URL...")
    reference_results = get_video_analysis_results_from_url(video2_url, description)
    
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
    
    # Compare scene counts (simplified for streaming)
    uploaded_scene_count = len(uploaded_results['scenes'])
    reference_scene_count = len(reference_results['scenes'])
    if reference_scene_count == 0:
        comparison['similarities'].append(f"Scene analysis: Uploaded has {uploaded_scene_count} scenes (reference analysis limited for streaming)")
    elif abs(uploaded_scene_count - reference_scene_count) > 2:
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
            comparison['differences'].append(f"{label}: Uploaded video has {uploaded_issues} instances, Reference video has none (sample-based analysis)")
        elif uploaded_issues == 0 and reference_issues > 0:
            comparison['differences'].append(f"{label}: Uploaded video has none, Reference video has {reference_issues} instances (sample-based analysis)")
        elif uploaded_issues > 0 and reference_issues > 0:
            comparison['similarities'].append(f"{label}: Both videos have issues ({uploaded_issues} vs {reference_issues} instances)")
        else:
            comparison['similarities'].append(f"{label}: Neither video has {label.lower()}")
    
    # Compare text content
    uploaded_text_count = len(uploaded_results['all_texts'])
    reference_text_count = len(reference_results['all_texts'])
    if reference_text_count == 0:
        comparison['similarities'].append(f"Text analysis: Uploaded has {uploaded_text_count} text elements (reference analysis limited for streaming)")
    elif abs(uploaded_text_count - reference_text_count) > 2:
        comparison['differences'].append(f"Text overlay count differs: Uploaded has {uploaded_text_count} text elements, Reference has {reference_text_count} text elements (sample-based)")
    else:
        comparison['similarities'].append(f"Similar text overlay usage: Uploaded has {uploaded_text_count} text elements, Reference has {reference_text_count} text elements")
    
    return comparison

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
    
    # Freeze detection - suppress if photo compilation
    if is_photo_compilation(description):
        results['freeze_frames'] = []  # Suppress freeze detection for photo compilations
    else:
        results['freeze_frames'] = detect_freeze(video_path)
    
    # OCR & spell check
    text_mistakes, all_texts = spell_check_texts(video_path, ocr, spell)
    results['text_mistakes'] = text_mistakes
    results['all_texts'] = all_texts
    
    # Title detection
    title_result = detect_title_text(video_path, ocr)
    results['title_text'] = title_result['text'] if title_result['found'] else ''
    
    # Credit detection
    credit_result = detect_credit_lines_and_typos(video_path, ocr, spell)
    results['credit_detected'] = credit_result['found']
    results['credit_text'] = credit_result['best_line']
    results['credit_text_mistakes'] = credit_result['mistakes']
    
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
    # Freeze detection - suppress if photo compilation
    if not is_photo_compilation(description):
        freeze_ts = detect_freeze(video_path)
        for ts in freeze_ts:
            mistakes.append((ts, "Frozen frame or video freeze detected"))
    # OCR & spell check
    text_mistakes, all_texts = spell_check_texts(video_path, ocr, spell)
    for ts, msg in text_mistakes:
        mistakes.append((ts, msg))
    # Credit detection and spell check
    credit_result = detect_credit_lines_and_typos(video_path, ocr, spell)
    for ts, msg in credit_result['mistakes']:
        mistakes.append((ts, msg))
    # Compare to notes
    notes_mismatch = compare_to_notes(all_texts, description)
    mistakes.append(("Notes Check", notes_mismatch))
    return mistakes if mistakes else [("00:00", "No obvious mistakes detected.")]

# --- MAIN LOGIC ---
# Handle URL-based analysis
if analyze_url_button and video_url_for_analysis.strip():
    try:
        with st.spinner("Downloading and analyzing video... (may take a few minutes for large files)"):
            results = get_video_analysis_results_from_url(video_url_for_analysis.strip(), description)
        
        st.subheader("Detected Issues (URL video)")
        
        # Display results similar to uploaded flow
        mistakes = []
        
        # Aspect ratio
        if results.get('aspect_ratio'):
            mistakes.append(("00:00", results['aspect_ratio']))
        
        # Scenes
        for start, end in results.get('short_scenes', []):
            mistakes.append((f"{start//60:02d}:{start%60:02d}", "Detected very short clip segment (<1s)"))
        
        # Black frames
        for ts in results.get('black_frames', []):
            mistakes.append((ts, "Black frame detected"))
        
        # Flicker
        for ts in results.get('flicker_frames', []):
            mistakes.append((ts, "Flicker/flash detected"))
        
        # Freeze frames
        for ts in results.get('freeze_frames', []):
            mistakes.append((ts, "Frozen frame or video freeze detected"))
        
        # Text mistakes
        for ts, msg in results.get('text_mistakes', []):
            mistakes.append((ts, msg))
        
        # Credit mistakes
        for ts, msg in results.get('credit_text_mistakes', []):
            mistakes.append((ts, msg))
        
        # Notes comparison
        notes_mismatch = compare_to_notes(results.get('all_texts', []), description)
        mistakes.append(("Notes Check", notes_mismatch))
        
        # Check if we have no real issues (excluding Notes Check)
        real_issues = [m for m in mistakes if m[0] != "Notes Check"]
        
        if not real_issues:
            st.success("All good to go!!👍")
        
        if not mistakes:
            mistakes = [("00:00", "No obvious mistakes detected.")]
        
        for ts, mistake in mistakes:
            st.write(f"**[{ts}]** {mistake}")
        
        # Additional analysis results
        st.markdown("---")
        st.subheader("📋 Additional Analysis Results")
        
        if results.get('title_text'):
            st.write(f"**🎬 Detected Title:** {results['title_text']}")
        
        if results.get('credit_detected', False):
            st.write(f"**✅ Credit Detected:** {results.get('credit_text', 'Found credit text')}")
        
        if is_photo_compilation(description):
            st.write("**📸 Photo Compilation Detected:** Freeze detection suppressed for slideshow content")
        
    except Exception as e:
        st.error("Couldn't analyze the video from the URL. Try another link or ensure it's public.")
        print(f"[analyze_from_url] error: {e}")

# Handle file upload analysis
if submit_button and can_submit:
    temp_video_path = None
    try:
        # Preserve original file suffix for better decoder compatibility
        suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
            uploaded_file.seek(0)
            # Read and write in 4MB chunks to avoid memory spikes
            while True:
                chunk = uploaded_file.read(4 * 1024 * 1024)  # 4MB chunks
                if not chunk:
                    break
                temp_video.write(chunk)
            temp_video_path = temp_video.name

        st.video(temp_video_path)
        st.write(f"**Your Notes:** {description}")

        try:
            with st.spinner("Analyzing uploaded video with AI... (may take a few minutes for large files)"):
                mistakes = analyze_video(temp_video_path, description)
            
            st.subheader("Detected Mistakes & Content Mismatches (with Timestamps)")
            
            # Check if we have no real issues (excluding Notes Check)
            real_issues = [m for m in mistakes if m[0] != "Notes Check"]
            
            if not real_issues:
                st.success("All good to go!!👍")
            
            for ts, mistake in mistakes:
                st.write(f"**[{ts}]** {mistake}")

            detailed_results = get_video_analysis_results(temp_video_path, description)
            
            # Show detected title and credit information
            st.markdown("---")
            st.subheader("📋 Additional Analysis Results")
            
            # Title detection
            if detailed_results.get('title_text'):
                st.write(f"**🎬 Detected Title:** {detailed_results['title_text']}")
            
            # Credit detection
            if detailed_results.get('credit_detected', False):
                st.write(f"**✅ Credit Detected:** {detailed_results.get('credit_text', 'Found credit text')}")
                if detailed_results.get('credit_text_mistakes'):
                    st.write("**Credit Text Issues:**")
                    for ts, mistake in detailed_results['credit_text_mistakes']:
                        st.write(f"  - **[{ts}]** {mistake}")
            
            # Photo compilation info
            if is_photo_compilation(description):
                st.write("**📸 Photo Compilation Detected:** Freeze detection suppressed for slideshow content")
            
            # If reference video URL is provided, do comparison using URL directly
            if reference_url and reference_url.strip():
                st.markdown("---")
                st.subheader("Reference Video Comparison")
                
                try:
                    with st.spinner("Analyzing reference video from URL... (may take up to 3 minutes)"):
                        comparison = compare_videos_with_url(temp_video_path, reference_url.strip(), description)
                    
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
                        st.write(f"- **Title text:** {uploaded_results.get('title_text', 'None detected')}")
                        st.write(f"- **Credit text:** {uploaded_results.get('credit_text', 'None detected') if uploaded_results.get('credit_detected', False) else 'None detected'}")
                        
                        st.markdown("#### Reference Video Analysis:")
                        reference_results = comparison['reference']
                        st.write(f"- **Scenes detected:** {len(reference_results['scenes'])}")
                        st.write(f"- **Black frames:** {len(reference_results['black_frames'])}")
                        st.write(f"- **Flicker instances:** {len(reference_results['flicker_frames'])}")
                        st.write(f"- **Freeze instances:** {len(reference_results['freeze_frames'])}")
                        st.write(f"- **Text elements:** {len(reference_results['all_texts'])}")
                        st.write(f"- **Text errors:** {len(reference_results['text_mistakes'])}")
                        st.write(f"- **Title text:** {reference_results.get('title_text', 'None detected')}")
                        st.write(f"- **Credit text:** {reference_results.get('credit_text', 'None detected') if reference_results.get('credit_detected', False) else 'None detected'}")
                
                except Exception as e:
                    st.error(f"Failed to analyze reference video: {str(e)}")
            
        except Exception as e:
            st.error("We hit an error while analyzing your video. Try MP4/H.264 or a shorter clip.")
            print(f"[analyze_video] error: {e}")
    finally:
        # Always clean up temp file, even if analysis fails
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception as e:
                print(f"[cleanup] failed to remove temp file: {e}")

elif not can_submit and not (analyze_url_button and video_url_for_analysis.strip()):
    st.info("Please upload a video and provide detailed notes to enable analysis, or use the URL analysis option for large files.")
elif not (submit_button and can_submit) and not (analyze_url_button and video_url_for_analysis.strip()):
    st.info("Click 'Analyze Video' button to start the analysis, or use 'Analyze Video from URL' for large files.")