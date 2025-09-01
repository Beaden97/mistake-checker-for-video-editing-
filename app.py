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
import difflib
from wordfreq import top_n_list
import json
import datetime

# --- CONFIGURATION ---
LANGUAGETOOL_API_URL = os.getenv("LANGUAGETOOL_API_URL", "https://api.languagetool.org/v2/check")
LANGUAGETOOL_API_KEY = os.getenv("LANGUAGETOOL_API_KEY", "")
LANGUAGETOOL_LANG = os.getenv("LANGUAGETOOL_LANG", "en-US")

# --- HELPER FUNCTIONS ---
def is_credit_text(text):
    """Determine if text is likely a credit line."""
    credit_indicators = ['credit', 'credits', '@', 'by:', 'created by', 'made by', 'from:', 'original']
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in credit_indicators)

def call_languagetool_api(text, use_online=True):
    """Call LanguageTool API for grammar checking."""
    if not use_online:
        return []
    
    try:
        payload = {
            'text': text,
            'language': LANGUAGETOOL_LANG
        }
        if LANGUAGETOOL_API_KEY:
            payload['apikey'] = LANGUAGETOOL_API_KEY
        
        response = requests.post(
            LANGUAGETOOL_API_URL,
            data=payload,
            timeout=5  # 5 second timeout
        )
        response.raise_for_status()
        
        result = response.json()
        suggestions = []
        
        for match in result.get('matches', []):
            if match.get('replacements'):
                suggestions.append({
                    'original': text[match['offset']:match['offset'] + match['length']],
                    'suggestion': match['replacements'][0]['value'],
                    'message': match.get('message', ''),
                    'offset': match['offset']
                })
        
        return suggestions
    except Exception as e:
        st.warning(f"LanguageTool API call failed: {str(e)}")
        return []

def get_offline_suggestions(text):
    """Get offline grammar/spelling suggestions using difflib and wordfreq."""
    try:
        words = text.split()
        suggestions = []
        common_words = set(top_n_list('en', 10000))  # Top 10k English words
        
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word and clean_word not in common_words and len(clean_word) > 3:
                # Find close matches
                matches = difflib.get_close_matches(clean_word, common_words, n=1, cutoff=0.6)
                if matches:
                    suggestions.append({
                        'original': word,
                        'suggestion': matches[0],
                        'message': 'Possible spelling error (offline suggestion)',
                        'offset': len(' '.join(words[:i]))
                    })
        
        return suggestions
    except Exception as e:
        st.warning(f"Offline suggestion generation failed: {str(e)}")
        return []

def grammarly_like_check_credit_line(text, use_online=True, cfg=None):
    """Check credit line for grammar and spelling issues."""
    result = {
        'suggestions': [],
        'errors': []
    }
    
    # Try LanguageTool first if enabled
    if use_online:
        online_suggestions = call_languagetool_api(text, use_online)
        result['suggestions'].extend(online_suggestions)
    
    # If no online suggestions or offline mode, use offline fallback
    if not result['suggestions'] or not use_online:
        offline_suggestions = get_offline_suggestions(text)
        # Merge, preferring online suggestions
        existing_originals = {s['original'] for s in result['suggestions']}
        for suggestion in offline_suggestions:
            if suggestion['original'] not in existing_originals:
                result['suggestions'].append(suggestion)
    
    return result

def build_checklist_items(mistakes, credit_grammar_warnings, results=None):
    """Build checklist items from analysis results."""
    checklist_items = []
    item_id = 0
    
    # Process mistakes
    for timestamp, message in mistakes:
        if message != "No obvious mistakes detected.":
            checklist_items.append({
                'id': f"item_{item_id}",
                'label': f"[{timestamp}] {message}",
                'checked': False,
                'timestamp': timestamp,
                'type': 'mistake'
            })
            item_id += 1
    
    # Process credit grammar warnings separately if not already in mistakes
    for warning in credit_grammar_warnings:
        item_found = False
        for item in checklist_items:
            if warning['original'] in item['label'] and warning['suggestion'] in item['label']:
                item_found = True
                break
        
        if not item_found:
            checklist_items.append({
                'id': f"item_{item_id}",
                'label': f"[{warning['timestamp']}] Grammar: '{warning['original']}' → '{warning['suggestion']}'",
                'checked': False,
                'timestamp': warning['timestamp'],
                'type': 'grammar'
            })
            item_id += 1
    
    return checklist_items

def export_checklist_to_markdown(checklist_items, video_name="video"):
    """Export checklist to markdown format."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = f"""# Fix Checklist for {video_name}
Generated on: {timestamp}

## Issues to Fix

"""
    
    for item in checklist_items:
        status = "✅" if item['checked'] else "❌"
        md_content += f"- [{status}] {item['label']}\n"
    
    md_content += f"""
## Summary
- Total items: {len(checklist_items)}
- Completed: {sum(1 for item in checklist_items if item['checked'])}
- Remaining: {sum(1 for item in checklist_items if not item['checked'])}
"""
    
    return md_content

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

# --- GRAMMAR CHECKING OPTIONS ---
st.markdown("---")
st.subheader("Grammar Checking Options")
use_online_grammar = st.toggle(
    "Use online grammar check (LanguageTool)",
    value=True,
    help="When enabled, detected credit text will be sent to LanguageTool API for grammar analysis"
)

if use_online_grammar:
    st.info("ℹ️ **Privacy Notice**: Detected 'credit' text will be sent to the configured grammar API for analysis.")

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

def spell_check_texts(video_path, ocr, spell, n_samples=5, use_online_grammar=True):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    mistakes = []
    all_texts = []
    credit_grammar_warnings = []
    
    if ocr is None:
        cap.release()
        return mistakes, all_texts, credit_grammar_warnings
    
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
                        # Check if this is credit text for grammar checking
                        if is_credit_text(text):
                            grammar_result = grammarly_like_check_credit_line(text, use_online_grammar)
                            for suggestion in grammar_result['suggestions']:
                                timestamp = int(idx // fps) if fps > 0 else 0
                                credit_grammar_warnings.append({
                                    'original': suggestion['original'],
                                    'suggestion': suggestion['suggestion'],
                                    'message': suggestion['message'],
                                    'timestamp': f"{timestamp//60:02d}:{timestamp%60:02d}",
                                    'offset': suggestion.get('offset', 0)
                                })
                                mistakes.append((f"{timestamp//60:02d}:{timestamp%60:02d}", 
                                               f"Credit text grammar suggestion: '{suggestion['original']}' → '{suggestion['suggestion']}'"))
                        
                        # Continue with regular spell checking
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
    return mistakes, all_texts, credit_grammar_warnings

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
            # Use yt-dlp for YouTube videos
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                ydl_opts = {
                    'format': 'best[ext=mp4]',
                    'outtmpl': temp_file.name,
                    'quiet': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
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

def get_video_analysis_results_from_url(video_url, description):
    """Get detailed analysis results for a video from URL (used for comparison)."""
    results = {}
    
    # Create a temporary VideoCapture object from URL
    cap = cv2.VideoCapture(video_url)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video from URL: {video_url}")
    
    try:
        # For URL-based videos, we'll do a simplified analysis
        # Aspect ratio analysis
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            ratio = w / h
            if not (0.55 < ratio < 0.6):  # 9:16 is ~0.5625
                results['aspect_ratio'] = f"Aspect ratio is not TikTok vertical (9:16), got {w}:{h}"
            else:
                results['aspect_ratio'] = None
        else:
            results['aspect_ratio'] = "Could not read video frame"
        
        # For streaming videos, we'll do basic analysis only
        # Scene detection is complex for streaming, so we'll skip it
        results['scenes'] = []
        results['short_scenes'] = []
        
        # Basic frame analysis - sample a few frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        black_frames = []
        flicker_frames = []
        freeze_frames = []
        
        # Sample 5 frames for basic analysis
        sample_indices = np.linspace(0, max(frame_count-1, 0), num=min(5, frame_count), dtype=int)
        prev_mean = None
        prev_frame = None
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # Check for black frames
            if mean_brightness < 15:
                timestamp = int(idx // fps) if fps > 0 else 0
                black_frames.append(f"{timestamp//60:02d}:{timestamp%60:02d}")
            
            # Check for flicker
            if prev_mean is not None and abs(mean_brightness - prev_mean) > 40:
                timestamp = int(idx // fps) if fps > 0 else 0
                flicker_frames.append(f"{timestamp//60:02d}:{timestamp%60:02d}")
            
            # Check for freeze (simplified)
            if prev_frame is not None:
                similarity = np.corrcoef(prev_frame.flatten(), gray.flatten())[0,1]
                if similarity > 0.99:
                    timestamp = int(idx // fps) if fps > 0 else 0
                    freeze_frames.append(f"{timestamp//60:02d}:{timestamp%60:02d}")
            
            prev_mean = mean_brightness
            prev_frame = gray
        
        results['black_frames'] = list(set(black_frames))
        results['flicker_frames'] = list(set(flicker_frames))
        results['freeze_frames'] = list(set(freeze_frames))
        
        # For OCR analysis on streaming video, we'll do a very basic check
        # Sample 2 frames for text analysis
        text_mistakes = []
        all_texts = []
        
        if ocr is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)  # Middle frame
            ret, frame = cap.read()
            if ret:
                img_path = f"temp_stream_frame.jpg"
                cv2.imwrite(img_path, frame)
                try:
                    ocr_results = ocr.ocr(img_path)
                    if ocr_results:
                        texts = [l[1][0] for line in ocr_results if line for l in line if l]
                        all_texts.extend(texts)
                        
                        # Basic spell check
                        if texts and spell:
                            words = []
                            for text in texts:
                                words.extend([w for w in text.split() if w.isalpha()])
                            misspelled = spell.unknown(words)
                            for word in misspelled:
                                text_mistakes.append(("Stream sample", f"Potential typo: '{word}'"))
                    
                    os.remove(img_path)
                except Exception:
                    pass  # Skip OCR errors for streaming
        
        results['text_mistakes'] = text_mistakes
        results['all_texts'] = all_texts
        results['credit_grammar_warnings'] = []  # Simplified for streaming
        
    finally:
        cap.release()
    
    return results

def compare_videos_with_url(video1_path, video2_url, description, use_online_grammar=True):
    """Compare uploaded video (file path) with reference video (URL)."""
    st.write("Analyzing uploaded video...")
    uploaded_results = get_video_analysis_results(video1_path, description, use_online_grammar)
    
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

def get_video_analysis_results(video_path, description, use_online_grammar=True):
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
    
    # OCR & spell check with grammar checking
    text_mistakes, all_texts, credit_grammar_warnings = spell_check_texts(video_path, ocr, spell, use_online_grammar=use_online_grammar)
    results['text_mistakes'] = text_mistakes
    results['all_texts'] = all_texts
    results['credit_grammar_warnings'] = credit_grammar_warnings
    
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

def analyze_video(video_path, description, use_online_grammar=True):
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
    # OCR & spell check with grammar checking
    text_mistakes, all_texts, credit_grammar_warnings = spell_check_texts(video_path, ocr, spell, use_online_grammar=use_online_grammar)
    for ts, msg in text_mistakes:
        mistakes.append((ts, msg))
    # Compare to notes
    notes_mismatch = compare_to_notes(all_texts, description)
    mistakes.append(("Notes Check", notes_mismatch))
    
    # Return results structure with credit grammar warnings
    return {
        'mistakes': mistakes if mistakes else [("00:00", "No obvious mistakes detected.")],
        'credit_grammar_warnings': credit_grammar_warnings,
        'all_texts': all_texts
    }

# --- MAIN LOGIC ---
if submit_button and uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name
    
    st.video(temp_video_path)
    st.write(f"**Your Notes:** {description}")
    
    # Analyze uploaded video
    with st.spinner("Analyzing uploaded video with AI... (may take up to 2 minutes)"):
        analysis_results = analyze_video(temp_video_path, description, use_online_grammar)
    
    mistakes = analysis_results['mistakes']
    credit_grammar_warnings = analysis_results['credit_grammar_warnings']
    
    st.subheader("Detected Mistakes & Content Mismatches (with Timestamps)")
    for ts, mistake in mistakes:
        st.write(f"**[{ts}]** {mistake}")
    
    # If reference video URL is provided, do comparison using URL directly
    if reference_url and reference_url.strip():
        st.markdown("---")
        st.subheader("Reference Video Comparison")
        
        try:
            with st.spinner("Analyzing reference video from URL... (may take up to 3 minutes)"):
                comparison = compare_videos_with_url(temp_video_path, reference_url.strip(), description, use_online_grammar)
            
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
                
                # Show credit grammar warnings
                if uploaded_results.get('credit_grammar_warnings'):
                    st.write(f"- **Credit grammar warnings:** {len(uploaded_results['credit_grammar_warnings'])}")
                    grammar_method = "LanguageTool (online)" if use_online_grammar else "Offline suggestions"
                    st.write(f"  - Grammar checking method: {grammar_method}")
                
                st.markdown("#### Reference Video Analysis:")
                reference_results = comparison['reference']
                st.write(f"- **Scenes detected:** {len(reference_results['scenes'])}")
                st.write(f"- **Black frames:** {len(reference_results['black_frames'])}")
                st.write(f"- **Flicker instances:** {len(reference_results['flicker_frames'])}")
                st.write(f"- **Freeze instances:** {len(reference_results['freeze_frames'])}")
                st.write(f"- **Text elements:** {len(reference_results['all_texts'])}")
                st.write(f"- **Text errors:** {len(reference_results['text_mistakes'])}")
        
        except Exception as e:
            st.error(f"Failed to analyze reference video: {str(e)}")
    
    # Fix Checklist UI
    st.markdown("---")
    st.subheader("Fix Checklist")
    
    # Initialize session state for checklist
    if 'checklist_items' not in st.session_state:
        st.session_state.checklist_items = build_checklist_items(mistakes, credit_grammar_warnings)
    
    # Checklist controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Mark all complete"):
            for item in st.session_state.checklist_items:
                item['checked'] = True
    
    with col2:
        if st.button("Reset checklist"):
            for item in st.session_state.checklist_items:
                item['checked'] = False
    
    with col3:
        # Download checklist button
        if st.button("Download checklist"):
            video_name = uploaded_file.name if uploaded_file else "video"
            markdown_content = export_checklist_to_markdown(st.session_state.checklist_items, video_name)
            st.download_button(
                label="Download as Markdown",
                data=markdown_content,
                file_name=f"fix_checklist_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    # Display checklist items
    if st.session_state.checklist_items:
        st.write("Check off items as you fix them:")
        for item in st.session_state.checklist_items:
            # Use session state to track checkbox states
            checked = st.checkbox(
                item['label'],
                value=item['checked'],
                key=item['id']
            )
            # Update the item's checked state
            item['checked'] = checked
    else:
        st.write("No issues found to add to checklist!")
    
    # Show checklist summary
    total_items = len(st.session_state.checklist_items)
    completed_items = sum(1 for item in st.session_state.checklist_items if item['checked'])
    remaining_items = total_items - completed_items
    
    if total_items > 0:
        st.write(f"**Progress:** {completed_items}/{total_items} items completed ({remaining_items} remaining)")
        progress = completed_items / total_items
        st.progress(progress)
    
    # Clean up uploaded video
    os.remove(temp_video_path)
elif not can_submit:
    st.info("Please upload a video and provide detailed notes to enable analysis.")
else:
    st.info("Click 'Analyze Video' button to start the analysis.")