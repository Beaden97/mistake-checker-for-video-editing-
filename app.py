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

# --- ANALYSIS SETTINGS SIDEBAR ---
st.sidebar.header("⚙️ Analysis Settings")

st.sidebar.subheader("Scene Cut Detection")
cut_sensitivity = st.sidebar.selectbox(
    "Cut Detection Sensitivity",
    ["low", "medium", "high"],
    index=1,  # Default to medium
    help="Higher sensitivity detects more cuts but may produce false positives"
)

st.sidebar.subheader("Text Analysis")
bleed_tolerance = st.sidebar.slider(
    "Bleed Tolerance (seconds)",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Maximum time text can linger after a cut before being flagged as bleeding"
)

flash_min_duration = st.sidebar.slider(
    "Flash Min Duration (seconds)",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum duration for text to not be considered flashing"
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

def detect_scene_cuts(video_path, sensitivity='medium', is_photo_comp=False):
    """Lightweight scene cut detection using histogram differences.
    
    Args:
        video_path: Path to video file
        sensitivity: 'low', 'medium', 'high' - affects threshold
        is_photo_comp: If True, reduces sensitivity for slideshow transitions
        
    Returns:
        List of cut timestamps with confidence: [(timestamp_seconds, confidence_delta), ...]
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return []
    
    # Set thresholds based on sensitivity and photo compilation
    thresholds = {
        'low': 0.7,
        'medium': 0.5, 
        'high': 0.3
    }
    base_threshold = thresholds.get(sensitivity, 0.5)
    
    # Relax threshold for photo compilations
    if is_photo_comp:
        base_threshold *= 1.5  # Make less sensitive
    
    cuts = []
    prev_hist = None
    frame_idx = 0
    
    # Sample every n-th frame for performance (aim for ~10-15 fps sampling)
    sample_rate = max(1, int(fps // 12))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only process every sample_rate-th frame
        if frame_idx % sample_rate == 0:
            # Convert to HSV and compute histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            
            # Normalize histogram
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            if prev_hist is not None:
                # Calculate histogram correlation (higher = more similar)
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                delta = 1.0 - correlation  # Convert to difference (higher = more different)
                
                # Detect cut if delta exceeds threshold
                if delta > base_threshold:
                    timestamp = frame_idx / fps
                    cuts.append((timestamp, delta))
            
            prev_hist = hist
            
        frame_idx += 1
    
    cap.release()
    return cuts

def detect_scenes(video_path):
    """Legacy function for backward compatibility."""
    cuts = detect_scene_cuts(video_path)
    # Convert cuts to scene segments for compatibility
    scenes = []
    last_end = 0
    for cut_time, _ in cuts:
        scenes.append((last_end, int(cut_time)))
        last_end = int(cut_time)
    
    # Add final scene if video has content after last cut
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if last_end < duration - 1:
        scenes.append((last_end, int(duration)))
    
    return scenes

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

def build_text_timeline(video_path, ocr, n_samples=20):
    """Build detailed text timeline from OCR sampling.
    
    Returns:
        text_segments: List of [text_string, start_time, end_time, confidence, bbox]
        all_ocr_samples: Raw OCR results with timestamps
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    all_ocr_samples = []
    
    if ocr is None or fps <= 0:
        cap.release()
        return [], []
    
    # Sample more densely for better text timeline construction
    for idx in np.linspace(0, frame_count-1, num=n_samples, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        timestamp = idx / fps
        img_path = f"frame_{idx}.jpg"
        cv2.imwrite(img_path, frame)
        
        try:
            ocr_results = ocr.ocr(img_path)
            if ocr_results:
                for line in ocr_results:
                    if line:
                        for detection in line:
                            if detection and len(detection) >= 2:
                                bbox = detection[0]  # Bounding box coordinates
                                text_info = detection[1]  # (text, confidence)
                                if text_info and len(text_info) >= 2:
                                    text = text_info[0]
                                    confidence = text_info[1]
                                    
                                    # Normalize text for grouping
                                    normalized_text = text.lower().strip()
                                    normalized_text = ''.join(c for c in normalized_text if c.isalnum() or c.isspace())
                                    normalized_text = ' '.join(normalized_text.split())
                                    
                                    if normalized_text:  # Only add non-empty text
                                        all_ocr_samples.append({
                                            'text': text,
                                            'normalized_text': normalized_text,
                                            'timestamp': timestamp,
                                            'confidence': confidence,
                                            'bbox': bbox
                                        })
        except Exception:
            pass  # Skip OCR errors
        
        try:
            os.remove(img_path)
        except:
            pass
    
    cap.release()
    
    # Group consecutive samples into segments
    text_segments = []
    if not all_ocr_samples:
        return text_segments, all_ocr_samples
    
    # Sort by timestamp
    all_ocr_samples.sort(key=lambda x: x['timestamp'])
    
    # Group by normalized text with temporal proximity
    current_segment = None
    gap_threshold = 0.5  # seconds
    
    for sample in all_ocr_samples:
        if current_segment is None:
            # Start new segment
            current_segment = {
                'text': sample['text'],
                'normalized_text': sample['normalized_text'],
                'start_time': sample['timestamp'],
                'end_time': sample['timestamp'],
                'confidence': sample['confidence'],
                'bbox': sample['bbox'],
                'samples': [sample]
            }
        elif (sample['normalized_text'] == current_segment['normalized_text'] and 
              sample['timestamp'] - current_segment['end_time'] <= gap_threshold):
            # Extend current segment
            current_segment['end_time'] = sample['timestamp']
            current_segment['samples'].append(sample)
            # Update confidence to average
            confidences = [s['confidence'] for s in current_segment['samples']]
            current_segment['confidence'] = sum(confidences) / len(confidences)
        else:
            # Finalize current segment and start new one
            text_segments.append([
                current_segment['text'],
                current_segment['start_time'],
                current_segment['end_time'],
                current_segment['confidence'],
                current_segment['bbox']
            ])
            
            current_segment = {
                'text': sample['text'],
                'normalized_text': sample['normalized_text'],
                'start_time': sample['timestamp'],
                'end_time': sample['timestamp'],
                'confidence': sample['confidence'],
                'bbox': sample['bbox'],
                'samples': [sample]
            }
    
    # Don't forget the last segment
    if current_segment:
        text_segments.append([
            current_segment['text'],
            current_segment['start_time'],
            current_segment['end_time'],
            current_segment['confidence'],
            current_segment['bbox']
        ])
    
    return text_segments, all_ocr_samples

def spell_check_texts(video_path, ocr, spell, n_samples=5):
    """Legacy function maintaining compatibility while using new timeline builder."""
    text_segments, all_ocr_samples = build_text_timeline(video_path, ocr, n_samples)
    
    mistakes = []
    all_texts = []
    
    # Extract all unique texts for compatibility
    seen_texts = set()
    for segment in text_segments:
        text = segment[0]
        if text not in seen_texts:
            all_texts.append(text)
            seen_texts.add(text)
    
    # Spell check using segments
    if spell and text_segments:
        for segment in text_segments:
            text = segment[0]
            start_time = segment[1]
            
            words = [w for w in text.split() if w.isalpha()]
            misspelled = spell.unknown(words)
            for word in misspelled:
                timestamp_str = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                mistakes.append((timestamp_str, f"Potential typo: '{word}'"))
    
    return mistakes, all_texts

def is_photo_compilation(description):
    """Detect if video is likely a photo compilation/slideshow."""
    description_lower = description.lower()
    slideshow_keywords = [
        'slideshow', 'photo', 'photos', 'compilation', 'images', 
        'pictures', 'gallery', 'album', 'memories', 'montage'
    ]
    return any(keyword in description_lower for keyword in slideshow_keywords)

def detect_text_bleed_across_cuts(text_segments, scene_cuts, bleed_tolerance=0.25):
    """Detect text that bleeds across scene cuts.
    
    Args:
        text_segments: List of [text, start_time, end_time, confidence, bbox]
        scene_cuts: List of (cut_timestamp, confidence) 
        bleed_tolerance: Maximum seconds after cut to consider as bleed
        
    Returns:
        List of mistakes: [(timestamp_str, message, details), ...]
    """
    mistakes = []
    
    for cut_time, cut_confidence in scene_cuts:
        # Find text segments that end shortly after this cut
        for segment in text_segments:
            text, start_time, end_time, confidence, bbox = segment
            
            # Check if text ends within bleed_tolerance after the cut
            # and started before the cut
            if (start_time < cut_time < end_time and 
                end_time - cut_time <= bleed_tolerance and
                end_time - cut_time > 0):
                
                linger_duration = end_time - cut_time
                timestamp_str = f"{int(cut_time//60):02d}:{int(cut_time%60):02d}"
                
                message = f"Text bleeds across cut: '{text[:30]}...'" if len(text) > 30 else f"Text bleeds across cut: '{text}'"
                details = {
                    'type': 'text_bleed',
                    'text': text,
                    'cut_time': cut_time,
                    'linger_duration': linger_duration,
                    'bbox': bbox,
                    'text_start': start_time,
                    'text_end': end_time
                }
                
                mistakes.append((timestamp_str, message, details))
    
    return mistakes

def detect_text_flashing(text_segments, scene_cuts, flash_min_duration=0.25, transition_buffer=0.2):
    """Detect text that flashes briefly.
    
    Args:
        text_segments: List of [text, start_time, end_time, confidence, bbox]
        scene_cuts: List of (cut_timestamp, confidence)
        flash_min_duration: Minimum duration to not be considered flashing
        transition_buffer: Seconds around cuts to ignore (transition periods)
        
    Returns:
        List of mistakes: [(timestamp_str, message, details), ...]
    """
    mistakes = []
    
    # Create set of transition periods around cuts
    transition_periods = []
    for cut_time, _ in scene_cuts:
        transition_periods.append((cut_time - transition_buffer, cut_time + transition_buffer))
    
    for segment in text_segments:
        text, start_time, end_time, confidence, bbox = segment
        duration = end_time - start_time
        
        # Check if duration is too short
        if duration < flash_min_duration:
            # Check if it's NOT in a transition period
            in_transition = False
            for trans_start, trans_end in transition_periods:
                if not (end_time < trans_start or start_time > trans_end):
                    # There's overlap with transition period
                    in_transition = True
                    break
            
            if not in_transition:
                timestamp_str = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                message = f"Text flashes briefly: '{text[:30]}...'" if len(text) > 30 else f"Text flashes briefly: '{text}'"
                details = {
                    'type': 'text_flash',
                    'text': text,
                    'duration': duration,
                    'start_time': start_time,
                    'end_time': end_time,
                    'bbox': bbox
                }
                
                mistakes.append((timestamp_str, message, details))
    
    return mistakes

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
            with tempfile.NamedTemporaryFile(delete=False, suffix='.%(ext)s') as temp_file:
                temp_path = temp_file.name
                
            ydl_opts = {
                'outtmpl': temp_path,
                'format': 'best[ext=mp4]/best',
                'noplaylist': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            # Find the actual downloaded file
            base_path = temp_path.replace('.%(ext)s', '')
            for ext in ['.mp4', '.mkv', '.webm']:
                if os.path.exists(base_path + ext):
                    return base_path + ext
            
            # If no specific extension found, return original path
            return temp_path.replace('.%(ext)s', '.mp4')
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
        
    finally:
        cap.release()
    
    return results

def compare_videos_with_url(video1_path, video2_url, description, cut_sensitivity='medium', bleed_tolerance=0.25, flash_min_duration=0.25):
    """Compare uploaded video (file path) with reference video (URL)."""
    st.write("Analyzing uploaded video...")
    uploaded_results = get_video_analysis_results(video1_path, description, cut_sensitivity, bleed_tolerance, flash_min_duration)
    
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

def get_video_analysis_results(video_path, description, cut_sensitivity='medium', bleed_tolerance=0.25, flash_min_duration=0.25):
    """Get detailed analysis results for a video (used for comparison)."""
    results = {}
    
    # Detect if this is a photo compilation
    is_photo_comp = is_photo_compilation(description)
    results['is_photo_compilation'] = is_photo_comp
    
    # Aspect ratio
    results['aspect_ratio'] = analyze_aspect_ratio(video_path)
    
    # Scene cut detection (new lightweight approach)
    try:
        scene_cuts = detect_scene_cuts(video_path, cut_sensitivity, is_photo_comp)
        results['scene_cuts'] = scene_cuts
        
        # Legacy scene detection for backward compatibility
        scenes = detect_scenes(video_path)
        results['scenes'] = scenes
        results['short_scenes'] = [s for s in scenes if s[1] - s[0] < 1]
    except Exception as e:
        results['scene_cuts'] = []
        results['scenes'] = []
        results['short_scenes'] = []
    
    # Build text timeline
    try:
        text_segments, all_ocr_samples = build_text_timeline(video_path, ocr)
        results['text_timeline'] = text_segments
        results['text_ocr_samples'] = all_ocr_samples
    except Exception as e:
        results['text_timeline'] = []
        results['text_ocr_samples'] = []
        text_segments = []
    
    # Text bleed detection
    try:
        if results['scene_cuts'] and results['text_timeline']:
            text_bleed_mistakes = detect_text_bleed_across_cuts(
                results['text_timeline'], 
                results['scene_cuts'], 
                bleed_tolerance
            )
            results['text_bleed_mistakes'] = text_bleed_mistakes
        else:
            results['text_bleed_mistakes'] = []
    except Exception as e:
        results['text_bleed_mistakes'] = []
    
    # Text flashing detection  
    try:
        if results['text_timeline']:
            text_flash_mistakes = detect_text_flashing(
                results['text_timeline'],
                results['scene_cuts'],
                flash_min_duration
            )
            results['text_flash_mistakes'] = text_flash_mistakes
        else:
            results['text_flash_mistakes'] = []
    except Exception as e:
        results['text_flash_mistakes'] = []
    
    # Black frames
    results['black_frames'] = detect_black_frames(video_path)
    
    # Flicker detection
    results['flicker_frames'] = detect_flicker(video_path)
    
    # Freeze detection
    results['freeze_frames'] = detect_freeze(video_path)
    
    # OCR & spell check (legacy)
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

def analyze_video(video_path, description, cut_sensitivity='medium', bleed_tolerance=0.25, flash_min_duration=0.25):
    mistakes = []
    
    # Detect if this is a photo compilation
    is_photo_comp = is_photo_compilation(description)
    
    # Aspect ratio
    aspect_mistake = analyze_aspect_ratio(video_path)
    if aspect_mistake:
        mistakes.append(("00:00", aspect_mistake))
    
    # Scene cut detection with new approach
    scene_cuts = []
    try:
        scene_cuts = detect_scene_cuts(video_path, cut_sensitivity, is_photo_comp)
        
        # Legacy scene detection for short scenes
        scenes = detect_scenes(video_path)
        for start, end in scenes:
            if end - start < 1:
                mistakes.append((f"{start//60:02d}:{start%60:02d}", "Detected very short clip segment (<1s)"))
    except Exception:
        pass
    
    # Build text timeline for new detections
    text_segments = []
    try:
        text_segments, _ = build_text_timeline(video_path, ocr)
    except Exception:
        pass
    
    # Text bleed detection
    try:
        if scene_cuts and text_segments:
            text_bleed_mistakes = detect_text_bleed_across_cuts(text_segments, scene_cuts, bleed_tolerance)
            for ts, msg, details in text_bleed_mistakes:
                mistakes.append((ts, msg))
    except Exception:
        pass
    
    # Text flashing detection
    try:
        if text_segments:
            text_flash_mistakes = detect_text_flashing(text_segments, scene_cuts, flash_min_duration)
            for ts, msg, details in text_flash_mistakes:
                mistakes.append((ts, msg))
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
    
    # Freeze detection (suppress for photo compilations)
    if not is_photo_comp:
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
if submit_button and uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name
    
    st.video(temp_video_path)
    st.write(f"**Your Notes:** {description}")
    
    # Analyze uploaded video
    with st.spinner("Analyzing uploaded video with AI... (may take up to 2 minutes)"):
        mistakes = analyze_video(temp_video_path, description, cut_sensitivity, bleed_tolerance, flash_min_duration)
    
    # Get detailed analysis results for summary
    analysis_results = get_video_analysis_results(temp_video_path, description, cut_sensitivity, bleed_tolerance, flash_min_duration)
    
    # Analysis Summary
    st.subheader("📊 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Scene Cuts", len(analysis_results.get('scene_cuts', [])))
        st.metric("Black Frames", len(analysis_results.get('black_frames', [])))
    
    with col2:
        st.metric("Text Bleed Issues", len(analysis_results.get('text_bleed_mistakes', [])))
        st.metric("Text Flash Issues", len(analysis_results.get('text_flash_mistakes', [])))
    
    with col3:
        st.metric("Flicker/Flash", len(analysis_results.get('flicker_frames', [])))
        st.metric("Freeze Issues", len(analysis_results.get('freeze_frames', [])))
    
    with col4:
        st.metric("Text Elements", len(analysis_results.get('all_texts', [])))
        st.metric("Spelling Errors", len(analysis_results.get('text_mistakes', [])))
    
    # Show if photo compilation detected
    if analysis_results.get('is_photo_compilation'):
        st.info("📸 Photo compilation detected - adjusted analysis sensitivity")
    
    st.subheader("Detected Mistakes & Content Mismatches (with Timestamps)")
    for ts, mistake in mistakes:
        st.write(f"**[{ts}]** {mistake}")
    
    # If reference video URL is provided, do comparison using URL directly
    if reference_url and reference_url.strip():
        st.markdown("---")
        st.subheader("Reference Video Comparison")
        
        try:
            with st.spinner("Analyzing reference video from URL... (may take up to 3 minutes)"):
                comparison = compare_videos_with_url(temp_video_path, reference_url.strip(), description, cut_sensitivity, bleed_tolerance, flash_min_duration)
            
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
                st.write(f"- **Scene cuts detected:** {len(uploaded_results.get('scene_cuts', []))}")
                st.write(f"- **Black frames:** {len(uploaded_results['black_frames'])}")
                st.write(f"- **Flicker instances:** {len(uploaded_results['flicker_frames'])}")
                st.write(f"- **Freeze instances:** {len(uploaded_results['freeze_frames'])}")
                st.write(f"- **Text elements:** {len(uploaded_results['all_texts'])}")
                st.write(f"- **Text errors:** {len(uploaded_results['text_mistakes'])}")
                st.write(f"- **Text bleed issues:** {len(uploaded_results.get('text_bleed_mistakes', []))}")
                st.write(f"- **Text flash issues:** {len(uploaded_results.get('text_flash_mistakes', []))}")
                
                # Show text timeline summary if available
                if uploaded_results.get('text_timeline'):
                    st.write(f"- **Text timeline segments:** {len(uploaded_results['text_timeline'])}")
                    longest_texts = sorted(uploaded_results['text_timeline'], 
                                         key=lambda x: x[2] - x[1], reverse=True)[:3]
                    if longest_texts:
                        st.write("  - **Longest text segments:**")
                        for text, start, end, conf, bbox in longest_texts:
                            duration = end - start
                            st.write(f"    - '{text[:30]}...' ({duration:.1f}s)" if len(text) > 30 else f"    - '{text}' ({duration:.1f}s)")
                
                # Show scene cuts if available
                if uploaded_results.get('scene_cuts'):
                    st.write("  - **Scene cuts:**")
                    for cut_time, confidence in uploaded_results['scene_cuts'][:5]:  # Show first 5
                        st.write(f"    - {cut_time:.1f}s (confidence: {confidence:.2f})")
                    if len(uploaded_results['scene_cuts']) > 5:
                        st.write(f"    - ... and {len(uploaded_results['scene_cuts']) - 5} more")
                
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
    
    # Clean up uploaded video
    os.remove(temp_video_path)
elif not can_submit:
    st.info("Please upload a video and provide detailed notes to enable analysis.")
else:
    st.info("Click 'Analyze Video' button to start the analysis.")