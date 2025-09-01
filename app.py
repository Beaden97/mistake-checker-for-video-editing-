import streamlit as st
import tempfile
import cv2
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from paddleocr import PaddleOCR
from spellchecker import SpellChecker
import os
import yt_dlp
import requests
import urllib.parse

# --- INSTRUCTIONS SECTION ---
st.title("AI TikTok Video QA (Deep Learning, Web-Based, No Install)")
st.markdown("""
#### How to Use This App
**Choose your analysis mode:**
- **Single Video Analysis**: Upload your TikTok video and get a detailed analysis of potential editing mistakes.
- **Video Comparison**: Compare your uploaded video to an online reference video for formatting consistency.

1. **Select your analysis mode below.**
2. **Upload your TikTok video.**
3. **For comparison mode**: Provide a URL to a reference video (YouTube or direct MP4 link).
4. **For single analysis**: Describe exactly what your video should be in the notes box.
   - Include as much detail as possible: scenes, text, actions, transitions, effects, audio, timing.
5. The app will analyze your video and provide detailed feedback.
""")

# --- MODE SELECTION ---
analysis_mode = st.radio(
    "Choose Analysis Mode:",
    ["Single Video Analysis", "Video Comparison"],
    help="Single mode analyzes one video for common mistakes. Comparison mode compares your video against a reference video."
)

# --- FILE UPLOAD & INPUTS ---
uploaded_file = st.file_uploader("Upload your TikTok video", type=["mp4", "mov"])

if analysis_mode == "Single Video Analysis":
    description = st.text_area(
        "Describe in detail what the video is supposed to be:",
        "A short dancing clip with text captions. Example: First 5 seconds - dancer enters from left, caption 'Welcome!' appears. Transition to close-up. Audio: upbeat pop, no silence. End with logo."
    )
    reference_url = None
else:  # Video Comparison
    st.markdown("### Reference Video")
    reference_url = st.text_input(
        "Enter URL of reference video to compare against:",
        placeholder="https://youtube.com/watch?v=... or https://example.com/video.mp4",
        help="Supports YouTube URLs and direct MP4 links. This video will be used as a reference for formatting comparison."
    )
    description = None

# --- AI/DEEP LEARNING VIDEO ANALYSIS FUNCTIONS ---
@st.cache_resource
def get_ocr():
    try:
        return PaddleOCR(use_angle_cls=True, lang='en')
    except Exception as e:
        st.warning(f"OCR functionality unavailable: {str(e)}")
        return None

@st.cache_resource
def get_spell():
    return SpellChecker()

try:
    ocr = get_ocr()
except:
    ocr = None
spell = get_spell()

# --- VIDEO DOWNLOAD FUNCTIONS ---
def download_video_from_url(url, output_path):
    """Download video from URL (YouTube or direct link) to specified path."""
    try:
        # Check if it's a direct video link
        if url.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
            st.info("Downloading video from direct link...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            # Assume it's a YouTube or other supported platform URL
            st.info("Downloading video from YouTube/platform...")
            ydl_opts = {
                'format': 'mp4/best[ext=mp4]',
                'outtmpl': output_path,
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return True
            
    except Exception as e:
        st.error(f"Failed to download video: {str(e)}")
        return False

def get_video_info(video_path):
    """Get basic video information for comparison."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return {
        'width': width,
        'height': height,
        'aspect_ratio': width / height,
        'fps': fps,
        'duration': duration,
        'frame_count': frame_count
    }

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
    mistakes = []
    all_texts = []
    
    if ocr is None:
        return mistakes, all_texts
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for idx in np.linspace(0, frame_count-1, num=n_samples, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img_path = f"frame_{idx}.jpg"
        cv2.imwrite(img_path, frame)
        try:
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
        except Exception:
            pass  # Skip if OCR fails for this frame
        try:
            os.remove(img_path)
        except:
            pass
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
    if ocr is not None:
        text_mistakes, all_texts = spell_check_texts(video_path, ocr, spell)
        for ts, msg in text_mistakes:
            mistakes.append((ts, msg))
        # Compare to notes
        if description:
            notes_mismatch = compare_to_notes(all_texts, description)
            mistakes.append(("Notes Check", notes_mismatch))
    else:
        if description:
            mistakes.append(("OCR Check", "Text analysis unavailable - OCR models could not be loaded"))
    return mistakes if mistakes else [("00:00", "No obvious mistakes detected.")]

# --- VIDEO COMPARISON FUNCTIONS ---
def compare_videos(video1_path, video2_path):
    """Compare two videos and return formatting differences."""
    differences = []
    
    # Get basic video info
    info1 = get_video_info(video1_path)
    info2 = get_video_info(video2_path)
    
    if not info1 or not info2:
        return [("Error", "Could not analyze one or both videos")]
    
    # Compare aspect ratios
    ratio_diff = abs(info1['aspect_ratio'] - info2['aspect_ratio'])
    if ratio_diff > 0.1:
        differences.append((
            "Aspect Ratio", 
            f"Significant difference - Your video: {info1['width']}x{info1['height']} (ratio: {info1['aspect_ratio']:.3f}), Reference: {info2['width']}x{info2['height']} (ratio: {info2['aspect_ratio']:.3f})"
        ))
    else:
        differences.append((
            "Aspect Ratio", 
            f"✓ Similar - Your video: {info1['width']}x{info1['height']}, Reference: {info2['width']}x{info2['height']}"
        ))
    
    # Compare duration
    duration_diff = abs(info1['duration'] - info2['duration'])
    if duration_diff > 2:  # More than 2 seconds difference
        differences.append((
            "Duration", 
            f"Significant difference - Your video: {info1['duration']:.1f}s, Reference: {info2['duration']:.1f}s (diff: {duration_diff:.1f}s)"
        ))
    else:
        differences.append((
            "Duration", 
            f"✓ Similar - Your video: {info1['duration']:.1f}s, Reference: {info2['duration']:.1f}s"
        ))
    
    # Compare scene structures
    try:
        scenes1 = detect_scenes(video1_path)
        scenes2 = detect_scenes(video2_path)
        
        scene_count_diff = abs(len(scenes1) - len(scenes2))
        if scene_count_diff > 2:
            differences.append((
                "Scene Structure", 
                f"Different scene counts - Your video: {len(scenes1)} scenes, Reference: {len(scenes2)} scenes"
            ))
        else:
            differences.append((
                "Scene Structure", 
                f"✓ Similar scene counts - Your video: {len(scenes1)} scenes, Reference: {len(scenes2)} scenes"
            ))
            
        # Compare average scene lengths
        if scenes1 and scenes2:
            avg_scene1 = np.mean([end - start for start, end in scenes1])
            avg_scene2 = np.mean([end - start for start, end in scenes2])
            scene_length_diff = abs(avg_scene1 - avg_scene2)
            
            if scene_length_diff > 2:
                differences.append((
                    "Scene Timing", 
                    f"Different average scene lengths - Your video: {avg_scene1:.1f}s, Reference: {avg_scene2:.1f}s"
                ))
            else:
                differences.append((
                    "Scene Timing", 
                    f"✓ Similar scene timing - Your video: {avg_scene1:.1f}s avg, Reference: {avg_scene2:.1f}s avg"
                ))
    except Exception:
        differences.append(("Scene Analysis", "Could not compare scene structures"))
    
    # Compare black frames
    black1 = detect_black_frames(video1_path)
    black2 = detect_black_frames(video2_path)
    
    if len(black1) > len(black2) + 1:
        differences.append((
            "Black Frames", 
            f"Your video has more black frames ({len(black1)}) than reference ({len(black2)})"
        ))
    elif len(black2) > len(black1) + 1:
        differences.append((
            "Black Frames", 
            f"Reference has more black frames ({len(black2)}) than your video ({len(black1)})"
        ))
    else:
        differences.append((
            "Black Frames", 
            f"✓ Similar black frame counts - Your video: {len(black1)}, Reference: {len(black2)}"
        ))
    
    # Compare flicker detection
    flicker1 = detect_flicker(video1_path)
    flicker2 = detect_flicker(video2_path)
    
    if len(flicker1) > len(flicker2) + 2:
        differences.append((
            "Flicker/Flash", 
            f"Your video has more flicker issues ({len(flicker1)}) than reference ({len(flicker2)})"
        ))
    elif len(flicker2) > len(flicker1) + 2:
        differences.append((
            "Flicker/Flash", 
            f"Reference has more flicker issues ({len(flicker2)}) than your video ({len(flicker1)})"
        ))
    else:
        differences.append((
            "Flicker/Flash", 
            f"✓ Similar flicker levels - Your video: {len(flicker1)}, Reference: {len(flicker2)}"
        ))
    
    # Compare text content (basic OCR comparison)
    if ocr is not None:
        try:
            _, texts1 = spell_check_texts(video1_path, ocr, spell, n_samples=3)
            _, texts2 = spell_check_texts(video2_path, ocr, spell, n_samples=3)
            
            if texts1 and texts2:
                differences.append((
                    "Text Content", 
                    f"Text detected in both videos - Your video: {len(texts1)} text elements, Reference: {len(texts2)} text elements"
                ))
            elif texts1 and not texts2:
                differences.append((
                    "Text Content", 
                    f"Your video has text overlays ({len(texts1)} elements) but reference doesn't"
                ))
            elif texts2 and not texts1:
                differences.append((
                    "Text Content", 
                    f"Reference has text overlays ({len(texts2)} elements) but your video doesn't"
                ))
            else:
                differences.append((
                    "Text Content", 
                    "✓ Neither video has detectable text overlays"
                ))
        except Exception:
            differences.append(("Text Analysis", "Could not compare text content"))
    else:
        differences.append(("Text Analysis", "Text comparison unavailable - OCR models could not be loaded"))
    
    return differences

# --- MAIN LOGIC ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name
    
    st.video(temp_video_path)
    
    if analysis_mode == "Single Video Analysis":
        # Original single video analysis
        st.write(f"**Your Notes:** {description}")
        with st.spinner("Analyzing video with AI... (may take up to 2 minutes)"):
            mistakes = analyze_video(temp_video_path, description)
        
        st.subheader("Detected Mistakes & Content Mismatches (with Timestamps)")
        for ts, mistake in mistakes:
            st.write(f"**[{ts}]** {mistake}")
            
    else:  # Video Comparison mode
        if not reference_url:
            st.warning("Please enter a reference video URL for comparison.")
        else:
            # Download reference video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_ref_video:
                temp_ref_path = temp_ref_video.name
            
            with st.spinner("Downloading reference video..."):
                if download_video_from_url(reference_url, temp_ref_path):
                    st.success("Reference video downloaded successfully!")
                    
                    # Show reference video info
                    ref_info = get_video_info(temp_ref_path)
                    if ref_info:
                        st.write("**Reference Video Info:**")
                        st.write(f"- Resolution: {ref_info['width']}x{ref_info['height']}")
                        st.write(f"- Duration: {ref_info['duration']:.1f} seconds")
                        st.write(f"- Aspect Ratio: {ref_info['aspect_ratio']:.3f}")
                    
                    # Perform comparison
                    with st.spinner("Comparing videos... (may take up to 3 minutes)"):
                        differences = compare_videos(temp_video_path, temp_ref_path)
                    
                    st.subheader("Video Comparison Results")
                    st.markdown("**Formatting Comparison Summary:**")
                    
                    for category, difference in differences:
                        if difference.startswith("✓"):
                            st.success(f"**{category}:** {difference}")
                        elif "Error" in category or "Could not" in difference:
                            st.error(f"**{category}:** {difference}")
                        else:
                            st.warning(f"**{category}:** {difference}")
                    
                    # Clean up reference video
                    try:
                        os.remove(temp_ref_path)
                    except:
                        pass
                else:
                    st.error("Failed to download reference video. Please check the URL and try again.")
    
    # Clean up uploaded video
    os.remove(temp_video_path)
    
else:
    if analysis_mode == "Single Video Analysis":
        st.info("Please upload a video and provide detailed notes to see analysis results.")
    else:
        st.info("Please upload your video and provide a reference video URL for comparison.")