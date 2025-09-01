import streamlit as st
import tempfile
import cv2
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from paddleocr import PaddleOCR
from spellchecker import SpellChecker
import os

# --- INSTRUCTIONS SECTION ---
st.markdown(
    """
    <style>
        .stApp {background-color: #800080;}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("AI TikTok Video QA (Deep Learning, Web-Based, No Install)")
st.markdown("""
#### How to Use This App
1. **Upload your TikTok video below.**
2. **Describe exactly what your video should be** in the notes box.  
   - Please include as much detail as possible!  
   - For example: *What scenes should appear? What text should be shown? What actions, transitions, or effects are expected? What should the audio sound like? What timing is important?*
3. The app will analyze your video for common editing mistakes **and compare it to your notes** to find any mismatches.
""")

# --- FILE UPLOAD & DESCRIPTION ---
uploaded_file = st.file_uploader("Upload your TikTok video", type=["mp4", "mov"])
description = st.text_area(
    "Describe in detail what the video is supposed to be:",
    "A short dancing clip with text captions. Example: First 5 seconds - dancer enters from left, caption 'Welcome!' appears. Transition to close-up. Audio: upbeat pop, no silence. End with logo."
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
    with st.spinner("Analyzing video with AI... (may take up to 2 minutes)"):
        mistakes = analyze_video(temp_video_path, description)
    st.subheader("Detected Mistakes & Content Mismatches (with Timestamps)")
    for ts, mistake in mistakes:
        st.write(f"**[{ts}]** {mistake}")
    os.remove(temp_video_path)
else:
    st.info("Please upload a video and provide detailed notes to see analysis results.")