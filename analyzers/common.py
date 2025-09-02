"""Common utilities for video analysis with timeout and error handling."""
import cv2
import time
import signal
import os
import subprocess
from contextlib import contextmanager
from typing import Generator, Tuple, Any, Dict, Optional


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")


@contextmanager
def timeout_context(seconds: int):
    """Context manager for operation timeout."""
    # Set the signal handler and a timeout alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Disable the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def safe_capture(video_path: str) -> cv2.VideoCapture:
    """Safely open video capture with error handling."""
    if not os.path.exists(video_path):
        raise ValueError(f"Video file does not exist: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video. Try MP4 (H.264) or re-encode your file.")
    return cap


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get basic video information safely."""
    try:
        cap = safe_capture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'aspect_ratio': width / height if height > 0 else 0
        }
    except Exception as e:
        return {
            'error': str(e),
            'fps': 0,
            'frame_count': 0,
            'width': 0,
            'height': 0,
            'duration': 0,
            'aspect_ratio': 0
        }


def frame_generator(video_path: str, step: int = 1, max_frames: Optional[int] = None) -> Generator[Tuple[int, Any], None, None]:
    """Generate frames from video with optional sampling."""
    cap = safe_capture(video_path)
    frame_idx = 0
    yielded_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % step == 0:
                yield frame_idx, frame
                yielded_frames += 1
                
                if max_frames and yielded_frames >= max_frames:
                    break
                    
            frame_idx += 1
    finally:
        cap.release()


def is_cloud_environment() -> bool:
    """Detect if running on Streamlit Cloud or similar cloud environment."""
    # Check for common cloud environment variables
    cloud_indicators = [
        'STREAMLIT_SHARING',
        'HEROKU',
        'VERCEL',
        'NETLIFY',
        'GITHUB_ACTIONS'
    ]
    
    for indicator in cloud_indicators:
        if os.environ.get(indicator):
            return True
    
    # Check if CUDA/GPU is not available (common in cloud)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        return result.returncode != 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return True


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS timestamp."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def get_memory_usage() -> Optional[float]:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return None