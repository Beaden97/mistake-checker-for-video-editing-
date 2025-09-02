"""Aspect ratio analysis for video content."""
import time
from typing import Dict, Any, Optional
from .common import safe_capture, timeout_context, get_memory_usage


def analyze_aspect_ratio(video_path: str, timeout_seconds: int = 20) -> Dict[str, Any]:
    """
    Analyze video aspect ratio to check for TikTok vertical format compliance.
    
    Args:
        video_path: Path to the video file
        timeout_seconds: Maximum time to spend on analysis
        
    Returns:
        Dict with analysis results, errors, and timing info
    """
    start_time = time.time()
    start_memory = get_memory_usage()
    
    result = {
        'issues': [],
        'metadata': {},
        'error': None,
        'duration': 0,
        'memory_used': 0
    }
    
    try:
        with timeout_context(timeout_seconds):
            cap = safe_capture(video_path)
            ret, frame = cap.read()
            
            if ret:
                h, w = frame.shape[:2]
                ratio = w / h
                
                result['metadata'] = {
                    'width': w,
                    'height': h,
                    'aspect_ratio': ratio,
                    'is_vertical': ratio < 1.0,
                    'is_tiktok_format': 0.55 < ratio < 0.6  # 9:16 is ~0.5625
                }
                
                # Check for TikTok vertical format
                if not (0.55 < ratio < 0.6):
                    result['issues'].append({
                        'timestamp': '00:00',
                        'type': 'aspect_ratio',
                        'severity': 'warning',
                        'message': f"Aspect ratio is not TikTok vertical (9:16), got {w}:{h} (ratio: {ratio:.3f})"
                    })
            
            cap.release()
            
    except Exception as e:
        result['error'] = str(e)
    
    # Calculate timing and memory usage
    result['duration'] = time.time() - start_time
    end_memory = get_memory_usage()
    if start_memory and end_memory:
        result['memory_used'] = end_memory - start_memory
    
    return result