"""Freeze detection for video content."""
import time
import cv2
import numpy as np
from typing import Dict, Any, List
from .common import safe_capture, timeout_context, get_memory_usage, format_timestamp


def analyze_freeze(video_path: str, freeze_threshold: float = 1.0, min_freeze_duration: int = 20,
                  timeout_seconds: int = 60) -> Dict[str, Any]:
    """
    Detect frozen frames or video freezes using absolute difference.
    
    Args:
        video_path: Path to the video file
        freeze_threshold: Threshold for frame difference to consider as freeze
        min_freeze_duration: Minimum number of consecutive frames to report as freeze
        timeout_seconds: Maximum time to spend on analysis
        
    Returns:
        Dict with analysis results, errors, and timing info
    """
    start_time = time.time()
    start_memory = get_memory_usage()
    
    result = {
        'issues': [],
        'freeze_sequences': [],
        'metadata': {},
        'error': None,
        'duration': 0,
        'memory_used': 0
    }
    
    try:
        with timeout_context(timeout_seconds):
            cap = safe_capture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default fallback to prevent division by zero
                
            prev_frame = None
            freeze_start = None
            freeze_count = 0
            frame_idx = 0
            total_freeze_frames = 0
            
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
                        total_freeze_frames += 1
                    else:
                        # End of potential freeze sequence
                        if freeze_start is not None and freeze_count >= min_freeze_duration:
                            start_time_sec = freeze_start / fps
                            duration_sec = freeze_count / fps
                            
                            freeze_sequence = {
                                'start': start_time_sec,
                                'duration': duration_sec,
                                'frame_count': freeze_count
                            }
                            result['freeze_sequences'].append(freeze_sequence)
                            
                            result['issues'].append({
                                'timestamp': format_timestamp(start_time_sec),
                                'type': 'freeze',
                                'severity': 'warning',
                                'message': f"Video freeze detected for {duration_sec:.1f}s ({freeze_count} frames)"
                            })
                        
                        freeze_start = None
                        freeze_count = 0
                
                prev_frame = gray
                frame_idx += 1
            
            # Handle case where video ends with a freeze
            if freeze_start is not None and freeze_count >= min_freeze_duration:
                start_time_sec = freeze_start / fps
                duration_sec = freeze_count / fps
                
                freeze_sequence = {
                    'start': start_time_sec,
                    'duration': duration_sec,
                    'frame_count': freeze_count
                }
                result['freeze_sequences'].append(freeze_sequence)
                
                result['issues'].append({
                    'timestamp': format_timestamp(start_time_sec),
                    'type': 'freeze',
                    'severity': 'warning',
                    'message': f"Video freeze detected for {duration_sec:.1f}s ({freeze_count} frames)"
                })
            
            cap.release()
            
            result['metadata'] = {
                'total_frames': frame_idx,
                'freeze_frames': total_freeze_frames,
                'freeze_percentage': (total_freeze_frames / frame_idx * 100) if frame_idx > 0 else 0,
                'freeze_sequences_count': len(result['freeze_sequences']),
                'threshold_used': freeze_threshold,
                'min_duration_frames': min_freeze_duration,
                'fps': fps
            }
            
    except Exception as e:
        result['error'] = str(e)
    
    # Calculate timing and memory usage
    result['duration'] = time.time() - start_time
    end_memory = get_memory_usage()
    if start_memory and end_memory:
        result['memory_used'] = end_memory - start_memory
    
    return result