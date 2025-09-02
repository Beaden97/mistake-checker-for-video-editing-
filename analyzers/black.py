"""Black frame detection for video content."""
import time
import cv2
import numpy as np
from typing import Dict, Any, List
from .common import safe_capture, timeout_context, get_memory_usage, format_timestamp


def analyze_black_frames(video_path: str, threshold: int = 15, min_duration: float = 1.0, 
                        timeout_seconds: int = 60) -> Dict[str, Any]:
    """
    Detect black frames in video content.
    
    Args:
        video_path: Path to the video file
        threshold: Luminance threshold for black detection (0-255)
        min_duration: Minimum duration of black sequence to report (seconds)
        timeout_seconds: Maximum time to spend on analysis
        
    Returns:
        Dict with analysis results, errors, and timing info
    """
    start_time = time.time()
    start_memory = get_memory_usage()
    
    result = {
        'issues': [],
        'black_sequences': [],
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
                fps = 30
                
            current_black_start = None
            frame_idx = 0
            black_frame_count = 0
            total_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                
                # Convert to grayscale and calculate mean luminance
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_luminance = np.mean(gray)
                
                if mean_luminance < threshold:
                    # This is a black frame
                    black_frame_count += 1
                    if current_black_start is None:
                        current_black_start = frame_idx
                else:
                    # Not a black frame
                    if current_black_start is not None:
                        # End of black sequence
                        black_duration = (frame_idx - current_black_start) / fps
                        start_time_sec = current_black_start / fps
                        
                        sequence_info = {
                            'start': start_time_sec,
                            'duration': black_duration,
                            'frame_count': frame_idx - current_black_start
                        }
                        result['black_sequences'].append(sequence_info)
                        
                        if black_duration >= min_duration:
                            result['issues'].append({
                                'timestamp': format_timestamp(start_time_sec),
                                'type': 'black_frame',
                                'severity': 'warning',
                                'message': f"Black frames detected for {black_duration:.1f}s"
                            })
                        
                        current_black_start = None
                
                frame_idx += 1
            
            # Handle case where video ends with black frames
            if current_black_start is not None:
                black_duration = (frame_idx - current_black_start) / fps
                start_time_sec = current_black_start / fps
                
                sequence_info = {
                    'start': start_time_sec,
                    'duration': black_duration,
                    'frame_count': frame_idx - current_black_start
                }
                result['black_sequences'].append(sequence_info)
                
                if black_duration >= min_duration:
                    result['issues'].append({
                        'timestamp': format_timestamp(start_time_sec),
                        'type': 'black_frame',
                        'severity': 'warning',
                        'message': f"Black frames detected for {black_duration:.1f}s"
                    })
            
            cap.release()
            
            result['metadata'] = {
                'total_frames': total_frames,
                'black_frames': black_frame_count,
                'black_percentage': (black_frame_count / total_frames * 100) if total_frames > 0 else 0,
                'threshold_used': threshold,
                'min_duration': min_duration,
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