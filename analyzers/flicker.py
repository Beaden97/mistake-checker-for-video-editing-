"""Flicker and flash detection for video content."""
import time
import cv2
import numpy as np
from typing import Dict, Any, List
from .common import safe_capture, timeout_context, get_memory_usage, format_timestamp


def analyze_flicker(video_path: str, threshold: float = 40.0, window_size: int = 5,
                   timeout_seconds: int = 60) -> Dict[str, Any]:
    """
    Detect flicker and sudden brightness changes in video content.
    
    Args:
        video_path: Path to the video file
        threshold: Brightness change threshold for flicker detection
        window_size: Number of frames to analyze for variance
        timeout_seconds: Maximum time to spend on analysis
        
    Returns:
        Dict with analysis results, errors, and timing info
    """
    start_time = time.time()
    start_memory = get_memory_usage()
    
    result = {
        'issues': [],
        'flicker_events': [],
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
                
            brightness_values = []
            frame_idx = 0
            flicker_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale and calculate mean brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                brightness_values.append(mean_brightness)
                
                # Analyze brightness variance within a rolling window
                if len(brightness_values) >= window_size:
                    window_values = brightness_values[-window_size:]
                    
                    # Calculate frame-to-frame differences
                    diffs = [abs(window_values[i] - window_values[i-1]) 
                            for i in range(1, len(window_values))]
                    
                    # Check for sudden changes
                    max_diff = max(diffs) if diffs else 0
                    
                    if max_diff > threshold:
                        timestamp_sec = frame_idx / fps
                        
                        # Check if this is a new flicker event (not too close to previous)
                        is_new_event = True
                        if result['flicker_events']:
                            last_event_time = result['flicker_events'][-1]['timestamp']
                            if timestamp_sec - last_event_time < 0.5:  # Within 0.5 seconds
                                is_new_event = False
                        
                        if is_new_event:
                            flicker_event = {
                                'timestamp': timestamp_sec,
                                'max_brightness_change': max_diff,
                                'brightness_variance': np.var(window_values)
                            }
                            result['flicker_events'].append(flicker_event)
                            flicker_count += 1
                            
                            result['issues'].append({
                                'timestamp': format_timestamp(timestamp_sec),
                                'type': 'flicker',
                                'severity': 'warning',
                                'message': f"Flicker/flash detected (brightness change: {max_diff:.1f})"
                            })
                
                frame_idx += 1
                
                # Limit brightness values buffer to prevent memory issues
                if len(brightness_values) > 1000:
                    brightness_values = brightness_values[-500:]
            
            cap.release()
            
            # Calculate overall brightness statistics
            if brightness_values:
                result['metadata'] = {
                    'total_frames': frame_idx,
                    'flicker_events': flicker_count,
                    'average_brightness': np.mean(brightness_values),
                    'brightness_std': np.std(brightness_values),
                    'brightness_range': (np.min(brightness_values), np.max(brightness_values)),
                    'threshold_used': threshold,
                    'window_size': window_size,
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