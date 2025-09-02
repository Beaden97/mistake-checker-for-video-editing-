"""Scene detection and analysis for video content."""
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from .common import safe_capture, timeout_context, get_memory_usage, format_timestamp


def analyze_scenes_lightweight(video_path: str, timeout_seconds: int = 30) -> Dict[str, Any]:
    """
    Lightweight scene detection using histogram differences.
    
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
        'scenes': [],
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
                
            prev_hist = None
            scene_start = 0
            frame_idx = 0
            scene_cuts = []
            
            # Parameters for scene detection
            threshold = 0.5  # Histogram difference threshold
            min_scene_length = 1.0  # Minimum scene length in seconds
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate histogram for luminance channel
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                if prev_hist is not None:
                    # Calculate histogram correlation
                    correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    
                    # If correlation is low, we might have a scene cut
                    if correlation < threshold:
                        current_time = frame_idx / fps
                        scene_length = current_time - scene_start
                        
                        # Record scene if it meets minimum length
                        if scene_length >= min_scene_length:
                            result['scenes'].append({
                                'start': scene_start,
                                'end': current_time,
                                'duration': scene_length
                            })
                        else:
                            # Very short scene detected
                            result['issues'].append({
                                'timestamp': format_timestamp(scene_start),
                                'type': 'short_scene',
                                'severity': 'warning',
                                'message': f"Very short scene detected ({scene_length:.1f}s < {min_scene_length}s)"
                            })
                        
                        scene_start = current_time
                        scene_cuts.append(current_time)
                
                prev_hist = hist
                frame_idx += 1
            
            # Add final scene
            final_time = frame_idx / fps
            if final_time > scene_start:
                result['scenes'].append({
                    'start': scene_start,
                    'end': final_time,
                    'duration': final_time - scene_start
                })
            
            cap.release()
            
            result['metadata'] = {
                'total_scenes': len(result['scenes']),
                'scene_cuts': len(scene_cuts),
                'average_scene_length': np.mean([s['duration'] for s in result['scenes']]) if result['scenes'] else 0,
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


def analyze_scenes_scenedetect(video_path: str, timeout_seconds: int = 60) -> Dict[str, Any]:
    """
    Advanced scene detection using PySceneDetect library.
    
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
        'scenes': [],
        'metadata': {},
        'error': None,
        'duration': 0,
        'memory_used': 0
    }
    
    try:
        with timeout_context(timeout_seconds):
            from scenedetect import VideoManager, SceneManager
            from scenedetect.detectors import ContentDetector
            
            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector())
            
            video_manager.set_downscale_factor()
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            
            scene_list = scene_manager.get_scene_list()
            video_manager.release()
            
            # Process scene list
            for i, (start, end) in enumerate(scene_list):
                start_sec = start.get_seconds()
                end_sec = end.get_seconds()
                duration = end_sec - start_sec
                
                result['scenes'].append({
                    'start': start_sec,
                    'end': end_sec,
                    'duration': duration
                })
                
                # Check for very short scenes
                if duration < 1.0:
                    result['issues'].append({
                        'timestamp': format_timestamp(start_sec),
                        'type': 'short_scene',
                        'severity': 'warning',
                        'message': f"Very short scene detected ({duration:.1f}s < 1.0s)"
                    })
            
            result['metadata'] = {
                'total_scenes': len(result['scenes']),
                'average_scene_length': np.mean([s['duration'] for s in result['scenes']]) if result['scenes'] else 0,
                'detector': 'SceneDetect ContentDetector'
            }
            
    except Exception as e:
        result['error'] = str(e)
    
    # Calculate timing and memory usage
    result['duration'] = time.time() - start_time
    end_memory = get_memory_usage()
    if start_memory and end_memory:
        result['memory_used'] = end_memory - start_memory
    
    return result