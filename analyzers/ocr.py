"""OCR text analysis for video content."""
import time
import cv2
import numpy as np
import traceback
import re
import os
import gc
from typing import Dict, Any, List, Optional
from .common import safe_capture, timeout_context, get_memory_usage, format_timestamp, frame_generator, is_cloud_environment


def _tokenize_for_spell(text: str) -> List[str]:
    """
    Tokenize text for spell checking with robust filtering.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of alphabetic tokens >= 3 characters, lowercased
    """
    # Skip URLs, @handles, #hashtags
    url_pattern = r'https?://[^\s]+|www\.[^\s]+|[^\s]+\.(com|org|net|edu|gov|co\.uk)[^\s]*'
    handle_pattern = r'@[^\s]+'
    hashtag_pattern = r'#[^\s]+'
    
    if re.search(url_pattern, text, re.IGNORECASE):
        return []
    if re.search(handle_pattern, text):
        return []
    if re.search(hashtag_pattern, text):
        return []
    
    # Split into words and filter
    words = []
    for word in text.split():
        # Strip punctuation except apostrophes, but keep internal apostrophes
        cleaned = re.sub(r"[^\w']", '', word)
        cleaned = cleaned.strip("'")  # Remove leading/trailing apostrophes
        
        # Keep only alphabetic words with length >= 3
        if cleaned.isalpha() and len(cleaned) >= 3:
            words.append(cleaned.lower())
    
    return words


def _preprocess_frame_for_ocr(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess frame to improve OCR accuracy, especially on mobile.
    
    Args:
        frame: Input video frame
        
    Returns:
        Processed frame optimized for OCR
    """
    try:
        # Get original dimensions
        height, width = frame.shape[:2]
        
        # Optimize size for OCR - scale to reasonable resolution
        # Target width: 1280px for good OCR results without excessive memory usage
        target_width = 1280
        if width > target_width:
            scale_factor = target_width / width
            new_width = target_width
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to RGB for better OCR results
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Enhance contrast and sharpness for better text detection
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(frame.shape) == 3:
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)
            
            # Merge back
            lab[:, :, 0] = l_channel
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Ensure the frame is in the correct format
        frame = np.ascontiguousarray(frame)
        
        return frame
        
    except Exception:
        # If preprocessing fails, return original frame
        return frame


def _detect_mobile_environment() -> bool:
    """
    Detect if running in a mobile or resource-constrained environment.
    
    Returns:
        True if mobile/constrained environment detected
    """
    # Check for cloud environment (likely constrained)
    if is_cloud_environment():
        return True
    
    # Check environment variables that might indicate mobile/constrained environment
    if os.environ.get('MOBILE_MODE', '').lower() == 'true':
        return True
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        # Consider mobile if less than 4GB available memory
        if memory.available < 4 * 1024 * 1024 * 1024:  # 4GB
            return True
        # Also check total memory (mobile devices typically have <8GB)
        if memory.total < 8 * 1024 * 1024 * 1024:  # 8GB
            return True
    except:
        # If can't determine, assume mobile-safe defaults
        return True
    
    return False


def _get_mobile_optimized_settings(max_frames: int, sample_step: int) -> tuple:
    """
    Get optimized settings for mobile/constrained environments.
    
    Args:
        max_frames: Requested max frames
        sample_step: Requested sample step
        
    Returns:
        Tuple of (optimized_max_frames, optimized_sample_step)
    """
    is_mobile = _detect_mobile_environment()
    
    if is_mobile:
        # More conservative settings for mobile
        mobile_max_frames = min(max_frames, 5)  # Max 5 frames on mobile
        mobile_sample_step = max(sample_step, 60)  # At least every 60 frames
        return mobile_max_frames, mobile_sample_step
    
    return max_frames, sample_step
    """
    Load UK spelling extras from assets/spell/en_GB_extra.txt.
    
    Returns:
        List of UK words, empty if file not found
    """
    try:
        # Try to find the file relative to the repository root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(current_dir)  # Go up one level from analyzers/
        assets_path = os.path.join(repo_root, 'assets', 'spell', 'en_GB_extra.txt')
        
        if os.path.exists(assets_path):
            with open(assets_path, 'r', encoding='utf-8') as f:
                words = [line.strip().lower() for line in f if line.strip()]
                return words
        return []
    except Exception:
        # Non-fatal if file is missing
        return []


def analyze_text_ocr(video_path: str, max_frames: int = 10, sample_step: int = 30,
                    timeout_seconds: int = 120, spell_variant: str = "US", 
                    custom_words: Optional[List[str]] = None, 
                    min_confidence_for_spell: float = 0.4) -> Dict[str, Any]:
    """
    Perform OCR text analysis on video frames with enhanced spell checking and mobile optimization.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to analyze
        sample_step: Frame sampling step (analyze every Nth frame)
        timeout_seconds: Maximum time to spend on analysis
        spell_variant: "US" or "UK" for spell checking
        custom_words: List of custom words to add to dictionary
        min_confidence_for_spell: Minimum OCR confidence for spell checking
        
    Returns:
        Dict with analysis results, errors, and timing info
    """
    start_time = time.time()
    start_memory = get_memory_usage()
    
    result = {
        'issues': [],
        'text_elements': [],
        'metadata': {},
        'error': None,
        'duration': 0,
        'memory_used': 0
    }
    
    # Apply mobile optimizations
    is_mobile = _detect_mobile_environment()
    mobile_max_frames, mobile_sample_step = _get_mobile_optimized_settings(max_frames, sample_step)
    
    # Use mobile settings if detected
    if is_mobile:
        max_frames = mobile_max_frames
        sample_step = mobile_sample_step
        result['metadata']['mobile_optimizations_applied'] = True
    
    try:
        with timeout_context(timeout_seconds):
            # Try to import PaddleOCR with error handling
            ocr = None
            ocr_available = False
            
            # Try to import and initialize PaddleOCR with robust error handling
            ocr = None
            ocr_available = False
            ocr_error_details = None
            
            try:
                from paddleocr import PaddleOCR
                # Try to initialize with minimal settings for maximum compatibility
                ocr = PaddleOCR(use_angle_cls=True, lang='en')
                ocr_available = True
                
                # Force garbage collection after OCR initialization
                gc.collect()
                
            except ImportError as e:
                ocr_error_details = f"PaddleOCR import failed: {str(e)}"
            except Exception as e:
                ocr_error_details = f"PaddleOCR initialization failed: {str(e)}"
                # Check if it's a network/model download issue
                if "model hosting" in str(e).lower() or "network" in str(e).lower():
                    ocr_error_details += " (Network/model download issue - common in restricted environments)"
            
            # If OCR is not available, return early with informative error
            if not ocr_available:
                result['error'] = f"OCR analysis skipped - {ocr_error_details}"
                result['metadata'] = {
                    'ocr_available': False,
                    'ocr_error': ocr_error_details,
                    'mobile_environment_detected': is_mobile,
                    'spell_checker_available': False
                }
                return result
            
            cap = safe_capture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            
            analyzed_frames = 0
            all_texts = []
            spell_errors = []
            
            # Initialize spell checker if available
            spell_checker = None
            uk_extra_loaded = 0
            try:
                from spellchecker import SpellChecker
                spell_checker = SpellChecker()
                
                # Load UK extra words if needed
                if spell_variant == "UK":
                    uk_words = _load_uk_extra_words()
                    if uk_words:
                        spell_checker.word_frequency.load_words(uk_words)
                        uk_extra_loaded = len(uk_words)
                
                # Load custom words if provided
                if custom_words:
                    custom_words_clean = [word.lower().strip() for word in custom_words if word.strip()]
                    if custom_words_clean:
                        spell_checker.word_frequency.load_words(custom_words_clean)
                        
            except ImportError:
                pass
            
            # Sample frames for analysis with mobile optimization
            frame_count = 0
            for frame_idx, frame in frame_generator(video_path, step=sample_step, max_frames=max_frames):
                if analyzed_frames >= max_frames:
                    break
                
                timestamp_sec = frame_idx / fps
                frame_count += 1
                
                try:
                    # Preprocess frame for better OCR accuracy
                    processed_frame = _preprocess_frame_for_ocr(frame)
                    
                    # Perform OCR on the processed frame
                    ocr_result = ocr.ocr(processed_frame, cls=True)
                    
                    # Force cleanup after each frame on mobile
                    if is_mobile:
                        del processed_frame
                        gc.collect()
                    
                    if ocr_result and ocr_result[0]:
                        for line in ocr_result[0]:
                            if line and len(line) >= 2:
                                text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                                confidence = line[1][1] if isinstance(line[1], tuple) and len(line[1]) > 1 else 1.0
                                
                                # Store text element with additional metadata
                                text_element = {
                                    'timestamp': timestamp_sec,
                                    'text': text,
                                    'confidence': confidence,
                                    'frame_index': frame_idx,
                                    'processed_for_mobile': is_mobile
                                }
                                result['text_elements'].append(text_element)
                                all_texts.append(text.lower())
                                
                                # Spell check if available
                                if spell_checker and confidence >= min_confidence_for_spell:
                                    words = _tokenize_for_spell(text)
                                    if words:  # Only check if we have valid words
                                        misspelled = spell_checker.unknown(words)
                                        
                                        if misspelled:
                                            # Get suggestions for misspelled words
                                            suggestions = {}
                                            for word in misspelled:
                                                try:
                                                    correction = spell_checker.correction(word)
                                                    if correction and correction != word:
                                                        suggestions[word] = correction
                                                except:
                                                    # Skip problematic words
                                                    pass
                                            
                                            spell_errors.append({
                                                'timestamp': timestamp_sec,
                                                'text': text,
                                                'misspelled_words': list(misspelled),
                                                'suggestions': suggestions
                                            })
                                            
                                            # Build message with suggestions
                                            message_parts = [f"Potential spelling errors in text: '{text}' - misspelled: {', '.join(misspelled)}"]
                                            if suggestions:
                                                suggestion_text = ', '.join([f"{word} → {sugg}" for word, sugg in suggestions.items()])
                                                message_parts.append(f"Suggestions: {suggestion_text}")
                                            
                                            result['issues'].append({
                                                'timestamp': format_timestamp(timestamp_sec),
                                                'type': 'spelling',
                                                'severity': 'info',
                                                'message': ' | '.join(message_parts)
                                            })
                
                except Exception as e:
                    # Individual frame OCR error - log but continue
                    error_trace = traceback.format_exc()
                    
                    # Add generic OCR error
                    result['issues'].append({
                        'timestamp': format_timestamp(timestamp_sec),
                        'type': 'ocr_error',
                        'severity': 'info',
                        'message': f"OCR failed for frame: {str(e)}"
                    })
                    
                    # Add detailed error trace (only in debug mode to save space)
                    if not is_mobile:  # Skip detailed traces on mobile to save memory
                        result['issues'].append({
                            'timestamp': format_timestamp(timestamp_sec),
                            'type': 'error_trace',
                            'severity': 'info',
                            'message': f"Full traceback for frame OCR error:\n{error_trace}"
                        })
                    
                    # Force cleanup on error
                    if is_mobile:
                        gc.collect()
                
                analyzed_frames += 1
                
                # Check memory usage periodically on mobile
                if is_mobile and frame_count % 2 == 0:  # Every 2 frames
                    current_memory = get_memory_usage()
                    if current_memory and start_memory and (current_memory - start_memory) > 500:  # 500MB threshold
                        result['issues'].append({
                            'timestamp': format_timestamp(timestamp_sec),
                            'type': 'memory_warning',
                            'severity': 'info',
                            'message': f"High memory usage detected ({current_memory - start_memory:.1f}MB), stopping OCR analysis early"
                        })
                        break
            
            cap.release()
            
            # Final cleanup
            if is_mobile:
                gc.collect()
            
            result['metadata'] = {
                'frames_analyzed': analyzed_frames,
                'total_text_elements': len(result['text_elements']),
                'spell_errors': len(spell_errors),
                'unique_texts': len(set(all_texts)),
                'sample_step': sample_step,
                'max_frames_limit': max_frames,
                'ocr_available': ocr_available,
                'spell_checker_available': spell_checker is not None,
                'spell_variant': spell_variant,
                'custom_word_count': len(custom_words) if custom_words else 0,
                'uk_extra_loaded': uk_extra_loaded,
                'min_confidence_for_spell': min_confidence_for_spell,
                'fps': fps,
                'mobile_environment_detected': is_mobile,
                'preprocessing_applied': True
            }
            
    except Exception as e:
        result['error'] = traceback.format_exc()
        # Force cleanup on major error
        if _detect_mobile_environment():
            gc.collect()
    
    # Calculate timing and memory usage
    result['duration'] = time.time() - start_time
    end_memory = get_memory_usage()
    if start_memory and end_memory:
        result['memory_used'] = end_memory - start_memory
    
    return result


def analyze_credit_text(video_path: str, description: str, timeout_seconds: int = 30) -> Dict[str, Any]:
    """
    Simple rule-based credit text detection.
    
    Args:
        video_path: Path to the video file
        description: Video description to check against
        timeout_seconds: Maximum time to spend on analysis
        
    Returns:
        Dict with analysis results, errors, and timing info
    """
    start_time = time.time()
    start_memory = get_memory_usage()
    
    result = {
        'issues': [],
        'credits_found': [],
        'metadata': {},
        'error': None,
        'duration': 0,
        'memory_used': 0
    }
    
    try:
        with timeout_context(timeout_seconds):
            # Define common credit keywords
            credit_keywords = [
                'credit', 'credits', 'created by', 'made by', 'produced by',
                'directed by', 'edited by', '@', 'tiktok.com', 'instagram.com',
                'youtube.com', 'follow', 'subscribe', 'like and subscribe'
            ]
            
            # Parse description for expected credits
            description_lower = description.lower()
            has_credit_expectation = any(keyword in description_lower for keyword in credit_keywords)
            
            # Simple heuristic: if description mentions credits but we're not detecting them properly,
            # we'll flag it as a potential issue
            if has_credit_expectation:
                result['metadata']['expected_credits'] = True
                result['issues'].append({
                    'timestamp': 'End',
                    'type': 'credit_check',
                    'severity': 'info',
                    'message': 'Description mentions credits - verify they appear correctly in video'
                })
            else:
                result['metadata']['expected_credits'] = False
            
            result['metadata'].update({
                'credit_keywords_checked': len(credit_keywords),
                'description_mentions_credits': has_credit_expectation
            })
            
    except Exception as e:
        result['error'] = traceback.format_exc()
    
    # Calculate timing and memory usage
    result['duration'] = time.time() - start_time
    end_memory = get_memory_usage()
    if start_memory and end_memory:
        result['memory_used'] = end_memory - start_memory
    
    return result