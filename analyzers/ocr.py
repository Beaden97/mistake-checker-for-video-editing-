"""OCR text analysis for video content."""
import time
import cv2
import numpy as np
import traceback
import re
import os
from typing import Dict, Any, List, Optional
from .common import safe_capture, timeout_context, get_memory_usage, format_timestamp, frame_generator


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


def _load_uk_extra_words() -> List[str]:
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
    Perform OCR text analysis on video frames with enhanced spell checking.
    
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
    
    try:
        with timeout_context(timeout_seconds):
            # Try to import PaddleOCR
            try:
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                ocr_available = True
            except ImportError:
                ocr_available = False
                result['error'] = "PaddleOCR not available - OCR analysis skipped"
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
            
            # Sample frames for analysis
            for frame_idx, frame in frame_generator(video_path, step=sample_step, max_frames=max_frames):
                if analyzed_frames >= max_frames:
                    break
                
                timestamp_sec = frame_idx / fps
                
                try:
                    # Perform OCR on the frame
                    ocr_result = ocr.ocr(frame, cls=True)
                    
                    if ocr_result and ocr_result[0]:
                        for line in ocr_result[0]:
                            if line and len(line) >= 2:
                                text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                                confidence = line[1][1] if isinstance(line[1], tuple) and len(line[1]) > 1 else 1.0
                                
                                # Store text element
                                text_element = {
                                    'timestamp': timestamp_sec,
                                    'text': text,
                                    'confidence': confidence,
                                    'frame_index': frame_idx
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
                                                correction = spell_checker.correction(word)
                                                if correction and correction != word:
                                                    suggestions[word] = correction
                                            
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
                    
                    # Add detailed error trace
                    result['issues'].append({
                        'timestamp': format_timestamp(timestamp_sec),
                        'type': 'error_trace',
                        'severity': 'info',
                        'message': f"Full traceback for frame OCR error:\n{error_trace}"
                    })
                
                analyzed_frames += 1
            
            cap.release()
            
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
                'fps': fps
            }
            
    except Exception as e:
        result['error'] = traceback.format_exc()
    
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