"""Main analysis runner with timeout and error isolation."""
import time
import subprocess
import tempfile
import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .common import is_cloud_environment, get_memory_usage, timeout_context
from .aspect_ratio import analyze_aspect_ratio
from .scenes import analyze_scenes_lightweight, analyze_scenes_scenedetect
from .black import analyze_black_frames
from .flicker import analyze_flicker
from .freeze import analyze_freeze
from .ocr import analyze_text_ocr, analyze_credit_text


@dataclass
class AnalysisConfig:
    """Configuration for video analysis."""
    safe_mode: bool = True
    deep_ocr: bool = False
    use_scenedetect: bool = False
    pre_transcode: bool = True
    frame_sampling_step: int = 1
    max_ocr_frames: int = 10
    
    # Timeouts for each analyzer (seconds)
    timeout_aspect_ratio: int = 20
    timeout_scenes: int = 60
    timeout_black: int = 60
    timeout_flicker: int = 60
    timeout_freeze: int = 60
    timeout_ocr: int = 120
    timeout_credits: int = 30
    timeout_transcode: int = 300


class AnalyzerRunner:
    """Main runner for video analysis with isolation and error handling."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.results = {
            'analyzers': {},
            'summary': {
                'total_issues': 0,
                'analyzers_run': 0,
                'analyzers_failed': 0,
                'total_duration': 0,
                'memory_peak': 0
            },
            'metadata': {
                'config': config.__dict__,
                'environment': {
                    'is_cloud': is_cloud_environment(),
                    'safe_mode_enabled': config.safe_mode
                }
            }
        }
    
    def pre_transcode_video(self, input_path: str, output_path: str) -> bool:
        """
        Pre-transcode video to normalized format for reliable analysis.
        
        Args:
            input_path: Path to original video
            output_path: Path for transcoded video
            
        Returns:
            True if successful, False otherwise
        """
        if not self.config.pre_transcode:
            return False
            
        try:
            with timeout_context(self.config.timeout_transcode):
                # ffmpeg command for normalization
                cmd = [
                    'ffmpeg', '-i', input_path,
                    '-vf', 'scale=-1:720',  # Max height 720p
                    '-c:v', 'libx264',      # H.264 codec
                    '-preset', 'fast',      # Fast encoding
                    '-c:a', 'aac',          # AAC audio
                    '-r', '30',             # 30fps constant
                    '-pix_fmt', 'yuv420p',  # Compatible pixel format
                    '-y',                   # Overwrite output
                    output_path
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=self.config.timeout_transcode
                )
                
                if result.returncode == 0 and os.path.exists(output_path):
                    # Check if output file is valid
                    file_size = os.path.getsize(output_path)
                    if file_size > 1024:  # At least 1KB
                        return True
                
                return False
                
        except Exception as e:
            print(f"Pre-transcode failed: {e}")
            return False
    
    def run_analyzer(self, name: str, analyzer_func, *args, **kwargs) -> Dict[str, Any]:
        """
        Run a single analyzer with error isolation.
        
        Args:
            name: Name of the analyzer
            analyzer_func: Function to run
            *args, **kwargs: Arguments for the analyzer
            
        Returns:
            Analyzer results with error handling
        """
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            result = analyzer_func(*args, **kwargs)
            success = True
            error = result.get('error')
        except Exception as e:
            result = {
                'issues': [],
                'metadata': {},
                'error': str(e),
                'duration': 0,
                'memory_used': 0
            }
            success = False
            error = str(e)
        
        # Update summary statistics
        duration = time.time() - start_time
        result['duration'] = duration
        
        end_memory = get_memory_usage()
        if start_memory and end_memory:
            memory_used = end_memory - start_memory
            result['memory_used'] = memory_used
            self.results['summary']['memory_peak'] = max(
                self.results['summary']['memory_peak'], 
                end_memory
            )
        
        self.results['summary']['total_duration'] += duration
        self.results['summary']['analyzers_run'] += 1
        
        if error:
            self.results['summary']['analyzers_failed'] += 1
        else:
            # Count issues
            issues_count = len(result.get('issues', []))
            self.results['summary']['total_issues'] += issues_count
        
        # Store results
        self.results['analyzers'][name] = {
            'success': success,
            'duration': duration,
            'memory_used': result.get('memory_used', 0),
            'issues_count': len(result.get('issues', [])),
            'result': result
        }
        
        return result
    
    def analyze_video(self, video_path: str, description: str = "") -> Dict[str, Any]:
        """
        Run complete video analysis pipeline.
        
        Args:
            video_path: Path to video file
            description: Video description for context
            
        Returns:
            Complete analysis results
        """
        analysis_start = time.time()
        working_video_path = video_path
        temp_transcoded_path = None
        
        try:
            # Pre-transcode if enabled
            if self.config.pre_transcode:
                temp_transcoded_path = tempfile.mktemp(suffix='.mp4')
                if self.pre_transcode_video(video_path, temp_transcoded_path):
                    working_video_path = temp_transcoded_path
                    self.results['metadata']['transcoded'] = True
                else:
                    self.results['metadata']['transcoded'] = False
                    self.results['metadata']['transcode_error'] = "Failed to pre-transcode"
            
            # Run aspect ratio analysis
            self.run_analyzer(
                'aspect_ratio',
                analyze_aspect_ratio,
                working_video_path,
                self.config.timeout_aspect_ratio
            )
            
            # Run scene analysis
            if self.config.use_scenedetect and not self.config.safe_mode:
                self.run_analyzer(
                    'scenes',
                    analyze_scenes_scenedetect,
                    working_video_path,
                    self.config.timeout_scenes
                )
            else:
                # Use lightweight scene detection
                self.run_analyzer(
                    'scenes',
                    analyze_scenes_lightweight,
                    working_video_path,
                    self.config.timeout_scenes
                )
            
            # Run black frame analysis
            self.run_analyzer(
                'black_frames',
                analyze_black_frames,
                working_video_path,
                timeout_seconds=self.config.timeout_black
            )
            
            # Run flicker analysis
            self.run_analyzer(
                'flicker',
                analyze_flicker,
                working_video_path,
                timeout_seconds=self.config.timeout_flicker
            )
            
            # Run freeze analysis (skip for photo compilations)
            if 'photo' not in description.lower() and 'slideshow' not in description.lower():
                self.run_analyzer(
                    'freeze',
                    analyze_freeze,
                    working_video_path,
                    timeout_seconds=self.config.timeout_freeze
                )
            
            # Run OCR analysis if enabled
            if self.config.deep_ocr:
                max_frames = 5 if self.config.safe_mode else self.config.max_ocr_frames
                self.run_analyzer(
                    'ocr',
                    analyze_text_ocr,
                    working_video_path,
                    max_frames=max_frames,
                    sample_step=self.config.frame_sampling_step * 10,
                    timeout_seconds=self.config.timeout_ocr
                )
            
            # Run credit analysis
            self.run_analyzer(
                'credits',
                analyze_credit_text,
                working_video_path,
                description,
                self.config.timeout_credits
            )
            
        finally:
            # Clean up transcoded file
            if temp_transcoded_path and os.path.exists(temp_transcoded_path):
                try:
                    os.remove(temp_transcoded_path)
                except:
                    pass
        
        # Finalize results
        self.results['summary']['total_duration'] = time.time() - analysis_start
        self.results['summary']['success_rate'] = (
            (self.results['summary']['analyzers_run'] - self.results['summary']['analyzers_failed']) 
            / max(1, self.results['summary']['analyzers_run'])
        )
        
        # Determine if analysis was successful overall
        real_issues = []
        for analyzer_name, analyzer_result in self.results['analyzers'].items():
            if analyzer_result['success'] and analyzer_result.get('result'):
                issues = analyzer_result['result'].get('issues', [])
                # Filter out informational issues for success determination
                critical_issues = [
                    issue for issue in issues 
                    if issue.get('severity') in ['warning', 'error']
                ]
                real_issues.extend(critical_issues)
        
        self.results['summary']['has_critical_issues'] = len(real_issues) > 0
        self.results['summary']['critical_issues_count'] = len(real_issues)
        
        return self.results
    
    def get_json_report(self) -> str:
        """Get analysis results as JSON string."""
        return json.dumps(self.results, indent=2, default=str)