"""Main analysis runner with timeout and error isolation."""
import time
import subprocess
import tempfile
import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, fields

from .common import is_cloud_environment, get_memory_usage, timeout_context
from .aspect_ratio import analyze_aspect_ratio
from .scenes import analyze_scenes_lightweight, analyze_scenes_scenedetect
from .black import analyze_black_frames
from .flicker import analyze_flicker
from .freeze import analyze_freeze
from .ocr import analyze_text_ocr, analyze_credit_text
from .description_parser import DescriptionParser, ParsedDescription


@dataclass
class AnalysisConfig:
    """Configuration for video analysis."""
    safe_mode: bool = True
    deep_ocr: bool = False
    use_scenedetect: bool = False
    pre_transcode: bool = True
    frame_sampling_step: int = 1
    max_ocr_frames: int = 10
    spell_variant: str = "US"
    custom_words: Optional[List[str]] = None
    min_confidence_for_spell: float = 0.4
    
    # Timeouts for each analyzer (seconds)
    timeout_aspect_ratio: int = 20
    timeout_scenes: int = 60
    timeout_black: int = 60
    timeout_flicker: int = 60
    timeout_freeze: int = 60
    timeout_ocr: int = 120
    timeout_credits: int = 30
    timeout_transcode: int = 300
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnalysisConfig':
        """Create AnalysisConfig from dict, ignoring unknown keys for backward compatibility."""
        # Get all field names from dataclass
        field_names = {f.name for f in fields(cls)}
        
        # Filter dict to only include known fields
        filtered_kwargs = {k: v for k, v in config_dict.items() if k in field_names}
        
        return cls(**filtered_kwargs)
    
    def __post_init__(self):
        """Post-initialization to ensure custom_words is never None."""
        if self.custom_words is None:
            self.custom_words = []


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
        
        # Parse the description for AI prompt-style content
        description_parser = DescriptionParser()
        parsed_description = description_parser.parse(description)
        
        # Store parsed description in metadata
        self.results['metadata']['parsed_description'] = parsed_description.to_dict()
        
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
            
            # Determine which analyzers to run based on analysis instructions
            should_run_analyzers = self._determine_analyzers_to_run(parsed_description)
            
            # Run aspect ratio analysis
            if should_run_analyzers.get('aspect_ratio', True):
                self.run_analyzer(
                    'aspect_ratio',
                    analyze_aspect_ratio,
                    working_video_path,
                    self.config.timeout_aspect_ratio
                )
            
            # Run scene analysis
            if should_run_analyzers.get('scenes', True):
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
            if should_run_analyzers.get('black_frames', True):
                self.run_analyzer(
                    'black_frames',
                    analyze_black_frames,
                    working_video_path,
                    timeout_seconds=self.config.timeout_black
                )
            
            # Run flicker analysis
            if should_run_analyzers.get('flicker', True):
                self.run_analyzer(
                    'flicker',
                    analyze_flicker,
                    working_video_path,
                    timeout_seconds=self.config.timeout_flicker
                )
            
            # Run freeze analysis (skip for photo compilations or if instructed)
            should_skip_freeze = (
                'photo' in parsed_description.general_description.lower() or 
                'slideshow' in parsed_description.general_description.lower() or
                any('freeze' in keyword.lower() and 'no' in instruction.lower() 
                    for instruction in parsed_description.analysis_instructions 
                    for keyword in instruction.split())
            )
            
            if should_run_analyzers.get('freeze', True) and not should_skip_freeze:
                self.run_analyzer(
                    'freeze',
                    analyze_freeze,
                    working_video_path,
                    timeout_seconds=self.config.timeout_freeze
                )
            
            # Run OCR analysis if enabled or if expected text is provided
            has_expected_text = bool(parsed_description.expected_text)
            should_run_ocr = self.config.deep_ocr or has_expected_text
            
            if should_run_analyzers.get('ocr', True) and should_run_ocr:
                max_frames = 5 if self.config.safe_mode else self.config.max_ocr_frames
                
                # Enhance custom words with expected text
                enhanced_custom_words = list(self.config.custom_words)
                if parsed_description.expected_text:
                    enhanced_custom_words.extend(parsed_description.expected_text)
                
                self.run_analyzer(
                    'ocr',
                    analyze_text_ocr,
                    working_video_path,
                    max_frames=max_frames,
                    sample_step=self.config.frame_sampling_step * 10,
                    timeout_seconds=self.config.timeout_ocr,
                    spell_variant=self.config.spell_variant,
                    custom_words=enhanced_custom_words,
                    min_confidence_for_spell=self.config.min_confidence_for_spell,
                    expected_text=parsed_description.expected_text  # Pass expected text for comparison
                )
            
            # Run credit analysis
            if should_run_analyzers.get('credits', True):
                self.run_analyzer(
                    'credits',
                    analyze_credit_text,
                    working_video_path,
                    parsed_description.general_description,  # Use general description for credits
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
    
    def _determine_analyzers_to_run(self, parsed_description: ParsedDescription) -> Dict[str, bool]:
        """
        Determine which analyzers to run based on analysis instructions.
        
        Args:
            parsed_description: Parsed description object
            
        Returns:
            Dictionary indicating which analyzers should run
        """
        # Default: run all analyzers
        analyzers = {
            'aspect_ratio': True,
            'scenes': True,
            'black_frames': True,
            'flicker': True,
            'freeze': True,
            'ocr': True,
            'credits': True
        }
        
        # Check for specific instructions to skip or focus on certain analyzers
        for instruction in parsed_description.analysis_instructions:
            instruction_lower = instruction.lower()
            
            # Skip instructions
            if 'skip' in instruction_lower or 'ignore' in instruction_lower:
                if 'aspect' in instruction_lower or 'ratio' in instruction_lower:
                    analyzers['aspect_ratio'] = False
                if 'scene' in instruction_lower:
                    analyzers['scenes'] = False
                if 'black' in instruction_lower:
                    analyzers['black_frames'] = False
                if 'flicker' in instruction_lower:
                    analyzers['flicker'] = False
                if 'freeze' in instruction_lower:
                    analyzers['freeze'] = False
                if 'text' in instruction_lower or 'ocr' in instruction_lower:
                    analyzers['ocr'] = False
                if 'credit' in instruction_lower:
                    analyzers['credits'] = False
        
        # Check focus keywords for prioritization (doesn't disable others, but could be used for optimization)
        focus_keywords = set(kw.lower() for kw in parsed_description.look_for_keywords)
        
        # If only specific aspects are mentioned, we could optimize by focusing on those
        # For now, we keep all analyzers enabled but this could be enhanced
        
        return analyzers
    
    def get_json_report(self) -> str:
        """Get analysis results as JSON string."""
        return json.dumps(self.results, indent=2, default=str)