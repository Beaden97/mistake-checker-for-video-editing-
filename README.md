# AI TikTok Video QA (Deep Learning, Web-Based, No Install)

An intelligent video analysis tool that uses AI and deep learning to detect common editing mistakes in TikTok-style videos, with advanced on-screen text analysis capabilities.

## Features

### Core Video Analysis
- **Aspect Ratio Detection**: Ensures videos meet TikTok's 9:16 vertical format
- **Scene Cut Detection**: Lightweight OpenCV-based detection using histogram correlation
- **Black Frame Detection**: Identifies unwanted black frames with configurable duration thresholds
- **Flicker/Flash Detection**: Detects sudden brightness changes that may indicate problems
- **Freeze Frame Detection**: Identifies static frames that may indicate video freezing

### Advanced Text Analysis
- **OCR Text Detection**: Uses PaddleOCR to extract text from video frames
- **Text Timeline Construction**: Builds detailed timelines showing when text appears and disappears
- **Text Bleed Detection**: Identifies text that persists briefly into the next shot after a scene cut
- **Text Flash Detection**: Detects text that appears too briefly to be intentional
- **Spell Checking**: Identifies potential typos in detected text overlays

### Smart Analysis Features
- **Photo Compilation Detection**: Automatically detects slideshow-style videos and adjusts sensitivity
- **Content Matching**: Compares detected text with user-provided video descriptions
- **Reference Video Comparison**: Compare your video against reference videos from URLs

## User Interface

### Main Interface
- **Video Upload**: Drag and drop interface for MP4, MOV, and MPEG4 files (up to 2GB)
- **Description Input**: Detailed text area for describing expected video content
- **Reference URL**: Optional field for YouTube or direct video links for comparison
- **Analysis Results**: Comprehensive mistake detection with timestamps
- **Detailed Metrics**: Summary dashboard showing counts of different issue types

### Sidebar Controls
Located in the left sidebar for fine-tuning analysis parameters:

#### Scene Cut Detection
- **Cut Detection Sensitivity**: Choose from Low, Medium (default), or High sensitivity
  - Higher sensitivity detects more cuts but may produce false positives
  - Automatically reduced for photo compilations

#### Text Analysis
- **Bleed Tolerance**: Set maximum time (0.1-1.0 seconds, default 0.25) text can linger after a cut before being flagged
- **Flash Min Duration**: Set minimum duration (0.1-1.0 seconds, default 0.25) for text to not be considered flashing

## How It Works

### Scene Cut Detection
Uses OpenCV histogram correlation instead of heavy scene detection libraries:
1. Converts frames to HSV color space
2. Computes normalized histograms
3. Calculates correlation between consecutive frames
4. Flags cuts when correlation drops below threshold
5. Samples at ~10-15 fps for performance

### Text Timeline Analysis
1. **OCR Sampling**: Extracts text from 20 evenly distributed frames
2. **Text Grouping**: Groups similar text across consecutive samples
3. **Timeline Construction**: Creates segments with start/end times, confidence, and bounding boxes
4. **Mistake Detection**: Analyzes timeline against scene cuts to find issues

### Text Bleed Detection
- Identifies text segments that start before a scene cut but end shortly after
- Configurable tolerance for how long text can persist post-cut
- Reports exact linger duration and cut timing

### Text Flash Detection
- Finds text with duration below minimum threshold
- Excludes text that appears during scene transitions
- Helps identify unintentionally brief text overlays

## Usage Examples

### Basic Analysis
1. Upload your TikTok video
2. Describe what the video should contain
3. Click "Analyze Video"
4. Review detected mistakes with timestamps

### Advanced Configuration
1. Adjust cut detection sensitivity in sidebar
2. Set bleed tolerance based on your editing style
3. Configure flash duration based on your text timing preferences
4. Run analysis with custom settings

### Reference Comparison
1. Provide a YouTube URL or direct video link
2. App will analyze both videos and show differences
3. Detailed comparison shows metrics for both videos
4. Identifies where your video differs from the reference

## Technical Requirements

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- PaddleOCR
- PySpellChecker
- yt-dlp (for YouTube downloads)

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Analysis Output

### Mistake Types Detected
- **Text bleeds across cut**: Text that lingers briefly after scene transitions
- **Text flashes briefly**: Text appearing for very short durations
- **Potential typo**: Spelling errors in detected text
- **Black frame detected**: Unwanted black frames
- **Flicker/flash detected**: Sudden brightness changes
- **Frozen frame detected**: Static video segments
- **Aspect ratio issues**: Non-TikTok format videos
- **Content mismatches**: Text not matching provided description

### Analysis Summary
The app provides metrics including:
- Scene cuts detected with confidence scores
- Text element counts and timeline segments
- Issue counts by category
- Photo compilation detection status

### Detailed Results
Expandable section showing:
- Complete scene cut timeline with timestamps
- Text timeline with longest segments highlighted
- Comprehensive statistics for all detection categories
- Side-by-side comparison for reference videos

## Tips for Best Results

1. **Provide Detailed Descriptions**: Include expected text, timing, and scene information
2. **Adjust Sensitivity**: Use lower cut sensitivity for slideshow-style content
3. **Fine-tune Tolerances**: Adjust bleed and flash thresholds based on your editing style
4. **Use Reference Videos**: Compare against similar successful videos for insights
5. **Review Timestamps**: Check flagged issues at specific times for context

## Limitations

- OCR accuracy depends on text clarity and contrast
- Scene cut detection works best with clear visual changes
- Internet connection required for YouTube reference videos
- Large files may take several minutes to process