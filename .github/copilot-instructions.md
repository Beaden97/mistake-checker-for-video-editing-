# AI TikTok Video QA Tool (Mistake Checker for Video Editing)

This is a Streamlit web application that uses AI and deep learning to analyze TikTok videos for common editing mistakes. It performs OCR text detection, scene analysis, flicker detection, freeze detection, and compares videos against user descriptions or reference videos.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap, Build, and Run the Application
- **NEVER CANCEL: All dependency installation takes 60+ seconds. Set timeout to 300+ seconds.**
- Install system dependencies:
  - `sudo apt update`
  - `sudo apt install -y libgl1-mesa-dev ffmpeg` -- takes 60-120 seconds depending on network. NEVER CANCEL.
  - **Note**: Use `libgl1-mesa-dev` (not `libgl1-mesa-glx` as specified in packages.txt)
- Install Python dependencies:
  - `pip install --upgrade pip`
  - `pip install -r requirements.txt` -- takes 45-60 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
- Run the web application:
  - `streamlit run app.py` -- starts in 3-5 seconds
  - Access at http://localhost:8501
  - Use `streamlit run app.py --server.headless=true --server.port=8501` for headless operation

### Development and Code Quality
- **NEVER CANCEL: Linting takes 2-5 seconds but install takes 30+ seconds.**
- Install linting tools: `pip install flake8 black` -- takes 20-30 seconds
- Check code style: `flake8 app.py --max-line-length=88`
- Auto-format code: `black app.py --line-length=88`
- Validate Python syntax: `python -m py_compile app.py`
- Test imports: `python -c "import app; print('App imports successfully')"` -- takes 10-15 seconds due to ML model loading

### Key Dependencies and Their Purpose
- **streamlit**: Web framework for the application interface
- **opencv-python-headless**: Computer vision for video processing
- **scenedetect**: Scene change detection in videos
- **paddleocr**: Optical Character Recognition for text detection
- **pyspellchecker**: Spell checking for detected text
- **yt-dlp**: Download videos from URLs (YouTube, TikTok, etc.)
- **numpy**: Numerical operations for video analysis
- **paddlepaddle**: Deep learning framework for OCR models

## Validation Scenarios

### Manual Application Testing
ALWAYS test these scenarios after making changes to ensure functionality:

1. **Core Dependencies Validation Test**:
   ```bash
   python -c "
   import streamlit as st; print('✓ Streamlit import successful')
   import cv2; print('✓ OpenCV import successful')
   from scenedetect import VideoManager, SceneManager; print('✓ Scene detection import successful')
   from paddleocr import PaddleOCR; print('✓ PaddleOCR import successful (expect model warnings)')
   from spellchecker import SpellChecker; print('✓ Spell checker import successful')
   import yt_dlp; print('✓ yt-dlp import successful')
   print('\\n✓ All core dependencies imported successfully')
   "
   ```
   Expected output: All imports successful with possible PaddleOCR model host warnings (normal)

2. **Basic Application Launch Test**:
   - Run `streamlit run app.py --server.headless=true --server.port=8501`
   - Test accessibility: `curl -s http://localhost:8501 | head -5`
   - Should return HTML content starting with `<!--` (Streamlit header)
   - Verify the application loads without crashes
   - Stop with `pkill -f streamlit`

3. **Video Processing Capabilities Test**:
   ```bash
   python -c "
   import cv2
   print('Testing OpenCV video processing...')
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   print('✓ OpenCV codecs available')
   print('✓ Video processing capabilities confirmed')
   "
   ```

4. **Application Import Test**:
   - Run `python -c "import app; print('App imports successfully')"`
   - Expect warnings about missing model hosts (normal in sandbox environments)
   - Verify no import errors occur

5. **Linting Validation**:
   - Run `flake8 app.py --max-line-length=88`
   - Expect numerous style violations (existing codebase has many)
   - Focus on fixing only NEW violations you introduce

## Repository Structure

### Root Directory Contents
```
.
├── app.py                    # Main Streamlit application (800+ lines)
├── requirements.txt          # Python dependencies (10 packages)
├── packages.txt             # System packages (libgl1-mesa-glx, ffmpeg)
├── .streamlit/
│   └── config.toml          # Streamlit config (2GB upload limit)
├── .gitignore               # Git ignore patterns
└── .github/
    └── copilot-instructions.md
```

### Key Application Components in app.py
- **Lines 1-55**: Streamlit UI setup and file upload interface
- **Lines 56-87**: OCR and spell checker initialization (@st.cache_resource)
- **Lines 88-224**: Video analysis functions (scenes, black frames, flicker, freeze)
- **Lines 225-354**: Text detection and spell checking functions
- **Lines 355-489**: Video downloading and URL handling
- **Lines 490-556**: Comparison logic between videos
- **Lines 557-807**: Main analysis pipeline and results display

## Common Development Tasks

### Adding New Video Analysis Features
- Add new detection functions after line 224 (following existing pattern)
- Use OpenCV for video processing (cv2 already imported)
- Cache expensive operations with @st.cache_data or @st.cache_resource
- Add results to the analysis pipeline around line 557

### Modifying Text Detection
- Text parsing functions start at line 225
- OCR results processing at line 279
- Spell checking logic at line 354
- Always test with various text patterns in video descriptions

### Debugging Video Processing Issues
- Check video format compatibility (supports mp4, mov)
- Verify OpenCV can read video: `cv2.VideoCapture(video_path).isOpened()`
- Monitor memory usage - ML models are memory intensive
- Use smaller video files for testing (under 100MB recommended)

## Performance and Timing Expectations

### Installation Times (Set appropriate timeouts)
- System packages: 60-120 seconds
- Python dependencies: 45-60 seconds  
- Linting tools: 20-30 seconds

### Runtime Performance
- Application startup: 3-5 seconds
- OCR model loading: 10-15 seconds (first import)
- Video analysis: Depends on video length and complexity
- Small video (< 30 seconds): 30-60 seconds
- Medium video (1-2 minutes): 2-5 minutes
- Large video (> 5 minutes): 10+ minutes

### Critical: NEVER CANCEL Operations
- **NEVER CANCEL** any pip install commands - they may take 60+ seconds
- **NEVER CANCEL** apt package installation - can take 2+ minutes
- **NEVER CANCEL** video analysis for longer videos - can take 10+ minutes
- Always set timeouts of 300+ seconds for installs, 600+ seconds for video processing

## Error Handling and Common Issues

### Network-Related Issues
- OCR models may fail to download in restricted environments
- YouTube/TikTok downloads may fail due to network restrictions
- Document these as "fails due to network limitations" if they occur

### Memory and Resource Issues
- Large videos may cause memory errors
- ML models require significant RAM
- Consider processing videos in smaller chunks if needed

### Code Style and Standards
- Existing code has many flake8 violations - focus only on NEW code
- Use 88-character line length for new code
- Follow existing function naming patterns
- Add docstrings for new functions following existing style
- **Do NOT attempt to fix existing style violations** - focus only on new code
- **Do NOT reformat the entire app.py file** - it will create massive diffs

## Testing Strategy

### Before Committing Changes
1. Run `python -m py_compile app.py` to check syntax
2. Run `flake8 app.py --max-line-length=88` and fix NEW violations only
3. Test application launch with `streamlit run app.py`
4. Verify your changes work with a simple test scenario
5. Check that existing functionality still works

### No Automated Tests
- This repository has no test suite
- Manual testing is required for all changes
- Focus on testing the specific functionality you modified
- Always test with both valid and invalid inputs

## Development Environment Notes

### Python Version and Environment
- Tested with Python 3.12.3
- Uses pip for package management (no conda or pipenv)
- No virtual environment setup required (but recommended)

### IDE and Development Tools
- VS Code recommended for Python development
- Streamlit provides built-in auto-reload during development
- Use browser developer tools to debug frontend issues

### Documentation and Resources
- Streamlit docs: https://docs.streamlit.io/
- OpenCV Python docs: https://docs.opencv.org/
- PaddleOCR docs: https://github.com/PaddlePaddle/PaddleOCR

Remember: This application processes user-uploaded videos and performs complex ML operations. Always validate that your changes maintain data privacy and don't introduce security vulnerabilities.