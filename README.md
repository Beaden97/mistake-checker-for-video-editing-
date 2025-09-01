# AI TikTok Video QA Tool

A web-based AI tool for analyzing TikTok videos and detecting common editing mistakes using deep learning.

## Features

- **Video Analysis**: Upload TikTok videos and get detailed analysis of potential editing mistakes
- **Reference Comparison**: Compare your video against reference videos from YouTube or direct links
- **Content Verification**: Check if your video matches your detailed notes and requirements
- **Customizable Theming**: Clean, branded appearance with configurable themes

## Appearance Controls

The app includes a comprehensive theming system accessible via the **Appearance** expander in the sidebar:

### Theme Options
- **Theme Mode**: Choose between System, Light, or Dark themes
- **Brand Color**: Customize the primary brand color using a color picker (default: #7C3AED)
- **Density**: Adjust spacing with Cozy, Comfortable, or Compact layouts
- **Content Width**: Control the maximum content width (960-1400px)
- **App Title**: Customize the main page title in real-time

### Default Theming
The app uses a sophisticated dark theme by default with:
- Professional Inter font family
- Consistent border radius and shadows
- Branded button styling
- Polished form controls

### Logo Support
The app automatically detects and uses logo files if present in the `assets/` directory:
- Supported formats: `logo.png`, `logo.jpg`, `logo.svg`, `favicon.png`
- Falls back to 🎬 emoji if no logo is found

## How to Use

1. **Upload your TikTok video** using the file uploader
2. **Describe your video requirements** in detail in the notes box
3. **Optional**: Enter a reference video URL for comparison
4. Click **Analyze Video** to get AI-powered analysis
5. **Customize appearance** using the sidebar controls as needed

## Customization

### Changing the Default Title
You can change the hero title in two ways:
1. **Runtime**: Use the "App title" input in the Appearance sidebar
2. **Code**: Edit the `_DEFAULT_TITLE` constant in `app_theme.py`

### Custom Styling
- Base styles are defined in `assets/styles.css`
- Theme defaults are configured in `.streamlit/config.toml`
- Runtime theme overrides are handled via CSS variables

## Technical Details

- Built with Streamlit
- Uses PaddleOCR for text detection
- Scene detection with PySceneDetect
- Video analysis with OpenCV
- Spell checking with PySpellChecker
- Video downloading with yt-dlp