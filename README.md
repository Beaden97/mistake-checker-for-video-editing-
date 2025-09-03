# AI TikTok Video QA Tool

A web-based AI tool for analyzing TikTok videos and detecting common editing mistakes using deep learning.

## Features

- **Video Analysis**: Upload TikTok videos and get detailed analysis of potential editing mistakes
- **AI Prompt-Style Descriptions**: Intelligent parsing of video descriptions with support for expected text and analysis instructions
- **Mobile-Friendly Controls**: Analysis settings available inline on mobile devices (not just sidebar)
- **Enhanced Spell Checking**: UK/US language variants with custom dictionary support
- **Expected Text Verification**: Automatically compare detected text against your expected content
- **Reference Comparison**: Compare your video against reference videos from YouTube or direct links
- **Content Verification**: Check if your video matches your detailed notes and requirements
- **Full Error Debugging**: Complete Python tracebacks for easier troubleshooting
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

1. **Configure analysis settings** using the mobile-friendly inline controls (available on all devices)
2. **Upload your TikTok video** using the file uploader
3. **Describe your video requirements** in detail using AI prompt-style format (see below)
4. **Optional**: Enter a reference video URL for comparison
5. Click **Analyze Video** to get AI-powered analysis
6. **Customize appearance** using the sidebar controls as needed

### AI Prompt-Style Descriptions

The tool now supports intelligent parsing of video descriptions. You can structure your description to include:

#### Expected Text Content
Use quotes with colons to specify text that should appear in your video:
```
"Step 1: Welcome" : "Step 2: Start cooking" : "Step 3: Enjoy your meal"
```

#### Analysis Instructions  
Use keywords like "Look for:", "Check:", "Verify:", etc. to specify what to focus on:
```
Look for: timing accuracy, text visibility, audio sync
Check for: proper lighting, clear audio, no flicker
Skip: freeze detection
```

#### Text Keywords
Use specific keywords to identify expected text:
```
Title: "My Tutorial Video"
Expected text: "Welcome to my channel"
Text should be: "Subscribe now"
```

#### Example Complete Description
```
This is a cooking tutorial video about making pasta. The video has upbeat background music.

Expected text: "Step 1: Boil water" : "Step 2: Add pasta" : "Step 3: Cook for 10 minutes"

Look for: timing accuracy, text visibility, audio sync
Check for: proper lighting, no flicker
Skip: freeze detection for slideshow sections
```

The AI will automatically:
- Extract expected text for OCR comparison
- Focus analysis on specified areas
- Skip analyzers when instructed
- Provide detailed feedback on text matching

### Analysis Settings

The inline controls provide the following options:

- **Safe Mode**: Lightweight analysis with reduced timeouts (recommended for cloud environments)
- **Deep OCR Analysis**: Enable text detection and spell checking
- **Frame Sampling Step**: Control how many frames to skip (higher = faster but less thorough)
- **Max OCR Frames**: Limit the number of frames analyzed for text
- **Text Language**: Choose between English (US) and English (UK) for spell checking
- **Custom Dictionary**: Add brand names, slang, or specialized terms that shouldn't be flagged as misspelled

### Configuration Compatibility

The app now features **backward-compatible configuration handling**:

- **Flexible Parameters**: The analysis configuration gracefully ignores unknown parameters, preventing crashes when new UI features are added
- **Safe Defaults**: All configuration options have sensible defaults that work out-of-the-box
- **Forward Compatibility**: New analysis features can be added without breaking existing functionality
- **Error Prevention**: No more "unexpected keyword argument" errors when the UI sends additional parameters

This makes the app more robust and easier to extend with new features.

### Spell Checking Features

- **Language Variants**: Supports both US and UK English spelling conventions
- **Smart Tokenization**: Ignores URLs, @handles, #hashtags, and numeric tokens
- **Custom Words**: Add your own words via the custom dictionary textarea
- **Confidence Thresholds**: Uses lower OCR confidence thresholds when Deep OCR is enabled (0.35 vs 0.4)
- **UK Wordlist**: Includes supplemental UK spelling words like "colour", "organise", "theatre"

Note: PaddleOCR uses English ('en') language model for both US and UK variants. The language selector only affects the spell checker, not the OCR detection itself.

## Email Feedback Configuration

The app includes a feedback widget that allows users to send feedback directly via email. To enable this feature:

### Setting Up SMTP

1. **Configure SMTP settings** in your `.streamlit/secrets.toml` file:
   ```toml
   [smtp]
   host = "your-smtp-server.com"
   port = 587
   user = "your-username"
   password = "your-password"
   from = "sender@example.com"
   to = "feedback@example.com"
   use_tls = true
   ```

2. **Use the provided examples** in `.streamlit/secrets.example.toml` for:
   - **Gmail**: Use App Passwords (not regular password)
   - **Microsoft 365**: Business/enterprise accounts
   - **SendGrid**: Reliable third-party service

### Testing Your Configuration

1. Visit the **Email Diagnostics** page in the app sidebar
2. Check your configuration status
3. Send a test email to verify everything works
4. Review any error messages and helpful troubleshooting hints

### Security Notes

- Never commit real credentials to source control
- Use environment variables or Streamlit Cloud secrets for production
- For Gmail, use App Passwords instead of your regular password
- Ensure your SMTP provider allows the configured sender address

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
- Uses PaddleOCR for text detection (English language model)
- Enhanced spell checking with PySpellChecker (US/UK variants)
- Smart text tokenization ignoring URLs, handles, hashtags
- Scene detection with PySceneDetect
- Video analysis with OpenCV
- Video downloading with yt-dlp
- Full Python traceback capture for debugging
- Mobile-responsive controls for all screen sizes

### UK Spelling Support

The optional UK spelling wordlist is loaded from `assets/spell/en_GB_extra.txt` and includes common British spellings like:
- colour, flavour, honour (vs color, flavor, honor)
- realise, organise, specialise (vs realize, organize, specialize)  
- centre, theatre, metre (vs center, theater, meter)
- licence/practice (noun), license/practise (verb)
- grey, aluminium, aeroplane (vs gray, aluminum, airplane)

If the file is missing, the app continues to work normally with only US spellings.