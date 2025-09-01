# Grammar Checking and Fix Checklist Features

## New Features Added

### 1. Online Grammar Checking (LanguageTool)

The application now includes intelligent grammar checking specifically for credit text using LanguageTool's public API.

#### Features:
- **Smart Credit Detection**: Automatically identifies credit lines containing keywords like "credit", "@", "by:", "created by", etc.
- **Online Grammar Checking**: Integrates with LanguageTool API for professional grammar suggestions
- **Offline Fallback**: Uses difflib + wordfreq for suggestions when online service is unavailable
- **Privacy Notice**: Displays notification when online checking is enabled

#### Configuration:
Set these environment variables to customize the grammar checking:

```bash
export LANGUAGETOOL_API_URL="https://api.languagetool.org/v2/check"  # Default
export LANGUAGETOOL_API_KEY="your_api_key"  # Optional for premium/self-hosted
export LANGUAGETOOL_LANG="en-US"  # Default language
```

#### Usage Example:
When the app detects credit text like "credit to @origincreatr", it will suggest:
- Grammar suggestion: "origincreatr" → "origincreator"

### 2. Fix Checklist UI

A comprehensive checklist system to help users track and fix issues found during video analysis.

#### Features:
- **Interactive Checkboxes**: Check off items as you fix them
- **Session Persistence**: Checkbox states are maintained during your session
- **Bulk Actions**: 
  - "Mark all complete" - Check all items at once
  - "Reset checklist" - Uncheck all items
  - "Download checklist" - Export to Markdown file
- **Progress Tracking**: Visual progress bar showing completion status
- **Structured Items**: Each item includes timestamp and issue description

#### Checklist Items Include:
- Black frame detections
- Flicker/flash instances
- Frozen frame detections
- OCR typos and spelling errors
- Credit text grammar suggestions
- Content mismatches with user notes

### 3. Enhanced Analysis Results

The detailed analysis section now includes:
- Credit grammar warnings count
- Grammar checking method used (LanguageTool online vs offline)
- Structured grammar suggestions with timestamps

## Technical Implementation

### Key Functions Added:

1. **`is_credit_text(text)`** - Detects if text is likely a credit line
2. **`call_languagetool_api(text, use_online)`** - Makes API calls to LanguageTool
3. **`get_offline_suggestions(text)`** - Generates offline suggestions using difflib+wordfreq
4. **`grammarly_like_check_credit_line(text, use_online, cfg)`** - Main grammar checking function
5. **`build_checklist_items(mistakes, credit_grammar_warnings)`** - Creates checklist structure
6. **`export_checklist_to_markdown(checklist_items, video_name)`** - Exports checklist to MD

### Dependencies Added:
- `wordfreq` - For offline grammar suggestions using common English words
- Updated to use `yt-dlp` instead of `pytube` for video downloading

### API Integration:
- 5-second timeout for LanguageTool API calls
- Graceful fallback to offline suggestions on network issues
- Structured error handling and user notifications

## Usage

1. **Enable/Disable Online Grammar Checking**: Use the toggle in the "Grammar Checking Options" section
2. **Privacy Notice**: When enabled, you'll see a notice about data being sent to the API
3. **Upload and Analyze**: Upload your video and provide notes as usual
4. **Review Results**: Check the analysis results for grammar suggestions
5. **Use Checklist**: After analysis, use the Fix Checklist to track your progress
6. **Export Progress**: Download your checklist as a Markdown file for external tracking

## Example Workflow

1. Upload a TikTok video with credit text like "credit to @origincreatr"
2. Enable online grammar checking (default)
3. Run analysis
4. See grammar suggestion: "Credit text grammar suggestion: 'origincreatr' → 'origincreator'"
5. Use the checklist to track fixing this issue
6. Export checklist for reference while editing

This implementation maintains backward compatibility while adding powerful new grammar checking and issue tracking capabilities.