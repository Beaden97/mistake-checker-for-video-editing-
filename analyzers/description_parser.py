"""
Description parser for AI prompt-style user input.

Parses user descriptions to extract:
- General video description
- Expected text content (marked with ":" symbols)
- Analysis instructions (marked with keywords like "Look for:", "Check:", etc.)
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParsedDescription:
    """Structured representation of parsed user description."""
    general_description: str
    expected_text: List[str]
    analysis_instructions: List[str]
    look_for_keywords: List[str]
    raw_description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'general_description': self.general_description,
            'expected_text': self.expected_text,
            'analysis_instructions': self.analysis_instructions,
            'look_for_keywords': self.look_for_keywords,
            'raw_description': self.raw_description
        }


class DescriptionParser:
    """Parser for AI prompt-style video descriptions."""
    
    # Keywords that indicate analysis instructions
    INSTRUCTION_KEYWORDS = [
        'look for', 'check for', 'verify', 'ensure', 'watch for',
        'pay attention to', 'focus on', 'analyze', 'examine',
        'detect', 'find', 'search for', 'monitor', 'skip', 'ignore'
    ]
    
    # Keywords that indicate expected text content
    TEXT_KEYWORDS = [
        'text:', 'caption:', 'title:', 'subtitle:', 'overlay:',
        'expected text:', 'should say:', 'displays:', 'shows text:',
        'text should be:', 'captions:', 'titles:', 'title:'
    ]
    
    def __init__(self):
        """Initialize the parser with compiled regex patterns."""
        # Pattern for text marked with colons (e.g., "Step 1: Boil water" : "Step 2: Add pasta")
        self.colon_text_pattern = re.compile(r'"([^"]+)"', re.IGNORECASE)
        
        # Pattern for instruction keywords
        instruction_pattern = '|'.join(re.escape(kw) for kw in self.INSTRUCTION_KEYWORDS)
        self.instruction_pattern = re.compile(f'({instruction_pattern})[:.]?\\s*(.+?)(?=\\n|$)', re.IGNORECASE | re.MULTILINE)
        
        # Pattern for text keywords
        text_keyword_pattern = '|'.join(re.escape(kw) for kw in self.TEXT_KEYWORDS)
        self.text_keyword_pattern = re.compile(f'({text_keyword_pattern})\\s*(.+?)(?=\\n|$)', re.IGNORECASE | re.MULTILINE)
    
    def extract_expected_text(self, description: str) -> Tuple[List[str], str]:
        """
        Extract expected text content from description.
        
        Args:
            description: Raw description text
            
        Returns:
            Tuple of (extracted_text_list, remaining_description)
        """
        expected_text = []
        remaining_description = description
        
        # Method 1: Extract text in quotes with colon separators
        # e.g., "Step 1: Boil water" : "Step 2: Add pasta"
        colon_matches = self.colon_text_pattern.findall(description)
        if colon_matches:
            expected_text.extend(colon_matches)
            # Remove the entire matched line/section from description
            full_colon_pattern = re.compile(r'.*?' + re.escape('"' + '" : "'.join(colon_matches) + '"') + r'.*?(?=\n|$)', re.DOTALL)
            remaining_description = full_colon_pattern.sub('', remaining_description)
        
        # Method 2: Extract text after text keywords
        # e.g., "Expected text: Welcome to the tutorial"
        text_keyword_matches = self.text_keyword_pattern.findall(description)
        for keyword, text_content in text_keyword_matches:
            # Check if text_content contains colon-separated quoted text
            colon_text_in_content = self.colon_text_pattern.findall(text_content)
            if colon_text_in_content:
                expected_text.extend(colon_text_in_content)
            else:
                # Split on common separators
                text_items = re.split(r'[,;|]', text_content.strip())
                expected_text.extend([item.strip().strip('"\'') for item in text_items if item.strip()])
            
            # Remove the matched pattern from description
            pattern_to_remove = re.escape(keyword) + r'\s*' + re.escape(text_content)
            remaining_description = re.sub(pattern_to_remove, '', remaining_description, flags=re.IGNORECASE)
        
        # Clean up expected text (remove duplicates, empty strings)
        expected_text = list(dict.fromkeys([text.strip() for text in expected_text if text.strip()]))
        
        return expected_text, remaining_description.strip()
    
    def extract_analysis_instructions(self, description: str) -> Tuple[List[str], List[str], str]:
        """
        Extract analysis instructions from description.
        
        Args:
            description: Raw description text
            
        Returns:
            Tuple of (instruction_list, keyword_list, remaining_description)
        """
        instructions = []
        keywords = []
        remaining_description = description
        
        # Find instruction patterns
        instruction_matches = self.instruction_pattern.findall(description)
        for trigger_keyword, instruction_content in instruction_matches:
            instructions.append(instruction_content.strip())
            
            # Extract specific keywords from the instruction
            instruction_keywords = self._extract_keywords_from_instruction(instruction_content)
            keywords.extend(instruction_keywords)
            
            # Remove the matched pattern from description
            pattern_to_remove = re.escape(trigger_keyword) + r'[:.]?\s*' + re.escape(instruction_content)
            remaining_description = re.sub(pattern_to_remove, '', remaining_description, flags=re.IGNORECASE)
        
        # Remove duplicates while preserving order
        instructions = list(dict.fromkeys(instructions))
        keywords = list(dict.fromkeys(keywords))
        
        return instructions, keywords, remaining_description.strip()
    
    def _extract_keywords_from_instruction(self, instruction: str) -> List[str]:
        """
        Extract specific analysis keywords from an instruction.
        
        Args:
            instruction: Instruction text
            
        Returns:
            List of extracted keywords
        """
        keywords = []
        
        # Common analysis terms
        analysis_terms = [
            'timing', 'audio', 'sync', 'text', 'visibility', 'clarity',
            'transitions', 'cuts', 'exposure', 'lighting', 'color',
            'focus', 'stabilization', 'noise', 'artifacts', 'compression',
            'frame rate', 'resolution', 'aspect ratio', 'black frames',
            'flicker', 'freeze', 'credits', 'logos', 'watermarks'
        ]
        
        instruction_lower = instruction.lower()
        for term in analysis_terms:
            if term in instruction_lower:
                keywords.append(term)
        
        return keywords
    
    def clean_general_description(self, description: str) -> str:
        """
        Clean up the general description by removing empty lines and extra whitespace.
        
        Args:
            description: Description text to clean
            
        Returns:
            Cleaned description
        """
        # Remove multiple consecutive newlines
        description = re.sub(r'\n\s*\n', '\n', description)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in description.split('\n') if line.strip()]
        
        return '\n'.join(lines)
    
    def parse(self, description: str) -> ParsedDescription:
        """
        Parse a user description into structured components.
        
        Args:
            description: Raw user description text
            
        Returns:
            ParsedDescription object with structured data
        """
        if not description or not description.strip():
            return ParsedDescription(
                general_description="",
                expected_text=[],
                analysis_instructions=[],
                look_for_keywords=[],
                raw_description=description or ""
            )
        
        # Start with the full description
        working_description = description.strip()
        
        # Extract expected text first
        expected_text, working_description = self.extract_expected_text(working_description)
        
        # Extract analysis instructions
        instructions, keywords, working_description = self.extract_analysis_instructions(working_description)
        
        # Clean up the remaining general description
        general_description = self.clean_general_description(working_description)
        
        return ParsedDescription(
            general_description=general_description,
            expected_text=expected_text,
            analysis_instructions=instructions,
            look_for_keywords=keywords,
            raw_description=description
        )
    
    def format_parsing_preview(self, parsed: ParsedDescription) -> str:
        """
        Format the parsed description for display to users.
        
        Args:
            parsed: ParsedDescription object
            
        Returns:
            Formatted string showing the parsing results
        """
        lines = []
        
        if parsed.general_description:
            lines.append(f"📝 **General Description:**\n{parsed.general_description}")
        
        if parsed.expected_text:
            lines.append(f"📄 **Expected Text ({len(parsed.expected_text)} items):**")
            for i, text in enumerate(parsed.expected_text, 1):
                lines.append(f"  {i}. \"{text}\"")
        
        if parsed.analysis_instructions:
            lines.append(f"🔍 **Analysis Instructions ({len(parsed.analysis_instructions)} items):**")
            for i, instruction in enumerate(parsed.analysis_instructions, 1):
                lines.append(f"  {i}. {instruction}")
        
        if parsed.look_for_keywords:
            lines.append(f"🎯 **Focus Keywords:** {', '.join(parsed.look_for_keywords)}")
        
        return "\n\n".join(lines) if lines else "No structured content detected."


# Example usage and testing
def example_usage():
    """Example of how to use the description parser."""
    parser = DescriptionParser()
    
    # Example 1: Simple description with expected text
    description1 = '''
    This is a cooking tutorial video about making pasta.
    Expected text: "Step 1: Boil water" : "Step 2: Add pasta" : "Step 3: Cook for 10 minutes"
    Look for: timing accuracy, text visibility, audio sync
    '''
    
    parsed1 = parser.parse(description1)
    print("Example 1:")
    print(parser.format_parsing_preview(parsed1))
    print("\n" + "="*50 + "\n")
    
    # Example 2: Complex description with multiple sections
    description2 = '''
    A TikTok dance video featuring 3 dancers in a studio setting.
    The video should have upbeat music and smooth transitions.
    
    Text should be: "Welcome to Dance Studio" : "Learn the moves" : "Follow us for more"
    
    Check for: proper lighting, clear audio, no flicker
    Pay attention to: frame stability, color consistency
    Verify: all text is readable and properly timed
    '''
    
    parsed2 = parser.parse(description2)
    print("Example 2:")
    print(parser.format_parsing_preview(parsed2))


if __name__ == "__main__":
    example_usage()