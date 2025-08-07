import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


def parse_annotations(anno_str: str) -> Dict[str, List[str]]:
    """
    Parse annotation string to extract noun+number patterns and their descriptions.

    Returns:
        Dict mapping "noun number" to list of descriptions
    """
    annotations = {}

    # Split by lines and process each annotation
    lines = anno_str.strip().split("\n")

    for line in lines:
        if not line.strip() or not line.startswith("-"):
            continue

        # Extract noun + number pattern (e.g., "person 1", "person 2")
        match = re.search(r"-\s*(\w+\s+\d+)", line)
        if not match:
            continue

        noun_number = match.group(1)

        # Extract descriptions within quotes
        descriptions = re.findall(r'"([^"]+)"', line)

        annotations[noun_number] = descriptions

    return annotations


def similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity score between two texts."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def find_best_match(
    target_text: str, annotations: Dict[str, List[str]], threshold: float = 0.3
) -> Optional[str]:
    """
    Find the best matching noun+number for a given target text.

    Args:
        target_text: The text to match against descriptions
        annotations: Dict of noun+number to descriptions
        threshold: Minimum similarity threshold

    Returns:
        Best matching noun+number or None if no good match found
    """
    best_match = None
    best_score = threshold

    for noun_number, descriptions in annotations.items():
        for desc in descriptions:
            score = similarity_score(target_text, desc)
            if score > best_score:
                best_score = score
                best_match = noun_number

    return best_match


def extract_tool_calls(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract all tool-calling patterns from text.

    Returns:
        List of tuples: (tool_call_content, start_pos, end_pos)
    """
    tool_calls = []
    pattern = r"\{([^}]+)\}"

    for match in re.finditer(pattern, text):
        content = match.group(1).strip()
        tool_calls.append((content, match.start(), match.end()))

    return tool_calls


def is_valid_crop_format(content: str, valid_nouns: set) -> bool:
    """
    Check if tool-calling content follows correct "Crop noun number" format.
    """
    # Remove "Crop" prefix if present
    if content.lower().startswith("crop"):
        content = content[4:].strip()

    # Split by comma to handle multiple entities
    parts = [part.strip() for part in content.split(",")]

    for part in parts:
        # Check if it matches "noun number" pattern
        match = re.match(r"^(\w+)\s+(\d+)$", part)
        if not match:
            return False

        noun_number = f"{match.group(1)} {match.group(2)}"
        if noun_number not in valid_nouns:
            return False

    return True


def fix_tool_calling_content(content: str, annotations: Dict[str, List[str]]) -> str:
    """
    Fix a single tool-calling content string.
    """
    # Remove "Crop" prefix if present
    crop_prefix = ""
    if content.lower().startswith("crop"):
        crop_prefix = "Crop "
        content = content[4:].strip()

    # Split by comma to handle multiple entities
    parts = [part.strip() for part in content.split(",")]
    fixed_parts = []

    for part in parts:
        # Check if it already follows correct format
        match = re.match(r"^(\w+)\s+(\d+)$", part)
        if match:
            noun_number = f"{match.group(1)} {match.group(2)}"
            if noun_number in annotations:
                fixed_parts.append(part)
                continue

        # Try to find best match in annotations
        best_match = find_best_match(part, annotations)
        if best_match:
            fixed_parts.append(best_match)
        else:
            # If no match found, keep original but log warning
            print(f"  Warning: Could not find match for '{part}'")
            fixed_parts.append(part)

    return crop_prefix + ", ".join(fixed_parts)


def fix_tool_calling_strings(image_id: str, qna_a2: str, anno_str: str) -> str:
    """
    Main function to fix tool-calling strings in qna["A2"].

    Args:
        qna_a2: The A2 string containing tool-calling patterns
        anno_str: The annotation string with correct noun+number patterns

    Returns:
        Fixed A2 string
    """
    # Parse annotations
    annotations = parse_annotations(anno_str)
    # Extract all tool-calling patterns
    tool_calls = extract_tool_calls(qna_a2)
    # Process tool calls in reverse order to maintain string positions
    fixed_text = qna_a2
    for content, start_pos, end_pos in reversed(tool_calls):
        # Check if it needs fixing
        valid_nouns = set(annotations.keys())
        if not is_valid_crop_format(content, valid_nouns):
            print(f"Processing: {image_id}")
            print(f"  Needs fixing: {content}")
            fixed_content = fix_tool_calling_content(content, annotations)
            print(f"  Fixed to: {fixed_content}")

            # Replace in the text
            fixed_text = fixed_text[: start_pos + 1] + fixed_content + fixed_text[end_pos - 1 :]
    return fixed_text


# def fix_text_strings(image_id: str, question:str, answer: str, anno_str: str) -> str:
#     # Parse annotations
#     annotations = parse_annotations(anno_str)
#     # anno_str_2 = anno_str[2:].split('- ')
#     # anno_str_2 = [re.sub(r" \[[^\]]*\]", "", _.strip()) for _ in anno_str_2]
#     # anno_str_2 = re.sub(r" \[[^\]]*\]", "", anno_str)
#     # Extract all tool-calling patterns
#     tool_calls = extract_tool_calls(answer)
#     tool_call_contents = [answer[start_pos:end_pos] for _, start_pos, end_pos in tool_calls]
#     text_only = re.sub(r'\{.*?\}', "{Tool call}", answer)
#     wrong_pattern = [_ for _ in annotations.keys() if _ in text_only]

#     wrong_pattern_str = "\n".join([f"{_} should be fixed by rephrasing the annotations {annotations[_]}" for _ in wrong_pattern])

#     prompt = """Refine the text below:
# I will provide the original question and LLM's response. The original response includes thought process to reach to the answer to the original question. Model is meant to generate response only from visuals, not descriptions. Thus, existence of object annotations should not be mentioned in the response, and they should be rephrased. Response can include {{Tool call}} and this tool call inside braces should be remained as is. Response should not include noun plus number pattern. Therefore, noun + number pattern should be fixed. Give me the fixed version of response.
# # Original Question:
# {question}

# # Original Response:
# {answer}

# # Wrong patterns:
# {wrong_pattern_str}
# """
#     prompt = prompt.format(question=question, answer=answer, wrong_pattern_str=wrong_pattern_str)
    
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant that refines multimodal reasoning responses.",
#         },
#         {"role": "user", "content": prompt},
#     ]
#     # Tokenize and generate
#     prompt = self.tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     with torch.inference_mode():
#         outputs = self.model.generate(prompt, self.sampling_params)


# Example usage and test
def test_function():
    """Test the function with the provided example."""

    # Example data
    anno_str = """- person 1 [170.75, 91.57, 425.55, 428.55], who is "a woman standing behind the net", "middle standing up", "standing with racket", "a lady standing up", "person standing u"
- person 2 [114.57, 237.84, 435.4, 655.33], who is "dark - haired woman sitting with leg extended and tennis racquet on lap", "the woman sitting down on the right", "woman sitting with leg extended", "right girl on floor", "ditting darker hair\""""

    qna_a2 = """<think>
{Crop dark - haired woman sitting with leg extended and tennis racquet on lap}

The woman sitting with a tennis racket on her lap is smiling, which suggests a sense of contentment or enjoyment. The tennis setting indicates they might be playing or about to play tennis, an activity that often brings happiness. Her relaxed posture and the context of a casual game could contribute to her smile, showing she's having a good time.
</think>

<answer>
The woman sitting with a tennis racket on her lap might be smiling because she is enjoying the moment, possibly before or after a game of tennis. The relaxed setting and the company of friends likely contribute to her happiness.
</answer>"""

    print("Original A2:")
    print(qna_a2)
    print("\n" + "=" * 50 + "\n")

    # Fix the tool-calling strings
    fixed_a2 = fix_tool_calling_strings(qna_a2, anno_str)

    print("Fixed A2:")
    print(fixed_a2)

    return fixed_a2
    """
    JavaScript implementation of the tool-calling string fixer.
    This version handles all edge cases more efficiently.
    """
    js_code = """
    function parseAnnotations(annoStr) {
        const annotations = {};
        const lines = annoStr.trim().split('\\n');
        
        for (const line of lines) {
            if (!line.trim() || !line.startsWith('-')) continue;
            
            const match = line.match(/-\\s*(\\w+\\s+\\d+)/);
            if (!match) continue;
            
            const nounNumber = match[1];
            const descriptions = [...line.matchAll(/"([^"]+)"/g)].map(m => m[1]);
            annotations[nounNumber] = descriptions;
        }
        
        return annotations;
    }

    function similarityScore(text1, text2) {
        const words1 = text1.toLowerCase().split(/\\s+/);
        const words2 = text2.toLowerCase().split(/\\s+/);
        
        const set1 = new Set(words1);
        const set2 = new Set(words2);
        
        const intersection = new Set([...set1].filter(x => set2.has(x)));
        const union = new Set([...set1, ...set2]);
        
        return intersection.size / union.size;
    }

    function findBestMatch(targetText, annotations, threshold = 0.3) {
        let bestMatch = null;
        let bestScore = threshold;
        
        for (const [nounNumber, descriptions] of Object.entries(annotations)) {
            for (const desc of descriptions) {
                const score = similarityScore(targetText, desc);
                if (score > bestScore) {
                    bestScore = score;
                    bestMatch = nounNumber;
                }
            }
        }
        
        return bestMatch;
    }

    function fixToolCallingStrings(qnaA2, annoStr) {
        const annotations = parseAnnotations(annoStr);
        
        const pattern = /\\{([^}]+)\\}/g;
        const matches = [];
        let match;
        
        while ((match = pattern.exec(qnaA2)) !== null) {
            matches.push({
                content: match[1].trim(),
                startPos: match.index,
                endPos: match.index + match[0].length
            });
        }
        
        let fixedText = qnaA2;
        
        for (let i = matches.length - 1; i >= 0; i--) {
            const toolCall = matches[i];
            
            let content = toolCall.content;
            if (content.toLowerCase().startsWith('crop')) {
                content = content.slice(4).trim();
            }
            
            const parts = content.split(',').map(part => part.trim());
            const fixedParts = [];
            let needsFix = false;
            
            for (const part of parts) {
                const nounNumberMatch = part.match(/^(\\w+)\\s+(\\d+)$/);
                if (nounNumberMatch && annotations[`${nounNumberMatch[1]} ${nounNumberMatch[2]}`]) {
                    fixedParts.push(part);
                } else {
                    const bestMatch = findBestMatch(part, annotations);
                    if (bestMatch) {
                        fixedParts.push(bestMatch);
                        needsFix = true;
                    } else {
                        fixedParts.push(part);
                        needsFix = true;
                    }
                }
            }
            
            if (needsFix) {
                const fixedContent = `Crop ${fixedParts.join(', ')}`;
                fixedText = fixedText.substring(0, toolCall.startPos) + 
                           `{${fixedContent}}` + 
                           fixedText.substring(toolCall.endPos);
            }
        }
        
        return fixedText;
    }
    
    return fixToolCallingStrings(arguments[0], arguments[1]);
    """

    # In a real implementation, you would execute this JavaScript code
    # For now, use the Python implementation
    return fix_tool_calling_strings(qna_a2, anno_str)


# Advanced version with LLM assistance for better matching
async def fix_with_llm_assistance(content: str, annotations: Dict[str, List[str]]) -> str:
    """
    Use a small LLM to help match rephrased descriptions with correct noun+number.

    Example LLM prompt for semantic matching:
    """
    llm_prompt = f"""
    Given these annotations:
    {json.dumps(annotations, indent=2)}
    
    And this description: "{content}"
    
    Which annotation key (noun + number) best matches this description?
    Return only the key (e.g., "person 1") or "none" if no good match.
    """

    # This would make an API call to a small LLM
    # For now, fall back to similarity-based approach
    return fix_tool_calling_content(content, annotations)


# Additional utility functions
def validate_fixed_strings(fixed_text: str, annotations: Dict[str, List[str]]) -> bool:
    """Validate that all tool-calling strings are now in correct format."""
    tool_calls = extract_tool_calls(fixed_text)
    valid_nouns = set(annotations.keys())

    for content, _, _ in tool_calls:
        if not is_valid_crop_format(content, valid_nouns):
            return False
    return True


def get_statistics(original_text: str, fixed_text: str) -> Dict[str, int]:
    """Get statistics about the fixing process."""
    original_calls = extract_tool_calls(original_text)
    fixed_calls = extract_tool_calls(fixed_text)

    changes_made = 0
    for i, (orig, _, _) in enumerate(original_calls):
        if i < len(fixed_calls) and orig != fixed_calls[i][0]:
            changes_made += 1

    return {
        "total_tool_calls": len(original_calls),
        "changes_made": changes_made,
        "success_rate": (len(original_calls) - changes_made) / len(original_calls)
        if original_calls
        else 1.0,
    }


if __name__ == "__main__":
    # Run comprehensive tests
    result = test_function()

    # Test edge cases
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    anno_str = """- person 1 [170.75, 91.57, 425.55, 428.55], who is "a woman standing behind the net", "middle standing up", "standing with racket", "a lady standing up", "person standing u"
- person 2 [114.57, 237.84, 435.4, 655.33], who is "dark - haired woman sitting with leg extended and tennis racquet on lap", "the woman sitting down on the right", "woman sitting with leg extended", "right girl on floor", "ditting darker hair\""""

    # Test multiple entities
    test_multi = """<think>
{Crop person 1, dark - haired woman sitting with leg extended and tennis racquet on lap}
</think>"""

    test_multi = """<think>\nTo understand the relationship between the woman standing and the two seated women, let's analyze the image. \n\nFirst, the woman standing is holding a tennis racket and is positioned behind the net, which might indicate she is ready to play or has just finished playing. The two seated women are also holding tennis rackets and are dressed in tennis attire. Their relaxed posture suggests they might be taking a break.\n\nGiven their similar attire and equipment, it's possible they are part of the same group, perhaps friends or teammates. The setting appears to be a casual tennis match or practice session.\n\nTo get a closer look at their expressions and possible interactions, let's crop the image to focus on their faces and upper bodies.\n\n{Crop person 1, right woman}\n</think>\n\n<think>\nLooking at the cropped image, the expressions of the women suggest familiarity and comfort with each other. The woman standing appears to be in a relaxed stance, and the seated women are smiling, which could indicate a friendly relationship.\n\nConsidering the context of a tennis setting, it's plausible that these women are friends or teammates enjoying a game together. Their similar attire and the shared activity support this idea.\n\nTherefore, the likely relationship between the woman standing and the two seated women is that they are friends or teammates engaged in a tennis activity.\n</think>\n\n<answer>\nThe likely relationship between the woman standing and the two seated women is that they are friends or teammates.\n</answer>"""

    print("Multi-entity test:")
    print("Original:", test_multi.strip())
    fixed_multi = fix_tool_calling_strings(test_multi, anno_str)
    print("Fixed:", fixed_multi.strip())

    # Test non-existent entity
    test_nonexistent = """<think>
{Crop person 1, banana}
</think>"""

    print("\nNon-existent entity test:")
    print("Original:", test_nonexistent.strip())
    fixed_nonexistent = fix_tool_calling_strings(test_nonexistent, anno_str)
    print("Fixed:", fixed_nonexistent.strip())

    # Get statistics
    annotations = parse_annotations(anno_str)
    stats = get_statistics(test_multi, fixed_multi)
    print(f"\nStatistics: {stats}")
    print(f"Validation passed: {validate_fixed_strings(fixed_multi, annotations)}")
