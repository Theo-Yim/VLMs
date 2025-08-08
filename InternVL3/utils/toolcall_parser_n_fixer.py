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
    target_text: str, annotations: Dict[str, List[str]], threshold: float = 0.5
) -> Optional[str]:
    """
    Find the best matching noun+number for a given target text.
    Only returns a match if there's high confidence (threshold=0.5).

    Args:
        target_text: The text to match against descriptions
        annotations: Dict of noun+number to descriptions
        threshold: Minimum similarity threshold (conservative: 0.5)

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

    # Additional conservative check: only return match if score is significantly high
    if best_match and best_score > threshold:
        return best_match

    return None


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
    not_found_parts = []

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
            # print(f"  Warning: Could not find match for '{part}'")
            fixed_parts.append(part)
            not_found_parts.append(part)

    return crop_prefix + ", ".join(fixed_parts), ", ".join(not_found_parts)


def fix_tool_calling_strings(image_id: str, qna_a: str, anno_str: str) -> str:
    """
    Main function to fix tool-calling strings in qna["A2"].

    Args:
        qna_a2: The A2 string containing tool-calling patterns
        anno_str: The annotation string with correct noun+number patterns

    Returns:
        Fixed A2 string
    """
    # print(f"Processing: {image_id}")
    # Parse annotations
    annotations = parse_annotations(anno_str)
    # Extract all tool-calling patterns
    tool_calls = extract_tool_calls(qna_a)
    # Process tool calls in reverse order to maintain string positions
    fixed_text = qna_a
    for content, start_pos, end_pos in reversed(tool_calls):
        # Check if it needs fixing
        valid_nouns = set(annotations.keys())
        if not is_valid_crop_format(content, valid_nouns):
            print(f"  Needs fixing: {content}")
            fixed_content, not_found = fix_tool_calling_content(content, annotations)
            if fixed_content != content:
                print(f"  Fixed to: {fixed_content}")
            else:
                print(f"  Could not fix: {content}, not_found_parts: {not_found}")

            # Replace in the text
            fixed_text = fixed_text[: start_pos + 1] + fixed_content + fixed_text[end_pos - 1 :]
    return fixed_text


# Example usage and test
def test_function():
    """Test the function with the provided examples."""

    # Example data
    anno_str_1 = """- person 1 [170.75, 91.57, 425.55, 428.55], who is "a woman standing behind the net", "middle standing up", "standing with racket", "a lady standing up", "person standing u"
- person 2 [114.57, 237.84, 435.4, 655.33], who is "dark - haired woman sitting with leg extended and tennis racquet on lap", "the woman sitting down on the right", "woman sitting with leg extended", "right girl on floor", "ditting darker hair\""""

    qna_a2_1 = """<think>
{Crop woman sitting with tennis racquet on her leg}

The woman sitting with a tennis racket on her lap is smiling.
</think>

<answer>
The woman might be smiling because she's enjoying the moment.
</answer>"""

    print("=== TEST 1: Original Example ===")
    print("Original A2:")
    print(qna_a2_1)
    print("\n" + "-" * 50 + "\n")

    fixed_a2_1 = fix_tool_calling_strings("dummy_id", qna_a2_1, anno_str_1)

    print("Fixed A2:")
    print(fixed_a2_1)

    # User's new example - should be more conservative
    anno_str_2 = """- person 1 [103.93, 299.99, 342.08000000000004, 777.4], who is "the back of an older woman with her hair in a barrette with a blue jacket on", "a woman is wearing blue sweater", "the lady with the blue shirt", "woman back in blue", "lady with back to us"
- person 2 [216.58, 261.7, 514.8000000000001, 735.29], who is "the woman in the grey shirt with a watch on her wrist ..", "woman in gray shirt facing camera on right", "woman with gray shirt standing next to man", "a short haired woman in jeans shopping", "gray shirt wearing glasses"
- person 3 [83.6, 270.27, 252.24999999999997, 723.5999999999999], who is "a woman in glasses shops in an open air fruit market", "a woman in a gray coat and scarf"
"""

    qna_a2_2 = """<think>
To determine the type of market the person in the gray coat and scarf is shopping at, I will examine the items visible in the image. The market appears to have various fruits and vegetables, as well as other goods such as baskets and birds in cages. I will zoom in on the fruit and vegetable section to confirm the types of produce available.

{Crop person 3, bananas in front}

Upon closer inspection, there are bananas, oranges, and other fruits prominently displayed, along with a variety of vegetables like cabbages. This suggests a fresh produce market. Additionally, the presence of baskets and birds indicates that this is a general open-air market where various items are sold.

Therefore, the person in the gray coat and scarf is shopping at a fresh produce market.
</think>

<answer>
The person in the gray coat and scarf is shopping at a fresh produce market.
</answer>"""

    print("\n\n=== TEST 2: Conservative Approach (User's Example) ===")
    print("Original A2:")
    print(qna_a2_2)
    print("\n" + "-" * 50 + "\n")

    fixed_a2_2 = fix_tool_calling_strings("dummy_id", qna_a2_2, anno_str_2)

    print("Fixed A2:")
    print(fixed_a2_2)

    anno_str_3 = '- person 1 [187.13, 85.22, 530.01, 367.18000000000006], who is "man in front of lamp", "one right of lamp", "far left man", "striped shirt"\n- person 2 [309.9, 143.67, 879.27, 497.29999999999995], who is "person with legs crossed on right", "blue pants man black shirt", "old guy black shirt", "man in black shirt", "laid back man"\n- person 3 [461.44, 178.32, 1085.95, 586.3199999999999], who is "woman in gray with white sleeves", "person on far right", "girl on far right", "child far right", "ponytail girl"\n- person 4 [20.6, 128.36, 167.59, 429.70000000000005], who is "woman siting by herself", "old lady in the chare", "woman in dark shirt", "old lady by herself", "left person"\n- chair 1 [13.17, 139.77, 181.40999999999997, 482.87], which is "a tan sofa chair"'

    qna_a2_3 = "<think>\nFirst, I need to identify the setting where the people in the image are located. I'll start by looking at the background elements and furniture to get a sense of the environment.\n\n{Crop person 1, tan sofa chair}\n\nFrom the cropped image, I can see that person 1 is sitting on a tan sofa chair. The presence of the sofa chair suggests that this is an indoor setting, likely a living room.\n\nNext, I'll examine other details in the room to confirm the setting.\n\n{Crop desk 1, flower arrangement}\n\nThe desk has a flower arrangement, wine glasses, newspapers, and other casual items. This reinforces the idea of a relaxed, domestic environment.\n\nFinally, looking at the overall layout with other furniture and decorations such as the lamp, windows, and other seating arrangements:\n\n{Crop lamp, windows, other seating}\n\nThe combination of the lamp, windows, and additional seating, along with the casual arrangement of people, strongly indicates that this is a living room setting.\n\n</think>\n<answer>\nThe setting where the person is located is a living room.\n</answer>"

    print("\n\n=== TEST 3: ===")
    print("Original A2:")
    print(qna_a2_3)
    print("\n" + "-" * 50 + "\n")

    fixed_a2_3 = fix_tool_calling_strings("dummy_id", qna_a2_3, anno_str_3)

    print("Fixed A2:")
    print(fixed_a2_3)

    return fixed_a2_1, fixed_a2_3


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
    fixed_multi = fix_tool_calling_strings("dummy_id", test_multi, anno_str)
    print("Fixed:", fixed_multi.strip())

    # Test non-existent entity
    test_nonexistent = """<think>
{Crop person 1, banana}
</think>"""

    print("\nNon-existent entity test:")
    print("Original:", test_nonexistent.strip())
    fixed_nonexistent = fix_tool_calling_strings("dummy_id", test_nonexistent, anno_str)
    print("Fixed:", fixed_nonexistent.strip())

    # Get statistics
    annotations = parse_annotations(anno_str)
    stats = get_statistics(test_multi, fixed_multi)
    print(f"\nStatistics: {stats}")
    print(f"Validation passed: {validate_fixed_strings(fixed_multi, annotations)}")
