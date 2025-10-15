"""
Test script to verify <tool_response> masking in training labels.

This script validates that:
1. Tool calls are included in training (model learns to generate them)
2. Tool responses are masked (model doesn't learn to generate them)
3. Text after tool responses is included in training
"""

from ovis.util.constants import IGNORE_ID
from transformers import AutoTokenizer


def test_tool_response_masking():
    """Test that <tool_response> is correctly masked"""

    # Load tokenizer (use local_files_only to avoid HF validation)
    tokenizer = AutoTokenizer.from_pretrained(
        "/workspace/VLMs/Ovis/HF_Repo", trust_remote_code=True, local_files_only=True
    )

    # Test case 1: Identify tool with single response
    test_text_1 = """<think>
Looking at the image, I can see someone on the left wearing a blue jacket.
<tool_call>Identify [103.9,300.0,238.2,477.4]</tool_call><tool_response>Liam Hernandez</tool_response>
The identification confirms this is Liam Hernandez.
</think>
<answer>The person wearing the blue jacket is Liam Hernandez.</answer>"""

    # Test case 2: Multiple tool calls
    test_text_2 = """<think>
First person: <tool_call>Identify [103.9,300.0,238.2,477.4]</tool_call><tool_response>Liam Hernandez</tool_response>
Second person: <tool_call>Identify [216.6,261.7,298.2,473.6]</tool_call><tool_response>Amina Yusuf</tool_response>
Third person: <tool_call>Identify [83.6,270.3,168.6,453.3]</tool_call><tool_response>Aisha Benitez</tool_response>
The three people are identified.
</think>
<answer>The people are Liam Hernandez, Amina Yusuf, and Aisha Benitez.</answer>"""

    # Test case 3: Crop tool (no tool_response, should not mask anything)
    test_text_3 = """<think>
To examine the bottle, I'll use the crop tool.
<tool_call>Crop [234.57, 0.77, 545.81, 287.13]</tool_call>
Upon examining the cropped area, I can see the bottle label clearly.
</think>
<answer>The bottle contains chocolate wine.</answer>"""

    test_cases = [
        ("Single Identify", test_text_1),
        ("Multiple Identify", test_text_2),
        ("Crop tool", test_text_3),
    ]

    print("=" * 80)
    print("Testing <tool_response> Masking")
    print("=" * 80)

    for name, text in test_cases:
        print(f"\n{'=' * 80}")
        print(f"Test Case: {name}")
        print(f"{'=' * 80}")

        # Simulate assistant message
        full_text = f"<|im_start|>assistant\n{text}<|im_end|>"

        # Tokenize
        tokens = tokenizer.encode(full_text, add_special_tokens=False)

        print(f"\nTotal tokens: {len(tokens)}")
        print(f"\nFull text:\n{full_text}")

        # Simulate label generation
        labels = simulate_label_generation(tokenizer, tokens)

        # Analyze labels
        analyze_labels(tokenizer, tokens, labels)

        print()


def simulate_label_generation(tokenizer, input_ids_list):
    """Simulate the label generation logic with masking"""

    labels = [IGNORE_ID] * len(input_ids_list)

    # Find <|im_start|>assistant
    im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    target_sequence = im_start_ids + assistant_ids

    assistant_start_pos = None
    for i in range(len(input_ids_list) - len(target_sequence) + 1):
        if input_ids_list[i : i + len(target_sequence)] == target_sequence:
            assistant_start_pos = i + len(target_sequence)
            break

    if assistant_start_pos is None:
        return labels

    # Skip whitespace
    while assistant_start_pos < len(input_ids_list):
        decoded = tokenizer.decode([input_ids_list[assistant_start_pos]]).strip()
        if decoded:
            break
        assistant_start_pos += 1

    # Find <|im_end|>
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    end_pos = len(input_ids_list)
    for i in range(assistant_start_pos, len(input_ids_list) - len(im_end_ids) + 1):
        if input_ids_list[i : i + len(im_end_ids)] == im_end_ids:
            end_pos = i
            break

    # Set labels
    if end_pos > assistant_start_pos:
        labels[assistant_start_pos:end_pos] = input_ids_list[assistant_start_pos:end_pos]

    # Mask <tool_response>...</tool_response>
    masked_count = mask_sequence(
        tokenizer,
        labels,
        input_ids_list,
        "<tool_response>",
        "</tool_response>",
        assistant_start_pos,
    )

    print(f"\nMasked {masked_count} <tool_response> sequences")

    return labels


def mask_sequence(tokenizer, labels, input_ids_list, start_marker, end_marker, search_start):
    """Find and mask sequences"""

    start_ids = tokenizer.encode(start_marker, add_special_tokens=False)
    end_ids = tokenizer.encode(end_marker, add_special_tokens=False)

    masked_count = 0
    search_pos = search_start

    while search_pos < len(input_ids_list):
        # Find start
        start_found = False
        for i in range(search_pos, len(input_ids_list) - len(start_ids) + 1):
            if input_ids_list[i : i + len(start_ids)] == start_ids:
                start_pos = i
                start_found = True
                break

        if not start_found:
            break

        # Find end
        end_found = False
        for i in range(start_pos + len(start_ids), len(input_ids_list) - len(end_ids) + 1):
            if input_ids_list[i : i + len(end_ids)] == end_ids:
                end_pos = i + len(end_ids)
                end_found = True
                break

        if end_found:
            for i in range(start_pos, end_pos):
                labels[i] = IGNORE_ID
            masked_count += 1
            search_pos = end_pos
        else:
            break

    return masked_count


def analyze_labels(tokenizer, tokens, labels):
    """Analyze which tokens are masked vs trained"""

    print(f"\n{'Token Analysis':^80}")
    print("=" * 80)
    print(f"{'Position':<10} {'Token':<30} {'Label Status':<20}")
    print("-" * 80)

    # Find key sequences
    tool_call_positions = []
    tool_response_positions = []

    for i in range(len(tokens)):
        decoded = tokenizer.decode([tokens[i]])

        # Check if this is part of tool_call or tool_response
        is_tool_call = False
        is_tool_response = False

        # Look for context (5 tokens before and after)
        context_start = max(0, i - 5)
        context_end = min(len(tokens), i + 6)
        context = tokenizer.decode(tokens[context_start:context_end])

        if "<tool_call>" in context or "</tool_call>" in context:
            is_tool_call = True
            tool_call_positions.append(i)

        if "<tool_response>" in context or "</tool_response>" in context:
            is_tool_response = True
            tool_response_positions.append(i)

        # Show status
        status = "TRAINED ✓" if labels[i] != IGNORE_ID else "MASKED ✗"

        # Highlight important tokens
        if is_tool_call:
            marker = " [TOOL_CALL]"
        elif is_tool_response:
            marker = " [TOOL_RESPONSE]"
        else:
            marker = ""

        # Only show every 10th token, or important ones
        if i % 10 == 0 or is_tool_call or is_tool_response or i < 20 or i > len(tokens) - 10:
            print(f"{i:<10} {decoded[:28]:<30} {status:<20} {marker}")

    print("=" * 80)

    # Summary
    trained_tokens = sum(1 for l in labels if l != IGNORE_ID)
    masked_tokens = sum(1 for l in labels if l == IGNORE_ID)

    print(f"\nSummary:")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Trained tokens: {trained_tokens} ({trained_tokens / len(tokens) * 100:.1f}%)")
    print(f"  Masked tokens: {masked_tokens} ({masked_tokens / len(tokens) * 100:.1f}%)")
    print(f"  Tool call positions: {len(tool_call_positions)} occurrences")
    print(f"  Tool response positions: {len(tool_response_positions)} occurrences")

    # Verify correctness
    print(f"\n{'Verification':^80}")
    print("=" * 80)

    # Check that tool_call tokens are trained
    tool_call_trained = 0
    for pos in tool_call_positions:
        if labels[pos] != IGNORE_ID:
            tool_call_trained += 1

    # Check that tool_response tokens are masked
    tool_response_masked = 0
    for pos in tool_response_positions:
        if labels[pos] == IGNORE_ID:
            tool_response_masked += 1

    print(f"✓ Tool calls trained: {tool_call_trained}/{len(tool_call_positions)}")
    print(f"✓ Tool responses masked: {tool_response_masked}/{len(tool_response_positions)}")

    if tool_call_trained == len(tool_call_positions) and tool_response_masked == len(
        tool_response_positions
    ):
        print("\n✅ PASS: Masking is working correctly!")
    else:
        print("\n❌ FAIL: Masking has issues!")


if __name__ == "__main__":
    test_tool_response_masking()
