"""
Test script to validate the complete tool system for both training and inference.

This script tests:
1. Image-returning tools (CropTool) - training and inference
2. Text-returning tools (IdentifyTool) - training and inference
3. Tool registry auto-detection
4. Training data processing
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image
import numpy as np

# Import tool system
from src_theo.tools.tool_base import ToolRegistry
from src_theo.tools.crop_tool import CropTool
from src_theo.tools.mock_id_tool import IdentifyTool


def create_test_image(width=800, height=600):
    """Create a simple test image"""
    # Create a gradient image
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img_array[i, j] = [i % 256, j % 256, (i + j) % 256]
    return Image.fromarray(img_array)


def test_tool_registry():
    """Test 1: Auto-detection and registration"""
    print("\n" + "=" * 80)
    print("TEST 1: Tool Registry Auto-Detection")
    print("=" * 80)

    registry = ToolRegistry()

    # Check if tools were auto-detected
    assert "crop" in registry.tools, "CropTool should be auto-registered"
    assert "mock_id" in registry.tools, "IdentifyTool should be auto-registered"

    print(f"✅ Auto-detected tools: {list(registry.tools.keys())}")
    print(f"✅ CropTool registered: {registry.tools['crop']}")
    print(f"✅ IdentifyTool registered: {registry.tools['mock_id']}")


def test_image_tool_inference():
    """Test 2: Image-returning tool (CropTool) - Inference"""
    print("\n" + "=" * 80)
    print("TEST 2: Image-Returning Tool (CropTool) - Inference")
    print("=" * 80)

    registry = ToolRegistry()
    test_image = create_test_image()

    # Simulate inference scenario: model generates tool call
    generated_text = "Let me examine this region. <tool_call>Crop [100,100,300,400]</tool_call>"

    # 1. Detect tool call
    tool_call = registry.detect_tool_call(generated_text)
    assert tool_call is not None, "Should detect crop tool call"
    assert tool_call["tool_name"] == "crop", "Should identify as crop tool"

    print(f"✅ Detected tool call: {tool_call['tool_name']}")
    print(f"   Parameters: {tool_call['parameters']}")

    # 2. Execute tool
    result = registry.execute_tool_call(tool_call, test_image)
    assert result is not None, "Tool should return result"
    assert result["type"] == "image", "Should return image type"
    assert isinstance(result["content"], Image.Image), "Content should be PIL Image"

    print(f"✅ Tool executed successfully")
    print(f"   Result type: {result['type']}")
    print(f"   Result image size: {result['content'].size}")

    # 3. Verify cropped region size
    expected_size = (200, 300)  # (300-100, 400-100)
    assert result["content"].size == expected_size, f"Cropped size should be {expected_size}"

    print(f"✅ Crop dimensions correct: {result['content'].size}")


def test_text_tool_inference():
    """Test 3: Text-returning tool (IdentifyTool) - Inference"""
    print("\n" + "=" * 80)
    print("TEST 3: Text-Returning Tool (IdentifyTool) - Inference")
    print("=" * 80)

    registry = ToolRegistry()
    test_image = create_test_image()

    # Simulate inference scenario: model generates tool call
    generated_text = "Let me identify this person. <tool_call>Identify [200,150,500,450]</tool_call>"

    # 1. Detect tool call
    tool_call = registry.detect_tool_call(generated_text)
    assert tool_call is not None, "Should detect identify tool call"
    assert tool_call["tool_name"] == "identify", "Should identify as identify tool"

    print(f"✅ Detected tool call: {tool_call['tool_name']}")
    print(f"   Parameters: {tool_call['parameters']}")

    # 2. Execute tool
    result = registry.execute_tool_call(tool_call, test_image)
    assert result is not None, "Tool should return result"
    assert result["type"] == "text", "Should return text type"
    assert isinstance(result["content"], str), "Content should be string"

    print(f"✅ Tool executed successfully")
    print(f"   Result type: {result['type']}")
    print(f"   Result content: '{result['content']}'")

    # 3. Verify response format for inference
    expected_in_response = f"<tool_response>{result['content']}</tool_response>"
    print(f"✅ Should be inserted as: {expected_in_response}")


def test_image_tool_training():
    """Test 4: Image-returning tool (CropTool) - Training"""
    print("\n" + "=" * 80)
    print("TEST 4: Image-Returning Tool (CropTool) - Training Data Processing")
    print("=" * 80)

    registry = ToolRegistry()
    test_image = create_test_image()

    # Training data format for image tool
    training_text = (
        "Let me examine this region. "
        "<tool_call>Crop [100,100,300,400]</tool_call><image>\n"
        "The person is wearing a blue jacket."
    )

    print(f"📝 Training data input:")
    print(f"   Text: {training_text[:80]}...")

    # Process for training
    result_images, cleaned_text = registry.process_tools_for_training(
        training_text, test_image
    )

    # Verify results
    assert len(result_images) == 1, "Should extract 1 cropped image"
    assert isinstance(result_images[0], Image.Image), "Should be PIL Image"

    # Verify <image> marker was removed
    assert "<image>" not in cleaned_text, "<image> marker should be removed"
    assert "<tool_call>Crop [100,100,300,400]</tool_call>" in cleaned_text, "Tool call should remain"

    expected_cleaned = (
        "Let me examine this region. "
        "<tool_call>Crop [100,100,300,400]</tool_call>\n"
        "The person is wearing a blue jacket."
    )
    assert cleaned_text == expected_cleaned, "Text should only have <image> removed"

    print(f"✅ Extracted {len(result_images)} image(s)")
    print(f"   Image size: {result_images[0].size}")
    print(f"✅ Cleaned text (removed <image> marker):")
    print(f"   {cleaned_text[:80]}...")
    print(f"✅ Tool call preserved in text: ✓")


def test_text_tool_training():
    """Test 5: Text-returning tool (IdentifyTool) - Training"""
    print("\n" + "=" * 80)
    print("TEST 5: Text-Returning Tool (IdentifyTool) - Training Data Processing")
    print("=" * 80)

    registry = ToolRegistry()
    test_image = create_test_image()

    # Training data format for text tool
    training_text = (
        "Let me identify this person. "
        "<tool_call>Identify [200,150,500,450]</tool_call><tool_response>Theo</tool_response>\n"
        "This person is Theo and he is smiling."
    )

    print(f"📝 Training data input:")
    print(f"   Text: {training_text}")

    # Process for training
    result_images, cleaned_text = registry.process_tools_for_training(
        training_text, test_image
    )

    # Verify results
    assert len(result_images) == 0, "Text tools should not return images"
    assert cleaned_text == training_text, "Text should remain unchanged for text tools"

    print(f"✅ No images extracted (text tool): {len(result_images)} images")
    print(f"✅ Text unchanged (correct behavior):")
    print(f"   {cleaned_text}")
    print(f"✅ <tool_response> preserved: ✓")


def test_multiple_tools_training():
    """Test 6: Multiple tool calls in training data"""
    print("\n" + "=" * 80)
    print("TEST 6: Multiple Tool Calls - Training Data Processing")
    print("=" * 80)

    registry = ToolRegistry()
    test_image = create_test_image()

    # Mixed image and text tools
    training_text = (
        "First, let me crop the region. "
        "<tool_call>Crop [100,100,300,400]</tool_call><image>\n"
        "Now let me identify the person. "
        "<tool_call>Identify [200,150,500,450]</tool_call><tool_response>Theo</tool_response>\n"
        "This is Theo wearing a blue jacket."
    )

    print(f"📝 Training data with mixed tools:")
    print(f"   {training_text[:100]}...")

    # Process for training
    result_images, cleaned_text = registry.process_tools_for_training(
        training_text, test_image
    )

    # Verify results
    assert len(result_images) == 1, "Should extract 1 image from crop tool"
    assert "<image>" not in cleaned_text, "<image> marker should be removed"
    assert "<tool_response>Theo</tool_response>" in cleaned_text, "<tool_response> should remain"

    print(f"✅ Extracted {len(result_images)} image(s) from crop tool")
    print(f"✅ Cleaned text:")
    print(f"   {cleaned_text[:100]}...")
    print(f"✅ <image> removed: ✓")
    print(f"✅ <tool_response> preserved: ✓")


def test_system_prompt_generation():
    """Test 7: System prompt generation for inference"""
    print("\n" + "=" * 80)
    print("TEST 7: System Prompt Generation")
    print("=" * 80)

    registry = ToolRegistry()
    system_prompt = registry.get_system_prompt_tools()

    assert system_prompt, "System prompt should be generated"
    assert "Crop" in system_prompt, "Should mention Crop tool"
    assert "Identify" in system_prompt, "Should mention Identify tool"
    assert "<tool_call>" in system_prompt, "Should explain tool call format"

    print(f"✅ System prompt generated ({len(system_prompt)} chars)")
    print(f"📝 Preview:")
    print(f"   {system_prompt[:200]}...")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "=" * 80)
    print("TOOL SYSTEM VALIDATION TEST SUITE")
    print("=" * 80)

    try:
        test_tool_registry()
        test_image_tool_inference()
        test_text_tool_inference()
        test_image_tool_training()
        test_text_tool_training()
        test_multiple_tools_training()
        test_system_prompt_generation()

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\n✅ Image-returning tools (CropTool): WORKING")
        print("✅ Text-returning tools (IdentifyTool): WORKING")
        print("✅ Training data processing: WORKING")
        print("✅ Inference detection & execution: WORKING")
        print("✅ Auto-detection: WORKING")
        print("\nThe tool system is ready for both training and inference! 🚀\n")

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
