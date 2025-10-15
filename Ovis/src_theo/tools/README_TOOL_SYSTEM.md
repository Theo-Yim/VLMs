# Tool System for Ovis2.5

**Universal tool system** supporting any tool with both image and text return types. Features auto-detection and real-time execution during training and inference.

## Overview

The tool system supports two return types:
- **Image-Returning Tools** - Return processed images (e.g., Crop, Draw, Blur)
- **Text-Returning Tools** - Return text results (e.g., Identify, Calculate, OCR)

Both types work seamlessly with automatic detection and zero configuration needed.

## Key Features

- **🔍 Auto-Detection**: Automatically finds and registers all `*_tool.py` files
- **⚡ Real-time Execution**: Tools execute during generation with pause/resume
- **🎯 Universal Design**: Supports any tool with image or text returns
- **📝 LLM Guidance**: Tool docstrings automatically guide model usage
- **🏗️ Simple Interface**: All tools inherit from ToolBase
- **🚀 Flexible**: Create inference-only OR full-featured tools

## Quick Start

### Using Tools During Inference

```python
from src_theo.tools.inference_integration import chat_with_tool_execution

# Chat with automatic tool execution
response, thinking, history = chat_with_tool_execution(
    model=model,
    prompt="<image>Examine this person's face closely.",
    images=[image]
)
```

The model will automatically:
1. Generate tool call: `<tool_call>ToolName Parameters</tool_call>`
2. System pauses, executes tool, adds result to context
3. Model resumes with tool result available

### Training Data Format

**Image-Returning Tools:**
```json
{
  "from": "gpt",
  "value": "<tool_call>Crop [100,100,300,400]</tool_call><image>\nThe person is wearing blue."
}
```
- `<image>` marker indicates where tool's image result appears

**Text-Returning Tools:**
```json
{
  "from": "gpt",
  "value": "<tool_call>Identify [100,100,300,400]</tool_call><tool_response>James</tool_response>\nThis is James."
}
```
- `<tool_response>...</tool_response>` contains tool's text result
- Kept as-is in training (no processing needed)

## Creating New Tools

### Inference-Only Tool (Minimal)

For tools that only need to work during inference (no training needed):

```python
# Ovis/src_theo/tools/your_tool.py

from .tool_base import ToolBase
from PIL import Image
import re

class YourTool(ToolBase):
    """
    Tool description for LLM.
    Usage: <tool_call>YourTool [parameters]</tool_call>
    """

    def __init__(self):
        # Let's assume that parameter is [x, ...] format
        # For inference (without markers)
        self.tool_call_pattern = r"<tool_call>YourTool \[([0-9.,\s]+)\]</tool_call>"

        # For training (with <image> for image tools, <tool_response> for text tools)
        self.tool_call_with_image_pattern = r"<tool_call>YourTool \[([0-9.,\s]+)\]</tool_call><image>"

    def extract_tool_call(self, text: str):
        """REQUIRED - Extract tool call for inference"""
        match = re.search(self.tool_call_pattern, text)
        if match:
            params = [float(x.strip()) for x in match.group(1).split(",")]
            return {
                "parameters": params,
                "tool_name": "your_tool",
                "tool_instance": self
            }
        return None

    def execute(self, image, parameters):
        """REQUIRED - Execute tool - return image or text"""
        # For image-returning tool:
        result_image = process_image(image, parameters)
        return {"type": "image", "content": result_image}

        # For text-returning tool:
        # result_text = analyze(image, parameters)
        # return {"type": "text", "content": result_text}
```

That's it! This tool:
- ✅ Works during inference
- ✅ Auto-detected and registered
- ✅ Won't be used during training (no `extract_tool_calls()`)

### Full-Featured Tool (Inference + Training)

For tools that should work in both inference and training, additionally implement extract_tool_calls:

```python
class YourTool(ToolBase):

    ...

    # implement this function
    def extract_tool_calls(self, text: str):
        """OPTIONAL - Extract tool calls for training"""
        tool_calls = []
        for match in re.finditer(self.tool_call_with_image_pattern, text):
            params = [float(x.strip()) for x in match.group(1).split(",")]
            tool_calls.append({
                "start_pos": match.start(),
                "end_pos": match.end(),
                "full_match": match.group(0),
                "parameters": params
            })
        return tool_calls
```

This tool:
- ✅ Works during inference
- ✅ Works during training
- ✅ Auto-detected and registered
- ✅ Included in system prompts

## How It Works

### Inference Flow

**The docstring is crucial** - it tells the LLM:
- When to use the tool
- Exact syntax: `<tool_call>ToolName Parameters</tool_call>`
- Parameter format and examples

**When model generates a tool call:**
1. System detects `</tool_call>` → **PAUSE** generation
2. Execute tool → get result (image or text)
3. Add result to context:
   - **Image tools**: Add image to visual tokens, rebuild context
   - **Text tools**: Insert `<tool_response>text</tool_response>`
4. **RESUME** generation with tool result available

### Training Flow

**During dataset processing:**
1. Detect tool calls in assistant messages
2. Execute tools to get results
3. Process based on return type:
   - **Image tools**: Add image to visual tokens, remove `<image>` marker from text
   - **Text tools**: Keep `<tool_response>` in text, but **MASK from loss computation**
4. Model trains with tool results present in context (image tools: visual tokens; text tools: masked text)

## Architecture

### Core Components

**ToolBase** - Base class for all tools:
```python
class ToolBase:
    def extract_tool_call(text) -> Optional[Dict]    # REQUIRED: Parse single call (inference)
    def extract_tool_calls(text) -> List[Dict]       # OPTIONAL: Parse all calls (training)
    def execute(image, parameters) -> Optional[Dict]  # REQUIRED: Execute and return result
```

**Method Requirements:**
- ✅ **`extract_tool_call()`** - **REQUIRED** for inference detection
- ⚠️ **`extract_tool_calls()`** - **OPTIONAL** for training support (defaults to `return []` if not implemented)
- ✅ **`execute()`** - **REQUIRED** for tool execution

**Key Insight:** You can create inference-only tools by skipping `extract_tool_calls()` implementation!

**ToolRegistry** - Auto-detection and management:
```python
registry = ToolRegistry()  # Auto-detects all *_tool.py files
registry.tools             # {'crop': CropTool(), 'identify': IdentifyTool(), ...}
registry.detect_tool_call(text)                      # Find tool call (inference)
registry.execute_tool_call(call, image)              # Execute tool
registry.process_tools_for_training(text, image)     # Process for training
```

### File Structure

```
Ovis/src_theo/tools/
├── tool_base.py              # ToolBase + ToolRegistry
├── crop_tool.py              # Image-returning tool example
├── mock_id_tool.py           # Text-returning tool example
├── inference_integration.py  # Real-time execution during generation
└── your_tool.py              # Your custom tool (auto-detected!)

Ovis/ovis/train/dataset/
└── conversation_dataset.py   # Training data processing
```

## Debug Mode

Enable debug logging to see tool execution in detail:

```python
from src_theo.tools.inference_integration import enable_debug_logging

enable_debug_logging()
# Or set environment variable: OVIS_TOOL_DEBUG=true
```

## Important Notes

### Why `<image>` marker is needed in dataset generation and removed during training:

**When creating training data:**
- ✅ **DO add** `<image>` marker: `<tool_call>Crop [...]</tool_call><image>` for image-returning tools
- This tells the system "execute this tool and insert the returned image here"

**During training data processing:**
- System executes the tool → gets the image result
- Adds image to visual tokens
- Removes `<image>` marker from text (so model doesn't learn to generate it)
- Model learns: after `</tool_call>`, new image appears in context
- At inference: system replicates this by adding image to context

**Key:** The `<image>` marker is essential for data preparation but removed before model training.

### Why `<tool_response>` is needed for text-returning-tool in training dataset:

**Dataset format:**
```json
"<tool_call>Identify [x,y,x2,y2]</tool_call><tool_response>Name</tool_response>\nThis is Name."
```

**Training processing:**
- `<tool_response>Name</tool_response>` is kept in text but **MASKED from loss** (no backprop)
- Model observes the format but doesn't learn to generate it
- Model learns: tool call → system provides response → use response in reasoning

**Why masking is critical:**
- ❌ **Without masking**: Model learns to predict/hallucinate tool responses
- ✅ **With masking**: Model learns to wait for system-provided responses
- **Result**: Perfect training-inference alignment - model delegates to tool, doesn't predict

**Implementation**: See `Ovis/ovis/train/dataset/conversation_dataset.py:_generate_labels()`


## Testing

Verify the tool system is working correctly:

```bash
python Ovis/src_theo/tools/test_tool_system.py
```

**Run this test when:**
- ✅ After adding a new tool
- ✅ After modifying tool processing logic
- ✅ To verify training data processing works correctly
- ✅ Before deploying to production

All 7 tests should pass (auto-detection, inference, training for both image and text tools).

## Summary

This universal tool system:
- ✅ Supports any tool returning images or text
- ✅ Auto-detects tools from `*_tool.py` files
- ✅ Executes tools in real-time during inference
- ✅ Processes tool results correctly during training (if `extract_tool_calls()` implemented)
- ✅ Maintains perfect training-inference alignment
- ✅ Supports inference-only tools (skip `extract_tool_calls()`)
- ✅ Requires zero configuration - just add a tool file!

Add new tools by simply creating a `*_tool.py` file. That's it!
