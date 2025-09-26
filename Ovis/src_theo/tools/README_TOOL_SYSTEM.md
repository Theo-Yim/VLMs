# Tool System for Ovis2.5

**Zero-configuration tool system** with auto-detection and real-time execution during generation.

## Key Features

- **🔍 Auto-Detection**: Automatically finds and registers all `*_tool.py` files
- **⚡ Real-time Execution**: Tools execute during generation with pause/resume
- **🖼️ Visual Feedback**: Tool results (images, text) instantly fed back to model
- **📝 LLM Guidance**: Tool docstrings automatically guide model usage
- **🏗️ Unified Architecture**: All tools follow the same simple ToolBase pattern

## System Architecture

```
Generation: "I see a person <tool_call>Crop [100,100,200,200]</tool_call>"
                                           ↓ PAUSE & EXECUTE
                              ToolRegistry (scans *_tool.py files)
                                           ↓
                               CropTool.execute() → cropped_image
                                           ↓ RESUME WITH RESULT
Generation: " The person is wearing a blue jacket."
```

## Core Components

### **ToolBase** (`tool_base.py`)
```python
class ToolBase:
    def extract_tool_call(self, text: str) -> Dict  # Parse tool calls
    def execute(self, image, parameters) -> Dict    # Execute & return result
```

### **ToolRegistry** (`tool_base.py`)
```python
registry = ToolRegistry()  # Auto-detects all *_tool.py files
registry.tools             # {'crop': CropTool(), 'identify': IdentifyTool()}
registry.detect_tool_call(text)     # Find tool calls in text
registry.execute_tool_call(call, image)  # Execute tool calls
```

### **Available Tools**

**CropTool** (`crop_tool.py`): `<tool_call>Crop [x1,y1,x2,y2]</tool_call>`
- Crops image regions → returns cropped image

**IdentifyTool** (`mock_id_tool.py`): `<tool_call>Identify [x1,y1,x2,y2]</tool_call>`
- Mock identification → returns text result

## How It Works

### **1. Auto-Detection on Startup**
```python
# ToolRegistry scans tools/ directory for *_tool.py files
# Automatically imports and registers any ToolBase subclasses
registry = ToolRegistry()
# Found: crop_tool.py → CropTool → registered as "crop"
# Found: mock_id_tool.py → IdentifyTool → registered as "identify"
```

### **2. Real-time Execution During Generation**
```python
# Model generates: "I see a person <tool_call>Crop [100,100,200,200]</tool_call>"
# ↓ System detects "</tool_call>" → PAUSE generation
# ↓ Execute: CropTool.crop_image(image, [100,100,200,200]) → cropped_image
# ↓ Add cropped_image to conversation context
# ↓ RESUME generation: " The person is wearing a blue jacket."
```

### **3. Technical Implementation**
- **Token-by-token** generation inside `<tool_call>` for precise detection
- **Batch generation** outside tool calls for efficiency
- **Context rebuilding** after tool execution via `model.preprocess_inputs()`
- **Structured results**: Tools return `{"type": "image|text", "content": ...}`

## Creating New Tools

### **Step 1: Create Tool File**
```python
# save as: draw_tool.py

from .tool_base import ToolBase

class DrawTool(ToolBase):
    """
    Tool for drawing annotations on images.
    Usage: <tool_call>Draw [description]</tool_call>
    Example: <tool_call>Draw [red circle around person]</tool_call>
    """

    def __init__(self):
        self.tool_call_pattern = r"<tool_call>Draw \[([^]]+)\]</tool_call>"

    def extract_tool_call(self, text: str):
        match = re.search(self.tool_call_pattern, text)
        if match:
            return {
                "parameters": [match.group(1)],
                "tool_name": "draw",
                "tool_instance": self
            }
        return None

    def execute(self, image, parameters):
        description = parameters[0]
        # Draw annotation on image based on description
        annotated_image = self.draw_annotation(image, description)
        return {"type": "image", "content": annotated_image}
```

### **Step 2: That's It!**
```bash
# Just save the file as draw_tool.py in the tools/ directory
# ToolRegistry automatically detects and registers it on next startup
```

**The docstring is crucial** - it tells the LLM:
- When to use the tool
- Exact syntax: `<tool_call>Draw [...]</tool_call>`
- Parameter format and examples

## Usage Examples

### **Basic Inference**
```python
from src_theo.tools.inference_integration import chat_with_tool_execution

# Load model
model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2.5-9B", trust_remote_code=True)

# Chat with automatic tool execution
response, thinking, history = chat_with_tool_execution(
    model=model,
    prompt="<image>Examine this person's clothing in detail",
    images=[image],
    temperature=0.6,
    max_new_tokens=1024
)

# Model might generate:
# "I can see a person. <tool_call>Crop [100,100,300,400]</tool_call>
# The person is wearing a blue denim jacket with silver buttons."
```

### **Multi-turn Conversations**
```python
history = []

# Turn 1
response1, _, history = chat_with_tool_execution(
    model=model,
    prompt="<image>What do you see?",
    images=[image],
    history=history
)

# Turn 2 - maintains context including any tool results
response2, _, history = chat_with_tool_execution(
    model=model,
    prompt="Focus on the person's accessories",
    history=history
)
```

### **Debug Mode**
```python
from src_theo.tools.inference_integration import enable_debug_logging
enable_debug_logging()

# Or set environment variable
export OVIS_TOOL_DEBUG=true
```

## Tool Reference

### **Current Tools**

| Tool | Syntax | Purpose | Returns |
|------|--------|---------|---------|
| **Crop** | `<tool_call>Crop [x1,y1,x2,y2]</tool_call>` | Crop image regions | Cropped image |
| **Identify** | `<tool_call>Identify [x1,y1,x2,y2]</tool_call>` | Mock identification | Text result |

### **Tool Result Types**
```python
{"type": "image", "content": PIL_Image}        # Visual result
{"type": "text", "content": "text response"}  # Text result
```

## Quick Start

1. **Create a tool**: `my_tool.py` with ToolBase subclass
2. **Add docstring**: LLM usage instructions
3. **Save in tools/**: Auto-detected on startup
4. **Use in inference**: Model automatically uses based on context

## Examples

See `example_usage.py` for complete working examples.
