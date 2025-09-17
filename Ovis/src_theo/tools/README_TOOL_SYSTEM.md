# Tool System for Ovis2.5

Train Ovis2.5 to use tools during generation with visual feedback loops.

**Key Features**:
- **Trained Tools**: Model learns tool usage during training (e.g., crop tool)
- **LLM-Driven Tools**: Instant tool addition via descriptions (no training needed)
- **Real-time Execution**: Tools execute during generation, not after
- **Visual Feedback**: Tool results fed back into generation context immediately

## System Architecture

```
Standard Conversation Data → ConversationDataset → Training
                 ↓                    ↓
    Tool Call Detection     Multimodal Sequence → [text, tool_call, cropped_img, text]
                                     ↓
                             Ovis2.5 Processing
                             ↓                    ↓
                       Text Tokenization    Visual Processing
                       (includes coords)    (original + crops)
```

## Components

### 1. **CropTool** (`src_theo/utils/crop_tool.py`)
- `extract_tool_calls(text)` - Parse `<tool_call>Crop [x1,y1,x2,y2]</tool_call>`
- `crop_image(image, coords)` - Crop image regions 
- `create_multimodal_content(text, image)` - Create training sequences
- `parse_and_replace_tool_calls(text)` - Format conversion

### 2. **ToolRegistry** (`conversation_dataset.py`)
- Detects available tools and routes processing
- Extensible: `register_tool()`, `detect_tool_calls()`, `process_text_with_tools()`

### 3. **ConversationDataset** (Modified)
- Detects tool calls in assistant messages
- Creates multimodal sequences via ToolRegistry
- Handles multimodal content during tokenization

## Multimodal Sequence Example

```python
# ConversationDataset + CropTool creates:
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "<tool_call>Crop [100,100,200,200]</tool_call>"},
    {"type": "image", "image": <cropped_region>},  # Right after tool call
    {"type": "text", "text": " The person is walking."}
  ]
}
```

## Training Behavior

**Model Learns:**
- **Coordinate Generation**: Output valid bounding boxes as text tokens
- **Spatial Mapping**: Map descriptions ("person in blue") to regions  
- **Tool Integration**: Use crop tools within conversation flow
- **Visual Feedback**: See crop results immediately after coordinate generation

**Gradient Flow:**
- Coordinates treated as regular text tokens (full gradient flow)
- Visual feedback enables coordinate accuracy learning

## Technical Design

**Key Decisions:**
1. **Coordinates as Text**: Treat `[100,100,200,200]` as regular tokens
2. **Visual Feedback**: Cropped images right after tool calls in sequences
3. **Native Processing**: Use Ovis2.5's existing multimodal capabilities
4. **Modular Tools**: ToolRegistry for extensibility

**Ovis2.5 Spatial Capabilities:**
- SigLIP vision tower with spatial patch embeddings
- 2D rotary embeddings for positional awareness  
- `grid_thws` maintains spatial structure

## Training Flow

```python
# 1. ConversationDataset detects tool calls in assistant messages
# 2. ToolRegistry routes to CropTool.create_multimodal_content()
# 3. Creates sequence: [text, tool_call, cropped_image, remaining_text]
# 4. Ovis2.5 processes all images together via torch.cat(pixel_values, dim=0)
# 5. Text tokenization includes coordinates as regular tokens
# 6. Model learns coordinate generation through visual feedback
```

## Data Format

**Input:**
```json
{"from": "gpt", "value": "<tool_call>Crop [100,100,200,200]</tool_call> Walking"}
```

**Training Sequence:**
```python
[
  {"type": "text", "text": "<tool_call>Crop [100,100,200,200]</tool_call>"},
  {"type": "image", "image": <cropped_region>},
  {"type": "text", "text": " Walking"}
]
```

## Expected Capabilities

**Coordinate Generation:**
- Generate meaningful bounding box coordinates as text
- Map descriptions ("person in blue") to spatial regions
- Handle multiple crop operations in reasoning chains

**Tool Integration:**
- Use crop tools within conversational responses
- Perform analysis on cropped regions
- Integrate crop insights with broader context

## Adding New Tools

1. **Create Tool Class:**
```python
class NewTool:
    def extract_tool_calls(self, text): ...
    def create_multimodal_content(self, text, image): ...
```

2. **Register in ToolRegistry:**
```python
self.register_tool('newtool', NewTool())
```

System automatically detects and routes new tool calls.

## Inference Usage

Noe that LLM will stop the inference when it sees </tool_call> and calls the tool. Return values of tool will be instantly fed into the inference process right after </tool_call> before LLM continues the process.

### Basic Usage

```python
from src_theo.tools.inference_integration_v2 import (
    chat_with_tool_execution_batch,
    chat_with_tool_execution_streaming,
    GenerationConfig
)

# Load model
model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2.5-9B", trust_remote_code=True)
config = GenerationConfig(tool_timeout=15.0)

# Batch generation (production-focused)
response, thinking, history = chat_with_tool_execution_batch(
    model=model,
    prompt="Examine this image in detail",
    images=[image],
    config=config
)

# Streaming generation (interactive UI-focused)
response, thinking, history = chat_with_tool_execution_streaming(
    model=model,
    prompt="Examine this image in detail",
    images=[image],
    config=config
)

# Model flow (example with Crop tool):
# 1. Generates: "Let me examine this person. <tool_call>Crop [100,100,200,200]</tool_call>"
# 2. System IMMEDIATELY detects "</tool_call>" → pauses generation
# 3. System executes crop → feeds cropped image back into model context
# 4. Model continues with cropped image: " The person is wearing a blue shirt."

# Works with ANY registered tool.
```

### Tool Types

**1. Trained Tools (use_tool_descriptions=False)**
```python
# Model uses learned patterns from training
response = chat_with_tool_execution_batch(
    model=model,
    prompt="Analyze this image",
    images=[image],
    use_tool_descriptions=False  # Default trained behavior
)
```

**2. LLM-Driven Tools (use_tool_descriptions=True)**
```python
# Register new tool instantly
tool_registry = ToolRegistry()
tool_registry.register_tool("draw", DrawingTool(), {
    "name": "draw",
    "description": "Draw shapes or annotations on the image",
    "usage": "<tool_call>Draw [action]</tool_call>",
    "example": "<tool_call>Draw [red box around person]</tool_call>"
})

# LLM automatically uses tools based on descriptions
response = chat_with_tool_execution_batch(
    model=model,
    prompt="Analyze and highlight regions",
    images=[image],
    use_tool_descriptions=True  # Enable LLM-driven tool selection
)
```

### Adding New Tools

**Method 1: Built-in Registration (for training)**
```python
# Add to _setup_tools() in inference_integration_v2.py
self.tools["new_tool"] = NewTool()
```

**Method 2: Runtime Registration (instant tools)**
```python
class NewTool:
    def extract_tool_calls(self, text): ...
    def execute(self, image, parameters): ...

tool_registry.register_tool("new_tool", NewTool(), {
    "name": "new_tool",
    "description": "Tool description for LLM",
    "usage": "<tool_call>NewTool [params]</tool_call>"
})
```

### Debug Logging

```bash
export OVIS_TOOL_DEBUG=true
python your_script.py
```

### Examples

See `example_usage.py` for complete examples including:
- Basic batch/streaming generation
- Trained vs LLM-driven tool usage
- Multi-turn conversations
- Performance benchmarking
