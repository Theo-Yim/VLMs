# Crop Tool System for Ovis2.5

Train Ovis2.5 to generate bounding box coordinates and use cropped regions for detailed analysis.

**Approach**: 
- Treat coordinates as regular text tokens
- Insert cropped images after tool calls in training
- Learn through visual feedback: generate coords → see crop → continue reasoning

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

**Real-time Tool Execution**: Execute tool calls during generation, just like training.

### Real-time Chat with Tools
```python
from src_theo.tools.inference_integration import (
    chat_with_tool_execution_streaming,  # Recommended - streaming experience
    chat_with_tool_execution            # Alternative - batch control
)

# Option 1: Streaming Generation (Recommended - Real-time streaming)
response, thinking, history = chat_with_tool_execution_streaming(
    model=model,
    prompt="Examine the person in detail", 
    images=[image],
    do_sample=True,
    temperature=0.7,
    max_new_tokens=1024
)

# Option 2: Batch Generation (Simple and precise control with batch delivery)  
response, thinking, history = chat_with_tool_execution(
    model=model,
    prompt="Examine the person in detail", 
    images=[image],
    do_sample=True,
    temperature=0.7,
    max_new_tokens=1024
)

# Model flow (example with Crop tool):
# 1. Generates: "Let me examine this person. <tool_call>Crop [100,100,200,200]</tool_call>"
# 2. System IMMEDIATELY detects "</tool_call>" → pauses generation  
# 3. System executes crop → feeds cropped image back into model context
# 4. Model continues with cropped image: " The person is wearing a blue shirt."

# Works with ANY registered tool: DrawingTool, AnnotationTool, etc.
```

### Debug Logging

Enable debug logging to see tool execution details. Multiple methods available, but below is the recommended method.

Setting the Environment Variable (Recommended):
```bash
# Set environment variable before running Python
export OVIS_TOOL_DEBUG=true
python your_script.py
```


### Scalable Tool System

**InferenceToolRegistry** automatically detects and executes multiple tool types:
- `detect_tool_calls()` - Finds any registered tool calls in text
- `execute_tool_call()` - Routes to appropriate tool execution
- `create_multimodal_context()` - Builds context with tool results

### Adding New Inference Tools

1. **Create Tool Class** (same as training):
```python
class DrawingTool:
    def extract_tool_calls(self, text): ...
    def draw_on_image(self, image, parameters): ...
```

2. **Register in InferenceToolRegistry**:
```python
# Add to _setup_available_tools() in inference_integration.py
try:
    from drawing_tool import DrawingTool
    self.register_tool('draw', DrawingTool())
except ImportError:
    pass
```

3. **Add Execution Logic**:
```python
# Add to execute_tool_call() method
elif tool_name == 'draw':
    return tool_instance.draw_on_image(original_image, tool_call['parameters'])
```

**System automatically handles**:
- Multi-tool detection in single text
- Tool execution in order
- Context building with results
- Generation resume with tool_call returns

### How It Works

**Training**: `"text <tool_call>ToolName [...]</tool_call> <tool_result> continuation"`

**Inference**: Same! Tool calls are executed in real-time and results are fed back into the generation context immediately.

**Key difference from normal chat**: Tool calls are executed **during generation**, not after. The model sees tool results before continuing to generate text.
