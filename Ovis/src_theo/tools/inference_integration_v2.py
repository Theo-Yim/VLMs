"""
Tool execution for Ovis2.5 generation.
Implements both batch and streaming approaches with comprehensive tool support.

Key Features:
- Device caching to avoid repeated queries
- Efficient text parsing with regex compilation
- Memory-efficient token counting
- Resource pooling and reuse
- Configuration management
"""

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Pattern

import torch
from PIL import Image
from transformers import TextIteratorStreamer

logger = logging.getLogger(__name__)
DEBUG_TOOL_EXECUTION = os.getenv("OVIS_TOOL_DEBUG", "false").lower() == "true"


@dataclass
class GenerationConfig:
    """Configuration for generation parameters"""
    tool_timeout: float = 10.0
    adaptive_batch_size: int = 6
    token_by_token_threshold: bool = True
    max_generation_cycles: int = 10


class DeviceManager:
    """Caches model device to avoid repeated queries"""

    def __init__(self, model):
        self._device = None
        self._model = model

    @property
    def device(self):
        if self._device is None:
            self._device = next(self._model.parameters()).device
        return self._device


class TextAnalyzer:
    """Text analysis with compiled regex patterns"""

    def __init__(self):
        # Compile regex patterns once for performance
        self._tool_call_pattern = re.compile(r'<tool_call>')
        self._tool_close_pattern = re.compile(r'</tool_call>')
        self._last_analyzed_text = ""
        self._last_result = False

    def is_inside_tool_call(self, text: str) -> bool:
        """Tool call detection with caching"""
        if not text:
            return False

        # Cache recent analysis to avoid repeated computation
        if text == self._last_analyzed_text:
            return self._last_result

        open_count = len(self._tool_call_pattern.findall(text))
        close_count = len(self._tool_close_pattern.findall(text))

        result = open_count > close_count

        # Cache result
        self._last_analyzed_text = text
        self._last_result = result

        return result

    def tool_just_completed(self, partial_response: str, full_text: str) -> bool:
        """Check if a tool call just completed in this response chunk"""
        return (
            "</tool_call>" in partial_response and
            not self.is_inside_tool_call(full_text)
        )


class TokenCounter:
    """Memory-efficient token counting"""

    def __init__(self):
        self._count = 0

    def add_tokens(self, new_tokens: List[int]):
        """Add new tokens to count"""
        self._count += len(new_tokens)

    def reset(self):
        """Reset counter"""
        self._count = 0

    @property
    def count(self) -> int:
        return self._count


class GenerationController:
    """Controls generation flow with resource management"""

    def __init__(self, model, config: GenerationConfig):
        self.model = model
        self.config = config
        self.device_manager = DeviceManager(model)
        self.text_analyzer = TextAnalyzer()
        self.token_counter = TokenCounter()

    def get_adaptive_batch_size(self, current_text: str, remaining_tokens: int) -> int:
        """Get adaptive batch size based on current context"""
        if self.config.token_by_token_threshold and self.text_analyzer.is_inside_tool_call(current_text):
            return 1  # Token-by-token inside tool calls
        return min(self.config.adaptive_batch_size, remaining_tokens)

    def prepare_generation_kwargs(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        grid_thws: Optional[torch.Tensor],
        batch_size: int,
        **kwargs
    ) -> Dict:
        """Prepare generation arguments with consistent parameters"""
        return {
            "inputs": input_ids,
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "max_new_tokens": batch_size,
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 0.6),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "enable_thinking": kwargs.get("enable_thinking", False),
            "enable_thinking_budget": kwargs.get("enable_thinking_budget", False),
            "thinking_budget": kwargs.get("thinking_budget", 2048),
            "eos_token_id": self.model.text_tokenizer.eos_token_id,
            "pad_token_id": self.model.text_tokenizer.pad_token_id,
        }

    def move_tensors_to_device(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thws: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Move tensors to device"""
        device = self.device_manager.device

        input_ids = input_ids.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        if grid_thws is not None:
            grid_thws = grid_thws.to(device)

        return input_ids, pixel_values, grid_thws


class ToolRegistry:
    """Tool registry with minimal overhead"""

    def __init__(self):
        self.tools = {}
        self._setup_tools()

    def _setup_tools(self):
        """Initialize available tools"""
        try:
            from src_theo.tools.crop_tool import CropTool
            self.tools["crop"] = CropTool()
        except ImportError:
            logger.warning("CropTool not available")

    def detect_tool_calls(self, text: str) -> List[Dict]:
        """Detect tool calls with minimal processing"""
        if not text or not isinstance(text, str):
            return []

        all_tool_calls = []
        for tool_name, tool_instance in self.tools.items():
            if hasattr(tool_instance, "extract_tool_calls"):
                try:
                    tool_calls = tool_instance.extract_tool_calls(text)
                    for tc in tool_calls:
                        tc["tool_name"] = tool_name
                        tc["tool_instance"] = tool_instance
                    all_tool_calls.extend(tool_calls)
                except Exception as e:
                    logger.error(f"Error detecting {tool_name} tools: {e}")

        return sorted(all_tool_calls, key=lambda x: x.get("start_pos", 0))

    def execute_tool_call(self, tool_call: Dict, image: Image.Image):
        """Execute tool call with error handling"""
        tool_instance = tool_call["tool_instance"]
        tool_name = tool_call["tool_name"]

        if tool_name == "crop":
            return tool_instance.crop_image(image, tool_call["coordinates"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def create_multimodal_context(self, text: str, images: List[Image.Image], executed_tools: List[Dict]) -> List[Dict]:
        """Create multimodal context"""
        if executed_tools and "crop" in self.tools and images:
            crop_tool = self.tools["crop"]
            if hasattr(crop_tool, "create_multimodal_content"):
                return crop_tool.create_multimodal_content(text, images[0])

        return [{"type": "text", "text": text}]


def chat_with_tool_execution_batch(
    model,
    prompt: str,
    history: Optional[List[Dict]] = None,
    images: Optional[List[Image.Image]] = None,
    videos: Optional[List[List[Image.Image]]] = None,
    config: Optional[GenerationConfig] = None,
    max_new_tokens: int = 1024,
    min_pixels: int = 448 * 448,
    max_pixels: int = 1792 * 1792,
    **kwargs,
) -> Tuple[str, Optional[str], List[Dict]]:
    """
    Batch generation with tool execution.

    Key features:
    - Device caching
    - Efficient text parsing
    - Memory-efficient token counting
    - Minimal object creation
    """
    # Initialize with configuration
    config = config or GenerationConfig()
    generation_controller = GenerationController(model, config)
    tool_registry = ToolRegistry()

    # Initialize history and prepare messages
    if history is None:
        history = []

    # Prepare content for current user message
    content = []
    current_images = images.copy() if images else []

    if images:
        for image in images:
            content.append({"type": "image", "image": image})

    if videos:
        for video_frames in videos:
            content.append({"type": "video", "video": video_frames})

    content.append({"type": "text", "text": prompt})

    user_message = {"role": "user", "content": content if len(content) > 1 else prompt}
    full_messages = history + [user_message]

    # Initialize generation state
    response_text = ""
    thinking = None
    executed_tools = []
    processed_tool_calls = 0

    # Initial preprocessing
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        full_messages,
        add_generation_prompt=True,
        enable_thinking=kwargs.get("enable_thinking", False),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Move to device once
    input_ids, pixel_values, grid_thws = generation_controller.move_tensors_to_device(
        input_ids, pixel_values, grid_thws
    )

    # Generation loop with optimizations
    generation_controller.token_counter.reset()

    while generation_controller.token_counter.count < max_new_tokens:
        # Get optimized batch size
        remaining_tokens = max_new_tokens - generation_controller.token_counter.count
        batch_size = generation_controller.get_adaptive_batch_size(response_text, remaining_tokens)

        # Generate with optimized parameters
        with torch.no_grad():
            generation_kwargs = generation_controller.prepare_generation_kwargs(
                input_ids, pixel_values, grid_thws, batch_size, **kwargs
            )

            outputs = model.generate(**generation_kwargs)
            new_tokens = outputs[0][input_ids.size(1):]
            generation_controller.token_counter.add_tokens(new_tokens.tolist())

            # Decode and analyze
            partial_response = model.text_tokenizer.decode(new_tokens, skip_special_tokens=True)
            response_text += partial_response

            # Check for tool completion
            tool_executed = False
            if generation_controller.text_analyzer.tool_just_completed(partial_response, response_text):
                tool_calls = tool_registry.detect_tool_calls(response_text)

                # Execute new tool calls
                if tool_calls and len(tool_calls) > processed_tool_calls and current_images:
                    new_tool_calls = tool_calls[processed_tool_calls:]

                    for tool_call in new_tool_calls:
                        tool_name = tool_call["tool_name"]
                        if DEBUG_TOOL_EXECUTION:
                            logger.debug(f"🔧 Executing {tool_name} tool")

                        try:
                            # Execute tool
                            original_image = current_images[0]
                            tool_result = tool_registry.execute_tool_call(tool_call, original_image)

                            if tool_name == "crop":
                                current_images.append(tool_result)
                                tool_executed = True

                            executed_tools.append(tool_call)
                            processed_tool_calls += 1

                        except Exception as e:
                            logger.warning(f"Tool execution failed: {e}")

                    # Rebuild context if tools were executed
                    if tool_executed:
                        multimodal_context = tool_registry.create_multimodal_context(
                            response_text, current_images, executed_tools
                        )

                        assistant_partial = {"role": "assistant", "content": multimodal_context}
                        updated_messages = full_messages + [assistant_partial]

                        # Reprocess inputs
                        input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                            updated_messages,
                            add_generation_prompt=False,
                            enable_thinking=kwargs.get("enable_thinking", False),
                            min_pixels=min_pixels,
                            max_pixels=max_pixels,
                        )

                        # Move to device
                        input_ids, pixel_values, grid_thws = generation_controller.move_tensors_to_device(
                            input_ids, pixel_values, grid_thws
                        )

            # Check for EOS
            if model.text_tokenizer.eos_token_id in new_tokens:
                break

            # Update input_ids for next iteration (only if no tool executed)
            if not tool_executed:
                input_ids = outputs[0].unsqueeze(0)

    # Parse thinking and response
    if kwargs.get("enable_thinking", False) and "<think>" in response_text and "</think>" in response_text:
        thinking_start = response_text.find("<think>") + 7
        thinking_end = response_text.find("</think>")
        thinking = response_text[thinking_start:thinking_end].strip()
        response_text = response_text[thinking_end + 8:].strip()

    # Clean up response
    response_text = response_text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

    # Create assistant message for history
    assistant_message = {"role": "assistant", "content": response_text}
    updated_history = history + [user_message, assistant_message]

    return response_text, thinking, updated_history


class ToolAwareStreamer(TextIteratorStreamer):
    """Streaming with reduced lock contention and memory usage"""

    def __init__(
        self,
        tokenizer,
        tool_registry: ToolRegistry,
        text_analyzer: TextAnalyzer,
        skip_prompt: bool = True,
        skip_special_tokens: bool = True,
        on_tool_detected: Optional[Callable] = None,
        **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens, **decode_kwargs)

        self.tool_registry = tool_registry
        self.text_analyzer = text_analyzer
        self.on_tool_detected = on_tool_detected

        # State management
        self.accumulated_text = ""
        self.tool_pause_requested = False
        self.generation_should_pause = threading.Event()
        self.tool_execution_complete = threading.Event()
        self.tool_execution_complete.set()

        # Reduced lock contention with RLock
        self._lock = threading.RLock()

        # Tool tracking
        self.detected_tools = []
        self.processed_tool_count = 0

    def put(self, value):
        """Put method with minimal lock time"""
        # Fast path: decode outside lock
        if isinstance(value, torch.Tensor):
            if len(value.shape) > 1:
                value = value[0]
            new_text = self.tokenizer.decode(value, skip_special_tokens=self.skip_special_tokens)
        else:
            new_text = str(value)

        # Critical section: minimize lock time
        with self._lock:
            self.accumulated_text += new_text

            # Quick tool completion check
            if self._check_for_tool_completion():
                self.tool_pause_requested = True
                self.generation_should_pause.set()

                if self.on_tool_detected:
                    self.on_tool_detected(self.accumulated_text)

        # Continue with normal streaming
        super().put(value)

    def _check_for_tool_completion(self) -> bool:
        """Tool completion check"""
        if not self.accumulated_text:
            return False

        try:
            current_tools = self.tool_registry.detect_tool_calls(self.accumulated_text)

            if len(current_tools) > self.processed_tool_count:
                if self.text_analyzer.tool_just_completed("</tool_call>", self.accumulated_text):
                    self.detected_tools = current_tools
                    return True
        except Exception as e:
            logger.error(f"Error checking tool completion: {e}")

        return False

    def reset_for_new_generation(self):
        """Reset streamer state"""
        with self._lock:
            self.accumulated_text = ""
            self.tool_pause_requested = False
            self.generation_should_pause.clear()
            self.tool_execution_complete.set()
            self.detected_tools = []
            self.processed_tool_count = 0

            # Reset text analyzer cache
            self.text_analyzer._last_analyzed_text = ""
            self.text_analyzer._last_result = False

    def should_pause_generation(self) -> bool:
        return self.tool_pause_requested

    def get_current_text(self) -> str:
        with self._lock:
            return self.accumulated_text

    def get_detected_tools(self) -> List[Dict]:
        with self._lock:
            return self.detected_tools.copy()

    def notify_tool_execution_complete(self, processed_count: int):
        with self._lock:
            self.processed_tool_count = processed_count
            self.tool_pause_requested = False
            self.generation_should_pause.clear()
            self.tool_execution_complete.set()


def chat_with_tool_execution_streaming(
    model,
    prompt: str,
    history: Optional[List[Dict]] = None,
    images: Optional[List[Image.Image]] = None,
    videos: Optional[List[List[Image.Image]]] = None,
    config: Optional[GenerationConfig] = None,
    max_new_tokens: int = 1024,
    min_pixels: int = 448 * 448,
    max_pixels: int = 1792 * 1792,
    **kwargs,
) -> Tuple[str, Optional[str], List[Dict]]:
    """
    Streaming generation with tool execution.

    Key features:
    - Reduced lock contention
    - Efficient text analysis caching
    - Resource pooling
    - Memory-efficient streaming
    """
    # Initialize components
    config = config or GenerationConfig()
    generation_controller = GenerationController(model, config)
    tool_registry = ToolRegistry()

    # Initialize history and prepare messages
    if history is None:
        history = []

    # Prepare content
    content = []
    current_images = images.copy() if images else []

    if images:
        for image in images:
            content.append({"type": "image", "image": image})

    if videos:
        for video_frames in videos:
            content.append({"type": "video", "video": video_frames})

    content.append({"type": "text", "text": prompt})

    user_message = {"role": "user", "content": content if len(content) > 1 else prompt}
    full_messages = history + [user_message]

    # Initialize state
    executed_tools = []
    final_response_text = ""
    thinking = None

    # Setup streamer
    def on_tool_detected_callback(current_text: str):
        if DEBUG_TOOL_EXECUTION:
            logger.debug(f"🔧 Tool detected: {len(current_text)} chars")

    streamer = ToolAwareStreamer(
        tokenizer=model.text_tokenizer,
        tool_registry=tool_registry,
        text_analyzer=generation_controller.text_analyzer,
        skip_prompt=True,
        skip_special_tokens=True,
        on_tool_detected=on_tool_detected_callback
    )

    # Generation loop with cycle limit
    current_conversation = full_messages.copy()
    generation_cycles = 0

    while generation_cycles < config.max_generation_cycles:
        generation_cycles += 1
        streamer.reset_for_new_generation()

        if DEBUG_TOOL_EXECUTION:
            logger.debug(f"🚀 Streaming cycle {generation_cycles}")

        # Preprocess inputs
        input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            current_conversation,
            add_generation_prompt=True,
            enable_thinking=kwargs.get("enable_thinking", False),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # Move to device
        input_ids, pixel_values, grid_thws = generation_controller.move_tensors_to_device(
            input_ids, pixel_values, grid_thws
        )

        # Start streaming generation
        generation_kwargs = generation_controller.prepare_generation_kwargs(
            input_ids, pixel_values, grid_thws, max_new_tokens, **kwargs
        )
        generation_kwargs["streamer"] = streamer

        # Generate in thread with timeout
        generation_thread = threading.Thread(
            target=lambda: model.generate(**generation_kwargs),
            daemon=True
        )
        generation_thread.start()

        # Monitor for tools
        tool_executed = False
        while generation_thread.is_alive():
            if streamer.should_pause_generation():
                if DEBUG_TOOL_EXECUTION:
                    logger.debug("🔧 Tool detected - executing")

                detected_tools = streamer.get_detected_tools()
                generation_thread.join(timeout=config.tool_timeout)

                if generation_thread.is_alive():
                    logger.warning("Generation thread timeout")

                # Execute tools
                if detected_tools and len(detected_tools) > len(executed_tools) and current_images:
                    new_tools = detected_tools[len(executed_tools):]

                    for tool_call in new_tools:
                        tool_name = tool_call["tool_name"]
                        try:
                            original_image = current_images[0]
                            tool_result = tool_registry.execute_tool_call(tool_call, original_image)

                            if tool_name == "crop":
                                current_images.append(tool_result)

                            executed_tools.append(tool_call)
                            tool_executed = True

                        except Exception as e:
                            logger.error(f"Tool execution failed: {e}")

                # Rebuild context
                if tool_executed:
                    partial_response = streamer.get_current_text()
                    final_response_text += partial_response

                    multimodal_context = tool_registry.create_multimodal_context(
                        final_response_text, current_images, executed_tools
                    )

                    assistant_partial = {"role": "assistant", "content": multimodal_context}
                    current_conversation = full_messages + [assistant_partial]

                    streamer.notify_tool_execution_complete(len(executed_tools))
                    break
                else:
                    streamer.notify_tool_execution_complete(len(executed_tools))

            time.sleep(0.005)  # Reduced sleep time for better responsiveness

        # Complete generation if no tools executed
        if not tool_executed:
            generation_thread.join()
            final_response_text = streamer.get_current_text()
            break

    # Parse thinking and response
    if kwargs.get("enable_thinking", False) and "<think>" in final_response_text and "</think>" in final_response_text:
        thinking_start = final_response_text.find("<think>") + 7
        thinking_end = final_response_text.find("</think>")
        thinking = final_response_text[thinking_start:thinking_end].strip()
        final_response_text = final_response_text[thinking_end + 8:].strip()

    # Clean up response
    final_response_text = final_response_text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

    # Create assistant message for history
    assistant_message = {"role": "assistant", "content": final_response_text}
    updated_history = history + [user_message, assistant_message]

    if DEBUG_TOOL_EXECUTION:
        logger.debug(f"🎉 Streaming complete: {len(final_response_text)} chars, {len(executed_tools)} tools")

    return final_response_text, thinking, updated_history


# Convenience functions
def enable_debug_logging():
    """Enable debug logging"""
    global DEBUG_TOOL_EXECUTION
    DEBUG_TOOL_EXECUTION = True
    logging.getLogger(__name__).setLevel(logging.DEBUG)


def disable_debug_logging():
    """Disable debug logging"""
    global DEBUG_TOOL_EXECUTION
    DEBUG_TOOL_EXECUTION = False


# Export main functions
__all__ = [
    "chat_with_tool_execution_batch",
    "chat_with_tool_execution_streaming",
    "GenerationConfig",
    "enable_debug_logging",
    "disable_debug_logging"
]