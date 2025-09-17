"""
Real-time tool execution during Ovis2.5 generation.
Implements streaming generation with tool call interruption and resume.
Scalable system supporting multiple tool types.
"""

import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import TextIteratorStreamer

# Set up logger for this module
logger = logging.getLogger(__name__)

# Global debug setting - can be controlled via environment variable
DEBUG_TOOL_EXECUTION = os.getenv("OVIS_TOOL_DEBUG", "false").lower() == "true"


def enable_debug_logging():
    """Enable debug logging for tool execution"""
    global DEBUG_TOOL_EXECUTION
    DEBUG_TOOL_EXECUTION = True
    logger.setLevel(logging.DEBUG)


def disable_debug_logging():
    """Disable debug logging for tool execution"""
    global DEBUG_TOOL_EXECUTION
    DEBUG_TOOL_EXECUTION = False


def _should_log_debug() -> bool:
    """Check if debug logging should be enabled"""
    return DEBUG_TOOL_EXECUTION or logger.isEnabledFor(logging.DEBUG)


class ToolAwareStreamer(TextIteratorStreamer):
    """
    Advanced streaming implementation that detects tool calls in real-time
    and coordinates with the generation loop for precise tool execution.

    Features:
    - Real-time tool call detection during streaming
    - Pause/resume mechanism for tool execution
    - Thread-safe communication with generation loop
    - Maintains all original streaming capabilities
    """

    def __init__(
        self,
        tokenizer,
        tool_registry,
        skip_prompt: bool = True,
        skip_special_tokens: bool = True,
        on_tool_detected: Optional[Callable] = None,
        **decode_kwargs,
    ):
        super().__init__(
            tokenizer,
            skip_prompt=skip_prompt,
            skip_special_tokens=skip_special_tokens,
            **decode_kwargs,
        )

        self.tool_registry = tool_registry
        self.on_tool_detected = on_tool_detected

        # Tool detection state
        self.accumulated_text = ""
        self.tool_pause_requested = False
        self.generation_should_pause = threading.Event()
        self.tool_execution_complete = threading.Event()
        self.tool_execution_complete.set()  # Initially not in tool execution

        # Thread-safe communication
        self._lock = threading.Lock()

        # Tool tracking
        self.detected_tools = []
        self.processed_tool_count = 0

        if _should_log_debug():
            logger.debug("🚀 ToolAwareStreamer initialized with real-time tool detection")

    def put(self, value):
        """
        Override put method to detect tool calls in real-time.
        This gets called for each token/batch of tokens during generation.
        """
        with self._lock:
            # Decode new tokens
            if isinstance(value, torch.Tensor):
                if len(value.shape) > 1:
                    value = value[0]  # Remove batch dimension if present
                new_text = self.tokenizer.decode(
                    value, skip_special_tokens=self.skip_special_tokens
                )
            else:
                new_text = str(value)

            # Accumulate text for tool detection
            self.accumulated_text += new_text

            # Check for tool completion
            if self._check_for_tool_completion():
                if _should_log_debug():
                    logger.debug("🔧 Tool completion detected in stream - requesting pause")

                # Signal main thread to pause generation
                self.tool_pause_requested = True
                self.generation_should_pause.set()

                # Call callback if provided
                if self.on_tool_detected:
                    self.on_tool_detected(self.accumulated_text)

        # Continue with normal streaming
        super().put(value)

    def _check_for_tool_completion(self) -> bool:
        """Check if a tool call has just been completed"""
        if not self.accumulated_text:
            return False

        # Detect all current tool calls
        current_tools = self.tool_registry.detect_tool_calls(self.accumulated_text)

        # Check if we have new completed tool calls
        if len(current_tools) > self.processed_tool_count:
            # Check if the latest tool call is actually complete
            latest_text = self.accumulated_text
            if "</tool_call>" in latest_text:
                # Verify we're not inside an incomplete tool call
                open_count = latest_text.count("<tool_call>")
                close_count = latest_text.count("</tool_call>")
                if close_count >= open_count and close_count > self.processed_tool_count:
                    self.detected_tools = current_tools
                    return True

        return False

    def wait_for_tool_execution(self, timeout: float = 30.0):
        """Wait for tool execution to complete before continuing generation"""
        if _should_log_debug():
            logger.debug("🔧 Streamer waiting for tool execution to complete")

        success = self.tool_execution_complete.wait(timeout=timeout)
        if not success:
            logger.warning(f"Tool execution timeout after {timeout}s")

        return success

    def notify_tool_execution_complete(self, processed_count: int):
        """Notify streamer that tool execution is complete and can resume"""
        with self._lock:
            self.processed_tool_count = processed_count
            self.tool_pause_requested = False
            self.generation_should_pause.clear()
            self.tool_execution_complete.set()

            if _should_log_debug():
                logger.debug(f"🔧 Tool execution complete - processed {processed_count} tools")

    def reset_for_new_generation(self):
        """Reset streamer state for a new generation cycle"""
        with self._lock:
            self.accumulated_text = ""
            self.tool_pause_requested = False
            self.generation_should_pause.clear()
            self.tool_execution_complete.set()
            self.detected_tools = []
            self.processed_tool_count = 0

            if _should_log_debug():
                logger.debug("🔄 ToolAwareStreamer reset for new generation")

    def should_pause_generation(self) -> bool:
        """Check if generation should pause for tool execution"""
        return self.tool_pause_requested

    def get_current_text(self) -> str:
        """Get current accumulated text (thread-safe)"""
        with self._lock:
            return self.accumulated_text

    def get_detected_tools(self) -> List[Dict]:
        """Get currently detected tools (thread-safe)"""
        with self._lock:
            return self.detected_tools.copy()


class InferenceToolRegistry:
    """Registry for handling multiple tool types during inference"""

    def __init__(self):
        self.tools = {}
        self._setup_available_tools()

    def _setup_available_tools(self):
        """Initialize available tools for inference"""
        # Try to load CropTool
        try:
            from src_theo.tools.crop_tool import CropTool

            self.register_tool("crop", CropTool())
        except ImportError:
            logger.warning("CropTool not available for inference.")

        # Future tools can be added here:
        # try:
        #     from drawing_tool import DrawingTool
        #     self.register_tool('draw', DrawingTool())
        # except ImportError:
        #     pass

    def register_tool(self, tool_name: str, tool_instance):
        """Register a tool instance for inference use"""
        self.tools[tool_name] = tool_instance

    def detect_tool_calls(self, text: str) -> List[Dict]:
        """Detect any tool calls in text and return details"""
        if not text or not isinstance(text, str):
            return []

        all_tool_calls = []
        for tool_name, tool_instance in self.tools.items():
            if hasattr(tool_instance, "extract_tool_calls"):
                tool_calls = tool_instance.extract_tool_calls(text)
                for tc in tool_calls:
                    tc["tool_name"] = tool_name
                    tc["tool_instance"] = tool_instance
                all_tool_calls.extend(tool_calls)

        # Sort by position to execute in order
        all_tool_calls.sort(key=lambda x: x.get("start_pos", 0))
        return all_tool_calls

    def execute_tool_call(self, tool_call: Dict, original_image: Image.Image):
        """Execute a single tool call and return result"""
        tool_instance = tool_call["tool_instance"]
        tool_name = tool_call["tool_name"]

        if tool_name == "crop":
            # Execute crop operation
            return tool_instance.crop_image(original_image, tool_call["coordinates"])

        # Future tool executions can be added here:
        # elif tool_name == 'draw':
        #     return tool_instance.draw_on_image(original_image, tool_call['parameters'])

        else:
            raise ValueError(f"Unknown tool execution for: {tool_name}")

    def create_multimodal_context(self, text: str, image: Image.Image, executed_tools: List[Dict]):
        """Create multimodal context with executed tool results"""
        # For now, prioritize crop tool for multimodal content creation
        if "crop" in self.tools:
            crop_tool = self.tools["crop"]
            if hasattr(crop_tool, "create_multimodal_content"):
                return crop_tool.create_multimodal_content(text, image)

        # Fallback to text-only content
        return [{"type": "text", "text": text}]


def chat_with_tool_execution(
    model,
    prompt: str,
    history: Optional[List[Dict]] = None,
    images: Optional[List[Image.Image]] = None,
    videos: Optional[List[List[Image.Image]]] = None,
    do_sample: bool = True,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    enable_thinking: bool = False,
    enable_thinking_budget: bool = False,
    thinking_budget: int = 2048,
    min_pixels: int = 448 * 448,
    max_pixels: int = 1792 * 1792,
    **kwargs,
) -> Tuple[str, Optional[str], List[Dict]]:
    """
    Chat with tool execution during generation.

    When model generates <tool_call>ToolName [...]</tool_call>, immediately:
    1. Pause generation
    2. Execute tool operation
    3. Add result to context (e.g., cropped image)
    4. Resume generation with tool_call results

    Mimics training behavior where tool results appear right after tool calls.
    Supports multiple tool types through InferenceToolRegistry.

    Args:
        model: Ovis2.5 model instance
        prompt: Text prompt for conversation
        history: Previous conversation history (optional)
        images: List of input images (optional)
        videos: List of video frame sequences (optional)
        do_sample: Whether to use sampling for generation (default: True)
        max_new_tokens: Maximum tokens to generate (default: 1024)
        temperature: Sampling temperature (default: 0.6)
        top_p: Top-p sampling parameter (default: 0.9)
        top_k: Top-k sampling parameter (default: 50)
        enable_thinking: Enable thinking mode (default: False)
        enable_thinking_budget: Enable thinking budget (default: False)
        thinking_budget: Token budget for thinking (default: 2048)
        min_pixels: Minimum pixels for image preprocessing (default: 448*448)
        max_pixels: Maximum pixels for image preprocessing (default: 1792*1792)
        **kwargs: Additional generation parameters

    Returns:
        Tuple of (response_text, thinking, updated_history)

    Note:
        Debug logging can be enabled globally using enable_debug_logging()
        or by setting OVIS_TOOL_DEBUG=true environment variable.
    """
    tool_registry = InferenceToolRegistry()

    # Initialize history
    if history is None:
        history = []

    # Prepare content for current user message
    content = []
    current_images = []

    # Add images if provided
    if images:
        for image in images:
            content.append({"type": "image", "image": image})
            current_images.append(image)

    # Add videos if provided
    if videos:
        for video_frames in videos:
            content.append({"type": "video", "video": video_frames})

    # Add text prompt
    content.append({"type": "text", "text": prompt})

    # Create current user message
    user_message = {"role": "user", "content": content if len(content) > 1 else prompt}

    # Build full conversation
    full_messages = history + [user_message]

    # Generate with tool interruption
    response_text = ""
    thinking = None
    executed_tools = []
    processed_tool_calls = 0  # Track already processed tool calls

    # Preprocess inputs
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        full_messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        **kwargs,
    )

    # Move to device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    if grid_thws is not None:
        grid_thws = grid_thws.to(device)

    # Generate with streaming and tool interruption
    generated_tokens = []
    inside_tool_call = False  # Track if we're currently inside a <tool_call>

    def _is_inside_tool_call(text: str) -> bool:
        """Check if we're currently inside an incomplete tool call"""
        if not text:
            return False

        # Count opening and closing tool call tags
        open_count = text.count("<tool_call>")
        close_count = text.count("</tool_call>")

        # If we have more opening tags than closing tags, we're inside a tool call
        return open_count > close_count

    def _get_adaptive_batch_size(
        current_text: str, remaining_tokens: int, prev_inside_tool_call: bool
    ) -> int:
        """Get batch size based on current context - slower inside tool calls"""
        currently_inside = _is_inside_tool_call(current_text)

        # Log when we transition states (for debugging)
        if currently_inside != prev_inside_tool_call:
            if currently_inside and _should_log_debug():
                logger.debug("🔧 Entering tool call - switching to token-by-token generation")
            elif not currently_inside and _should_log_debug():
                logger.debug("🔧 Exiting tool call - resuming batch generation")

        if currently_inside:
            # Token-by-token generation inside tool calls for precise control
            return 1
        else:
            # Normal batch size outside tool calls
            return min(6, remaining_tokens)

    while len(generated_tokens) < max_new_tokens:
        # Generate next batch of tokens (adaptive batch size)
        with torch.no_grad():
            prev_inside_state = inside_tool_call
            batch_size = _get_adaptive_batch_size(
                response_text, max_new_tokens - len(generated_tokens), prev_inside_state
            )

            generation_kwargs = {
                "inputs": input_ids,
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
                "max_new_tokens": batch_size,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "enable_thinking": enable_thinking,
                "enable_thinking_budget": enable_thinking_budget,
                "thinking_budget": thinking_budget,
                "eos_token_id": model.text_tokenizer.eos_token_id,
                "pad_token_id": model.text_tokenizer.pad_token_id,
            }

            outputs = model.generate(**generation_kwargs)
            # Get new tokens
            new_tokens = outputs[0][input_ids.size(1) :]  # Only new tokens
            generated_tokens.extend(new_tokens.tolist())

            # Decode to check for tool calls
            partial_response = model.text_tokenizer.decode(new_tokens, skip_special_tokens=True)
            response_text += partial_response

            # Update tool call state
            inside_tool_call = _is_inside_tool_call(response_text)

            tool_executed = False
            # Check for complete tool calls - must be a newly completed tool call
            tool_just_completed = "</tool_call>" in partial_response and not _is_inside_tool_call(
                response_text
            )

            if tool_just_completed:
                tool_calls = tool_registry.detect_tool_calls(response_text)

                # Execute any new tool calls we haven't processed yet
                if tool_calls and len(tool_calls) > processed_tool_calls and current_images:
                    new_tool_calls = tool_calls[processed_tool_calls:]

                    # Execute all new tool calls
                    for tool_call in new_tool_calls:
                        tool_name = tool_call["tool_name"]
                        if _should_log_debug():
                            logger.debug(f"🔧 Detected {tool_name} tool call during generation")

                        # Execute the tool
                        original_image = current_images[0]
                        try:
                            tool_result = tool_registry.execute_tool_call(tool_call, original_image)

                            if tool_name == "crop":
                                # For crop tools, add the cropped image to context
                                current_images.append(tool_result)
                                tool_executed = True
                            else:
                                # For other tools, handle differently based on tool type
                                # This is where future tools can be handled
                                tool_executed = True

                            executed_tools.append(tool_call)
                            processed_tool_calls += 1
                            if _should_log_debug():
                                logger.debug(f"🔧 Executed {tool_name} tool")

                        except Exception as e:
                            logger.warning(f"{tool_name} tool execution failed: {e}")

                    # Rebuild context with all tool results if any tools were executed
                    if tool_executed:
                        # Rebuild multimodal context with tool results
                        multimodal_context = tool_registry.create_multimodal_context(
                            response_text, current_images[0], executed_tools
                        )

                        # Create assistant partial with multimodal content
                        assistant_partial = {"role": "assistant", "content": multimodal_context}

                        # Rebuild messages with partial assistant response
                        updated_messages = full_messages + [assistant_partial]

                        # Reprocess inputs with tool results in context
                        input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                            updated_messages,
                            add_generation_prompt=False,  # Don't add prompt since we're continuing
                            enable_thinking=enable_thinking,
                            min_pixels=min_pixels,
                            max_pixels=max_pixels,
                            **kwargs,
                        )

                        # Move to device
                        input_ids = input_ids.to(device)
                        if pixel_values is not None:
                            pixel_values = pixel_values.to(device)
                        if grid_thws is not None:
                            grid_thws = grid_thws.to(device)

                        if _should_log_debug():
                            logger.debug(
                                f"🔧 Resumed generation with {len(new_tool_calls)} tool result(s) in context"
                            )

            # Check for EOS
            if model.text_tokenizer.eos_token_id in new_tokens:
                break

            # Update input_ids for next iteration (only if we didn't execute a tool)
            if not tool_executed:
                input_ids = outputs[0].unsqueeze(0)  # Full sequence for next generation

    # Parse thinking and response (similar to original chat function)
    if enable_thinking and "<think>" in response_text and "</think>" in response_text:
        thinking_start = response_text.find("<think>") + 7
        thinking_end = response_text.find("</think>")
        thinking = response_text[thinking_start:thinking_end].strip()
        response_text = response_text[thinking_end + 8 :].strip()

    # Clean up response
    response_text = response_text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

    # Create assistant message for history
    assistant_message = {"role": "assistant", "content": response_text}
    updated_history = history + [user_message, assistant_message]

    return response_text, thinking, updated_history


def chat_with_tool_execution_streaming(
    model,
    prompt: str,
    history: Optional[List[Dict]] = None,
    images: Optional[List[Image.Image]] = None,
    videos: Optional[List[List[Image.Image]]] = None,
    do_sample: bool = True,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    enable_thinking: bool = False,
    enable_thinking_budget: bool = False,
    thinking_budget: int = 2048,
    min_pixels: int = 448 * 448,
    max_pixels: int = 1792 * 1792,
    **kwargs,
) -> Tuple[str, Optional[str], List[Dict]]:
    """
    Advanced streaming chat with real-time tool execution during generation.

    This function provides the ultimate user experience by combining:
    - Real-time token streaming (immediate response feedback)
    - Precise tool call detection and execution
    - Context rebuilding with tool results
    - Seamless generation resume

    Advantages over batch-based approach:
    - Better perceived throughput (streaming)
    - Smoother user experience
    - Professional-grade real-time interaction
    - Thread-safe tool coordination

    Args:
        model: Ovis2.5 model instance
        prompt: Text prompt for conversation
        history: Previous conversation history (optional)
        images: List of input images (optional)
        videos: List of video frame sequences (optional)
        do_sample: Whether to use sampling for generation (default: True)
        max_new_tokens: Maximum tokens to generate (default: 1024)
        temperature: Sampling temperature (default: 0.6)
        top_p: Top-p sampling parameter (default: 0.9)
        top_k: Top-k sampling parameter (default: 50)
        enable_thinking: Enable thinking mode (default: False)
        enable_thinking_budget: Enable thinking budget (default: False)
        thinking_budget: Token budget for thinking (default: 2048)
        min_pixels: Minimum pixels for image preprocessing (default: 448*448)
        max_pixels: Maximum pixels for image preprocessing (default: 1792*1792)
        **kwargs: Additional generation parameters

    Returns:
        Tuple of (response_text, thinking, updated_history)

    Note:
        This is the premium streaming implementation that provides real-time
        feedback while maintaining precise tool execution control.
    """
    tool_registry = InferenceToolRegistry()

    # Initialize history
    if history is None:
        history = []

    # Prepare content for current user message
    content = []
    current_images = []

    # Add images if provided
    if images:
        for image in images:
            content.append({"type": "image", "image": image})
            current_images.append(image)

    # Add videos if provided
    if videos:
        for video_frames in videos:
            content.append({"type": "video", "video": video_frames})

    # Add text prompt
    content.append({"type": "text", "text": prompt})

    # Create current user message
    user_message = {"role": "user", "content": content if len(content) > 1 else prompt}

    # Build full conversation
    full_messages = history + [user_message]

    # Initialize tool execution state
    executed_tools = []
    final_response_text = ""
    thinking = None

    # Setup tool-aware streamer with coordination callback
    def on_tool_detected_callback(current_text: str):
        """Callback when tool is detected - this runs in streamer thread"""
        if _should_log_debug():
            logger.debug(f"🔧 Tool callback triggered with {len(current_text)} chars")

    # Create advanced streaming infrastructure
    streamer = ToolAwareStreamer(
        tokenizer=model.text_tokenizer,
        tool_registry=tool_registry,
        skip_prompt=True,
        skip_special_tokens=True,
        on_tool_detected=on_tool_detected_callback,
    )

    # Generation loop with streaming and tool coordination
    current_conversation = full_messages.copy()

    while (
        len(executed_tools) == 0 or streamer.should_pause_generation()
    ):  # Continue until no more tools
        # Reset streamer for new generation cycle
        streamer.reset_for_new_generation()

        if _should_log_debug():
            logger.debug(f"🚀 Starting streaming generation cycle {len(executed_tools) + 1}")

        # Preprocess inputs for current conversation state
        input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            current_conversation,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            **kwargs,
        )

        # Move to device
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        if grid_thws is not None:
            grid_thws = grid_thws.to(device)

        # Start streaming generation with tool awareness
        generation_kwargs = {
            "inputs": input_ids,
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "max_new_tokens": max_new_tokens
            - len(final_response_text.split()),  # Adjust for previous generations
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "enable_thinking": enable_thinking,
            "enable_thinking_budget": enable_thinking_budget,
            "thinking_budget": thinking_budget,
            "eos_token_id": model.text_tokenizer.eos_token_id,
            "pad_token_id": model.text_tokenizer.pad_token_id,
            "streamer": streamer,  # The magic happens here!
        }

        if _should_log_debug():
            logger.debug("🎯 Starting streaming generation with tool detection...")

        # Generate in separate thread to allow tool interruption
        generation_thread = threading.Thread(
            target=lambda: model.generate(**generation_kwargs), daemon=True
        )
        generation_thread.start()

        # Monitor for tool execution needs
        tool_executed = False
        while generation_thread.is_alive():
            # Check if streamer detected a tool
            if streamer.should_pause_generation():
                if _should_log_debug():
                    logger.debug("🔧 Tool detected - coordinating execution...")

                # Get current state from streamer
                detected_tools = streamer.get_detected_tools()

                # Wait for generation thread to pause/complete current batch
                generation_thread.join(timeout=5.0)
                if generation_thread.is_alive():
                    logger.warning("Generation thread did not pause within timeout")

                # Execute new tools
                if detected_tools and len(detected_tools) > len(executed_tools) and current_images:
                    new_tools = detected_tools[len(executed_tools) :]

                    for tool_call in new_tools:
                        tool_name = tool_call["tool_name"]
                        if _should_log_debug():
                            logger.debug(f"🔧 Executing {tool_name} tool from stream")

                        try:
                            # Execute tool
                            original_image = current_images[0]
                            tool_result = tool_registry.execute_tool_call(tool_call, original_image)

                            if tool_name == "crop":
                                current_images.append(tool_result)

                            executed_tools.append(tool_call)
                            tool_executed = True

                            if _should_log_debug():
                                logger.debug(f"🔧 Tool {tool_name} executed successfully")

                        except Exception as e:
                            logger.error(f"Tool execution failed: {e}")

                # Rebuild conversation with tool results
                if tool_executed:
                    # Get partial response text accumulated so far
                    partial_response = streamer.get_current_text()
                    final_response_text += partial_response

                    # Create multimodal context with tool results
                    multimodal_context = tool_registry.create_multimodal_context(
                        final_response_text, current_images[0], executed_tools
                    )

                    # Update conversation with partial response containing tool results
                    assistant_partial = {"role": "assistant", "content": multimodal_context}
                    current_conversation = full_messages + [assistant_partial]

                    # Notify streamer that tool execution is complete
                    streamer.notify_tool_execution_complete(len(executed_tools))

                    if _should_log_debug():
                        logger.debug(f"🔧 Context rebuilt with {len(executed_tools)} tool results")

                    # Break to start new generation cycle with updated context
                    break
                else:
                    # No new tools, continue generation
                    streamer.notify_tool_execution_complete(len(executed_tools))

            # Small sleep to prevent busy waiting
            time.sleep(0.01)

        # Wait for generation to complete if no tools were executed
        if not tool_executed:
            generation_thread.join()
            final_response_text = streamer.get_current_text()
            break

    # Parse thinking and response (similar to original chat function)
    if enable_thinking and "<think>" in final_response_text and "</think>" in final_response_text:
        thinking_start = final_response_text.find("<think>") + 7
        thinking_end = final_response_text.find("</think>")
        thinking = final_response_text[thinking_start:thinking_end].strip()
        final_response_text = final_response_text[thinking_end + 8 :].strip()

    # Clean up response
    final_response_text = (
        final_response_text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    )

    # Create assistant message for history
    assistant_message = {"role": "assistant", "content": final_response_text}
    updated_history = history + [user_message, assistant_message]

    if _should_log_debug():
        logger.debug(
            f"🎉 Streaming generation complete: {len(final_response_text)} chars, {len(executed_tools)} tools"
        )

    return final_response_text, thinking, updated_history
