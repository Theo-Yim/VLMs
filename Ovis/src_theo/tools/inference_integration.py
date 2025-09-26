"""
Real-time tool execution during Ovis2.5 generation.
Implements streaming generation with tool call interruption and resume.
Scalable system supporting multiple tool types.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import TextIteratorStreamer

from .tool_base import ToolRegistry

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
    tool_registry = ToolRegistry()
    system_prompt_tools = tool_registry.get_system_prompt_tools()
    system_prompt = [{"role": "system", "content": system_prompt_tools}]

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
    full_messages = system_prompt + history + [user_message]

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

    generation_kwargs = {
        "inputs": input_ids,
        "pixel_values": pixel_values,
        "grid_thws": grid_thws,
        "max_new_tokens": max_new_tokens,
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

    # Generate with streaming and tool interruption
    generated_tokens = []

    def _is_inside_tool_call(text: str) -> bool:
        """Check if we're currently inside an incomplete tool call"""
        if not text:
            return False

        # Count opening and closing tool call tags
        open_count = text.count("<tool_call>")
        close_count = text.count("</tool_call>")

        # If we have more opening tags than closing tags, we're inside a tool call
        return open_count > close_count

    def _get_adaptive_batch_size(current_text: str, remaining_tokens: int) -> int:
        """Get batch size based on current context - slower inside tool calls"""
        currently_inside = _is_inside_tool_call(current_text)
        if currently_inside:
            # Token-by-token generation inside tool calls for precise control
            return 1
        else:
            # Normal batch size outside tool calls; should not be larger than 6, 6 is sum of tokens of "<tool_call>ToolName [ ]</tool_call>"
            return min(6, remaining_tokens)

    while len(generated_tokens) < max_new_tokens:
        # Generate next batch of tokens (adaptive batch size)
        with torch.no_grad():
            batch_size = _get_adaptive_batch_size(
                response_text, max_new_tokens - len(generated_tokens)
            )

            generation_kwargs["max_new_tokens"] = batch_size

            outputs = model.generate(**generation_kwargs)
            # Ovis generate() returns only NEW tokens, not full sequence
            generated_tokens.extend(outputs[0].tolist())

            # Decode response
            partial_response = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text += partial_response

            # Check for complete tool calls
            tool_just_completed = "</tool_call>" in partial_response

            if tool_just_completed:
                # Find the position of the latest <tool_call> in response_text
                latest_tool_start = response_text.rfind("<tool_call>")
                assert latest_tool_start != -1, "Latest tool start not found"

                # Extract just this single tool call using optimized detection
                tool_call_text = response_text[latest_tool_start:]
                tool_call = tool_registry.detect_tool_call(tool_call_text)
                assert tool_call, "Tool call just completed but none detected"
                # Adjust position to match full text
                tool_call["start_pos"] += latest_tool_start
                tool_call["end_pos"] += latest_tool_start

                tool_name = tool_call["tool_name"]
                if _should_log_debug():
                    logger.debug(f"🔧 Detected {tool_name} tool call during generation")

                # Execute the tool. Handle both image and text-only calls.
                first_image = current_images[0] if current_images else None
                tool_result = tool_registry.execute_tool_call(tool_call, first_image)

                # Handle tool result - can be dict or None
                if tool_result is not None:
                    # Handle structured tool results generically
                    if tool_result["type"] == "image":
                        if current_images:  # Only append if we have image context
                            current_images.append(tool_result["content"])
                    elif tool_result["type"] == "text":
                        response_text += tool_result["content"]
                    elif tool_result["type"] == "multimodal":
                        # Handle mixed content
                        for item in tool_result["content"]:
                            if item["type"] == "text":
                                response_text += item["text"]
                            elif item["type"] == "image" and current_images:
                                current_images.append(item["image"])
                # If tool_result is None, nothing to append after tool call

                executed_tools.append(tool_call)
                processed_tool_calls += 1
                if _should_log_debug():
                    logger.debug(f"🔧 Executed {tool_name} tool")

                # Rebuild context after processing the tool call
                # Create assistant message with current response and images (if any)
                content_parts = []

                # Add text content
                if response_text.strip():
                    content_parts.append({"type": "text", "text": response_text})

                # Add all images (original + tool results) - only if we have images
                if current_images:
                    for img in current_images:
                        content_parts.append({"type": "image", "image": img})

                # Create assistant partial - handle both multimodal and text-only cases
                if len(content_parts) > 1:
                    # Multimodal content (text + images)
                    assistant_partial = {"role": "assistant", "content": content_parts}
                else:
                    # Text-only content
                    assistant_partial = {"role": "assistant", "content": response_text}

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
                    logger.debug("🔧 Resumed generation with 1 tool result in context")
            else:
                # Update input_ids for next iteration (only when no tool was executed)
                input_ids = torch.cat([input_ids, outputs], dim=-1)  # Append new tokens
                generation_kwargs["inputs"] = input_ids  # Update for next generation

            # Check for EOS
            if model.text_tokenizer.eos_token_id in outputs[0]:
                break

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


# This function is not finished yet.
class ToolAwareStreamer(TextIteratorStreamer):
    """
    Advanced streaming implementation that detects tool calls in real-time
    """


# This function is not finished yet.
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
    """

    return
