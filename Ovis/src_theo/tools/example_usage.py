"""
Example usage of professional tool execution for Ovis2.5

This demonstrates both batch and streaming approaches with proper configuration.
"""

import asyncio
from PIL import Image
from inference_integration_v2 import (
    chat_with_tool_execution_batch,
    chat_with_tool_execution_streaming,
    ToolExecutionConfig,
    enable_debug_logging
)

def load_model():
    """Load your Ovis2.5 model - replace with your model loading code"""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "AIDC-AI/Ovis2.5-9B",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    return model

def example_batch_generation():
    """Example of batch generation - recommended for production APIs"""

    # Load model
    model = load_model()

    # Load test image
    image = Image.open("test_image.jpg")

    # Configure tool execution
    config = ToolExecutionConfig(
        tool_timeout=15.0,
        max_concurrent_tools=2,
        enable_caching=True,
        adaptive_batching=True
    )

    # Enable debug logging to see tool execution details
    enable_debug_logging()

    # Chat with tool execution
    response, thinking, history = chat_with_tool_execution_batch(
        model=model,
        prompt="Please examine this image in detail. Focus on any people you see.",
        images=[image],
        config=config,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=1024,
        enable_thinking=True
    )

    print("=== BATCH GENERATION RESULT ===")
    print(f"Response: {response}")
    if thinking:
        print(f"Thinking: {thinking}")
    print(f"History length: {len(history)}")

    return response, thinking, history

def example_streaming_generation():
    """Example of streaming generation - recommended for interactive UIs"""

    # Load model
    model = load_model()

    # Load test image
    image = Image.open("test_image.jpg")

    # Configure for streaming (slightly different settings)
    config = ToolExecutionConfig(
        tool_timeout=10.0,
        max_concurrent_tools=1,  # Lower for streaming stability
        enable_caching=True,
        adaptive_batching=True,
        generation_batch_size=4  # Smaller batches for streaming
    )

    # Chat with streaming tool execution
    response, thinking, history = chat_with_tool_execution_streaming(
        model=model,
        prompt="Analyze this image step by step. Look closely at specific regions.",
        images=[image],
        config=config,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=1024,
        enable_thinking=True
    )

    print("=== STREAMING GENERATION RESULT ===")
    print(f"Response: {response}")
    if thinking:
        print(f"Thinking: {thinking}")
    print(f"History length: {len(history)}")

    return response, thinking, history

def example_multi_turn_conversation():
    """Example of multi-turn conversation with tool execution"""

    model = load_model()
    image = Image.open("test_image.jpg")

    config = ToolExecutionConfig(
        tool_timeout=15.0,
        enable_caching=True  # Important for multi-turn
    )

    # Start conversation
    history = []

    # First turn
    response1, thinking1, history = chat_with_tool_execution_batch(
        model=model,
        prompt="What do you see in this image?",
        images=[image],
        history=history,
        config=config,
        temperature=0.7
    )

    print("=== TURN 1 ===")
    print(f"Response: {response1}")

    # Second turn - follow up question
    response2, thinking2, history = chat_with_tool_execution_batch(
        model=model,
        prompt="Can you examine the clothing of any people in more detail?",
        history=history,  # Pass previous history
        config=config,
        temperature=0.7
    )

    print("=== TURN 2 ===")
    print(f"Response: {response2}")
    print(f"Final history length: {len(history)}")

    return history

def benchmark_comparison():
    """Compare batch vs streaming performance"""
    import time

    model = load_model()
    image = Image.open("test_image.jpg")

    config = ToolExecutionConfig(tool_timeout=15.0)
    prompt = "Examine this image in detail, looking at specific regions."

    # Benchmark batch
    start_time = time.time()
    batch_response, _, _ = chat_with_tool_execution_batch(
        model=model,
        prompt=prompt,
        images=[image],
        config=config,
        temperature=0.7
    )
    batch_time = time.time() - start_time

    # Benchmark streaming
    start_time = time.time()
    streaming_response, _, _ = chat_with_tool_execution_streaming(
        model=model,
        prompt=prompt,
        images=[image],
        config=config,
        temperature=0.7
    )
    streaming_time = time.time() - start_time

    print("=== PERFORMANCE COMPARISON ===")
    print(f"Batch time: {batch_time:.2f}s")
    print(f"Streaming time: {streaming_time:.2f}s")
    print(f"Batch faster by: {((streaming_time - batch_time) / batch_time * 100):.1f}%")
    print(f"Response lengths - Batch: {len(batch_response)}, Streaming: {len(streaming_response)}")

if __name__ == "__main__":
    # Run examples
    print("Starting Ovis2.5 Tool Execution Examples...\n")

    try:
        # Example 1: Batch generation (recommended for production)
        example_batch_generation()

        print("\n" + "="*50 + "\n")

        # Example 2: Streaming generation (recommended for UI)
        example_streaming_generation()

        print("\n" + "="*50 + "\n")

        # Example 3: Multi-turn conversation
        example_multi_turn_conversation()

        print("\n" + "="*50 + "\n")

        # Example 4: Performance comparison
        benchmark_comparison()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Ovis2.5 model installed and accessible")
        print("2. test_image.jpg in the current directory")
        print("3. Required dependencies installed")