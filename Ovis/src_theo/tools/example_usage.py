"""
Example usage of tool execution for Ovis2.5

Demonstrates batch/streaming generation, trained tools, and LLM-driven tool selection.
"""

from PIL import Image
from src_theo.tools.inference_integration_v2 import (
    GenerationConfig,
    ToolRegistry,
    chat_with_tool_execution_batch,
    chat_with_tool_execution_streaming,
)


class DrawingTool:
    """Example of a new tool that can be instantly added"""

    def extract_tool_calls(self, text: str):
        """Extract drawing tool calls from text"""
        import re

        pattern = r"<tool_call>Draw \[([^\]]+)\]</tool_call>"
        matches = []
        for match in re.finditer(pattern, text):
            matches.append(
                {
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "full_match": match.group(0),
                    "parameters": {"action": match.group(1)},
                    "tool_name": "draw",
                    "tool_instance": self,
                }
            )
        return matches

    def execute(self, image: Image.Image, parameters: dict):
        """Execute drawing operation"""
        print(f"Drawing: {parameters.get('action', '')}")
        return image


def load_model():
    """Load your Ovis2.5 model - replace with your model loading code"""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "AIDC-AI/Ovis2.5-9B", trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )
    return model


def example_batch_generation():
    """Example of batch generation - recommended for production APIs"""

    # Load model
    model = load_model()

    # Load test image
    image = Image.open("test_image.jpg")

    # Configure tool execution
    config = GenerationConfig(tool_timeout=15.0, enable_caching=True)

    # Chat with tool execution
    response, thinking, history = chat_with_tool_execution_batch(
        model=model,
        prompt="Please examine this image in detail. Focus on any people you see.",
        images=[image],
        config=config,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=1024,
        enable_thinking=True,
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

    # Configure for streaming
    config = GenerationConfig(tool_timeout=10.0, enable_caching=True)

    # Chat with streaming tool execution
    response, thinking, history = chat_with_tool_execution_streaming(
        model=model,
        prompt="Analyze this image step by step. Look closely at specific regions.",
        images=[image],
        config=config,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=1024,
        enable_thinking=True,
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

    config = GenerationConfig(tool_timeout=15.0, enable_caching=True)

    # Start conversation
    history = []

    # First turn
    response1, thinking1, history = chat_with_tool_execution_batch(
        model=model,
        prompt="What do you see in this image?",
        images=[image],
        history=history,
        config=config,
        temperature=0.7,
    )

    print("=== TURN 1 ===")
    print(f"Response: {response1}")

    # Second turn - follow up question
    response2, thinking2, history = chat_with_tool_execution_batch(
        model=model,
        prompt="Can you examine the clothing of any people in more detail?",
        history=history,  # Pass previous history
        config=config,
        temperature=0.7,
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

    config = GenerationConfig(tool_timeout=15.0)
    prompt = "Examine this image in detail, looking at specific regions."

    # Benchmark batch
    start_time = time.time()
    batch_response, _, _ = chat_with_tool_execution_batch(
        model=model, prompt=prompt, images=[image], config=config, temperature=0.7
    )
    batch_time = time.time() - start_time

    # Benchmark streaming
    start_time = time.time()
    streaming_response, _, _ = chat_with_tool_execution_streaming(
        model=model, prompt=prompt, images=[image], config=config, temperature=0.7
    )
    streaming_time = time.time() - start_time

    print("=== PERFORMANCE COMPARISON ===")
    print(f"Batch time: {batch_time:.2f}s")
    print(f"Streaming time: {streaming_time:.2f}s")
    print(f"Batch faster by: {((streaming_time - batch_time) / batch_time * 100):.1f}%")
    print(f"Response lengths - Batch: {len(batch_response)}, Streaming: {len(streaming_response)}")


def example_trained_tools():
    """Example using trained crop tool (existing behavior)"""

    model = load_model()
    image = Image.open("test_image.jpg")

    config = GenerationConfig(tool_timeout=15.0, enable_caching=True)

    # Use trained tools without descriptions
    response, thinking, history = chat_with_tool_execution_batch(
        model=model,
        prompt="Examine this image in detail. Focus on any people you see.",
        images=[image],
        config=config,
        use_tool_descriptions=False,  # Model uses trained behavior
        temperature=0.7,
    )

    print("=== TRAINED TOOL USAGE ===")
    print(f"Response: {response}")

    return response


def example_llm_driven_tools():
    """Example using LLM-driven tool selection with descriptions"""

    model = load_model()
    image = Image.open("test_image.jpg")

    config = GenerationConfig(tool_timeout=15.0, enable_caching=True)

    # Add new tool instantly
    tool_registry = ToolRegistry()
    drawing_tool = DrawingTool()
    tool_registry.register_tool(
        "draw",
        drawing_tool,
        {
            "name": "draw",
            "description": "Draw shapes or annotations on the image",
            "parameters": "[action] - description of what to draw",
            "usage": "<tool_call>Draw [circle around face]</tool_call>",
            "example": "To highlight a person: <tool_call>Draw [red box around person]</tool_call>",
        },
    )

    # Use LLM-driven tool selection
    response, thinking, history = chat_with_tool_execution_batch(
        model=model,
        prompt="Analyze this image and highlight interesting regions",
        images=[image],
        config=config,
        use_tool_descriptions=True,  # Enable tool descriptions
        temperature=0.7,
    )

    print("=== LLM-DRIVEN TOOL USAGE ===")
    print(f"Response: {response}")

    return response


def example_mixed_usage():
    """Example using both trained and instant tools together"""

    model = load_model()
    image = Image.open("test_image.jpg")

    config = GenerationConfig(tool_timeout=15.0)

    # Use both trained crop and instant drawing tools
    response, thinking, history = chat_with_tool_execution_batch(
        model=model,
        prompt="Crop the most interesting part of this image, then suggest what annotations to add",
        images=[image],
        config=config,
        use_tool_descriptions=True,  # LLM can see both trained and new tools
        temperature=0.7,
    )

    print("=== MIXED TOOL USAGE ===")
    print(f"Response: {response}")

    return response


if __name__ == "__main__":
    print("Starting Ovis2.5 Tool Execution Examples...\n")

    try:
        # Core examples
        example_batch_generation()
        print("\n" + "=" * 50 + "\n")

        example_streaming_generation()
        print("\n" + "=" * 50 + "\n")

        example_multi_turn_conversation()
        print("\n" + "=" * 50 + "\n")

        # Tool system examples
        example_trained_tools()
        print("\n" + "=" * 50 + "\n")

        example_llm_driven_tools()
        print("\n" + "=" * 50 + "\n")

        example_mixed_usage()
        print("\n" + "=" * 50 + "\n")

        # Performance comparison
        benchmark_comparison()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Ovis2.5 model installed and accessible")
        print("2. test_image.jpg in the current directory")
