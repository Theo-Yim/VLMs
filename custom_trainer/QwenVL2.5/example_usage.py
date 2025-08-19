"""
Example usage script for Qwen 2.5 VL training and inference
Demonstrates data preparation, training, and inference with crop tool
"""

import json
from pathlib import Path

from PIL import Image


# Example: Prepare training data
def prepare_sample_data():
    """Create a sample training data file"""

    sample_data = {
        "image_path": "coco/train2017/000000580837.jpg",
        "image_id": "000000580837",
        "QnA": [
            {
                "Q": "What is the visible clothing of the person on the far left?",
                "A3": '<think>\nTo determine the visible clothing of the person on the far left, I need to focus on that area of the image. Using the "Crop" tool, I can zoom in to get a clearer view of the person\'s attire.\n\n{Crop person 1 [0.00, 141.43, 79.23, 480.00]}\n\nUpon inspecting the cropped section, it appears that the person on the far left is wearing a dark jacket. The visible portion shows the shoulder and part of the arm, confirming that the clothing is predominantly dark.\n</think>\n\n<answer>The visible clothing of the person on the far left is a dark jacket.</answer>',
            },
            {
                "Q": "Describe the main objects in the scene.",
                "A3": "<think>\nLet me analyze the main objects in this scene. I'll examine different regions to provide a comprehensive description.\n\n{Crop center area [200.0, 100.0, 400.0, 300.0]}\n\nThe central area shows several key objects. Let me also check the right side.\n\n{Crop right side [400.0, 0.0, 640.0, 480.0]}\n\nBased on these observations, I can describe the main elements.\n</think>\n\n<answer>The scene contains multiple objects including people, furniture, and various items arranged in an indoor setting.</answer>",
            },
        ],
    }

    # Save to JSONL format
    output_path = Path("data/sample_train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(json.dumps(sample_data) + "\n")

    print(f"Sample data saved to {output_path}")
    return output_path


# Example: Train the model
def train_example():
    """Example training command"""

    print("Example training command for Stage 1 (SFT):")
    print("""
python train_qwenvl_25.py \\
    --stage sft \\
    --train_data data/sample_train.jsonl \\
    --val_data data/sample_val.jsonl \\
    --image_base_path /path/to/images \\
    --use_lora \\
    --lora_r 64 \\
    --learning_rate 2e-5 \\
    --num_epochs 3 \\
    --batch_size 4 \\
    --bf16
    """)

    print("\nExample training command for Stage 2 (R-GRPO):")
    print("""
python train_qwenvl_25.py \\
    --stage grpo \\
    --train_data data/sample_train.jsonl \\
    --image_base_path /path/to/images \\
    --sft_checkpoint outputs/sft/final_model \\
    --learning_rate 5e-6 \\
    --beta 0.1 \\
    --bf16
    """)


# Example: Inference with crop tool
def inference_example():
    """Example inference with crop tool execution"""

    from config import InferenceConfig
    from inference import QwenVLInference

    # Initialize inference
    config = InferenceConfig(
        model_path="outputs/sft/final_model",  # or grpo/final_model
        max_new_tokens=512,
        temperature=0.7,
        parse_tool_calls=True,
    )

    inference = QwenVLInference(config)

    # Example 1: Basic inference
    image = Image.open("test_image.jpg")
    question = "What objects can you see in this image?"

    result = inference.generate(image, question)

    print("Generated Response:")
    print("-" * 50)
    if result.think_content:
        print(f"Thinking: {result.think_content}")
    if result.answer_content:
        print(f"Answer: {result.answer_content}")
    if result.tool_calls:
        print(f"Tool Calls: {len(result.tool_calls)} crops executed")
        for i, tc in enumerate(result.tool_calls):
            print(f"  Crop {i + 1}: {tc['bbox']}")

    # Example 2: Inference with automatic tool execution
    result_with_tools = inference.generate_with_tool_execution(image, question, max_iterations=3)

    print("\nWith Tool Execution:")
    print("-" * 50)
    if hasattr(result_with_tools, "cropped_regions"):
        print(f"Executed {len(result_with_tools.cropped_regions)} crops")


# Example: Process existing annotations
def process_coco_style_data():
    """Example of converting COCO-style annotations to our format"""

    # This is just the structure - actual implementation would read from files
    coco_data = {
        "image_path": "path/to/image.jpg",
        "annotations": [{"bbox": [100, 100, 50, 50], "category_id": 1}],
    }

    # Convert to our training format
    training_data = {
        "image_path": coco_data["image_path"],
        "image_id": Path(coco_data["image_path"]).stem,
        "QnA": [
            {
                "Q": "Describe the object in the image.",
                "A3": "<think>Let me examine the object. {Crop object [100, 100, 150, 150]}</think><answer>The object is located at the specified region.</answer>",
            }
        ],
    }

    return training_data


def show_configuration_example():
    """Example of using the configuration tool"""
    print("\nOptimal Configuration Tool:")
    print("""
# Get recommended configuration for your dataset
python configure_training.py \\
    --dataset_size 40000 \\
    --gpu_memory 32 \\
    --show_command

# For larger datasets, consider full fine-tuning
python configure_training.py \\
    --dataset_size 100000 \\
    --gpu_memory 80 \\
    --force_full_ft \\
    --show_command \\
    --output_config config_100k.json
    """)


if __name__ == "__main__":
    print("Qwen 2.5 VL Training and Inference Examples")
    print("=" * 60)

    # Create sample data
    print("\n1. Creating sample training data...")
    data_path = prepare_sample_data()

    # Show configuration examples
    print("\n2. Configuration tool examples:")
    show_configuration_example()

    # Show training examples
    print("\n3. Manual training examples:")
    train_example()

    # Show inference example (commented out as it requires trained model)
    print("\n4. Inference example (requires trained model):")
    print("Uncomment inference_example() to run after training")
    # inference_example()

    print("\n" + "=" * 60)
    print("Setup complete! You can now:")
    print("1. Prepare your data in JSONL format")
    print("2. Use configure_training.py to get optimal settings")
    print("3. Run training with the recommended configuration")
    print("4. Use the trained model for inference with crop tool support")
