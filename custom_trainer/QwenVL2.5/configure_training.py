"""
Training configuration utility for Qwen 2.5 VL
Helps users choose optimal settings based on dataset size and hardware
"""

import argparse
import json
from typing import Any, Dict


def get_recommended_config(
    dataset_size: int, gpu_memory_gb: int, use_full_ft: bool = False
) -> Dict[str, Any]:
    """
    Get recommended training configuration based on dataset size and hardware

    Args:
        dataset_size: Number of training images
        gpu_memory_gb: Available GPU memory in GB
        use_full_ft: Force full fine-tuning regardless of other factors

    Returns:
        Dictionary with recommended configuration
    """

    config = {
        "training_approach": "",
        "model_config": {},
        "sft_config": {},
        "data_config": {},
        "hardware_recommendations": {},
        "expected_training_time": "",
    }

    # Determine training approach
    if use_full_ft or dataset_size > 100000:
        approach = "full_finetuning"
        config["model_config"]["use_lora"] = False
        config["sft_config"]["learning_rate"] = 5e-6
        min_memory = 80
    elif dataset_size < 10000:
        approach = "lora_small"
        config["model_config"]["use_lora"] = True
        config["model_config"]["lora_r"] = 64
        config["model_config"]["lora_alpha"] = 128
        config["sft_config"]["learning_rate"] = 2e-5
        min_memory = 16
    elif dataset_size < 50000:
        approach = "lora_medium"
        config["model_config"]["use_lora"] = True
        config["model_config"]["lora_r"] = 128
        config["model_config"]["lora_alpha"] = 256
        config["sft_config"]["learning_rate"] = 2e-5
        min_memory = 24
    else:  # 50k-100k
        approach = "lora_large"
        config["model_config"]["use_lora"] = True
        config["model_config"]["lora_r"] = 256
        config["model_config"]["lora_alpha"] = 512
        config["sft_config"]["learning_rate"] = 1e-5
        min_memory = 32

    config["training_approach"] = approach

    # Adjust batch size and gradient accumulation based on GPU memory
    if gpu_memory_gb >= min_memory * 2:
        batch_size = 8 if approach == "full_finetuning" else 16
        grad_accum = 2
    elif gpu_memory_gb >= min_memory:
        batch_size = 4 if approach == "full_finetuning" else 8
        grad_accum = 4
    else:
        batch_size = 2 if approach == "full_finetuning" else 4
        grad_accum = 8

    config["data_config"]["train_batch_size"] = batch_size
    config["sft_config"]["gradient_accumulation_steps"] = grad_accum

    # Common configurations
    config["sft_config"].update(
        {
            "num_train_epochs": 3,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "bf16": True,
            "save_steps": 500,
            "eval_steps": 500,
            "logging_steps": 10,
        }
    )

    config["data_config"].update(
        {
            "eval_batch_size": batch_size * 2,
            "max_length": 2048,
            "image_size": 448,
            "num_workers": 4,
        }
    )

    # Hardware recommendations
    config["hardware_recommendations"] = {
        "minimum_gpu_memory_gb": min_memory,
        "recommended_gpu_memory_gb": min_memory * 2,
        "estimated_training_time_hours": estimate_training_time(dataset_size, approach),
        "storage_requirements_gb": estimate_storage_requirements(dataset_size, approach),
    }

    return config


def estimate_training_time(dataset_size: int, approach: str) -> float:
    """Estimate training time in hours"""
    # Very rough estimates - actual time depends on hardware
    samples_per_hour = {
        "lora_small": 2000,
        "lora_medium": 1500,
        "lora_large": 1000,
        "full_finetuning": 500,
    }

    # 3 epochs default
    total_samples = dataset_size * 3
    hours = total_samples / samples_per_hour.get(approach, 1000)
    return round(hours, 1)


def estimate_storage_requirements(dataset_size: int, approach: str) -> int:
    """Estimate storage requirements in GB"""
    base_storage = 20  # Base model and checkpoints

    if approach == "full_finetuning":
        checkpoint_size = 15  # GB per checkpoint
        num_checkpoints = 5  # Keep last 5
        return base_storage + (checkpoint_size * num_checkpoints)
    else:
        checkpoint_size = 0.5  # GB per LoRA checkpoint
        num_checkpoints = 10  # Keep more LoRA checkpoints
        return base_storage + (checkpoint_size * num_checkpoints)


def generate_command_line(config: Dict[str, Any], output_dir: str = "outputs") -> str:
    """Generate command line for training"""

    cmd = ["python train_qwenvl_25.py"]
    cmd.append("--stage both")
    cmd.append("--train_data data/train.jsonl")
    cmd.append("--val_data data/val.jsonl")
    cmd.append("--image_base_path data/images")

    # Model settings
    if config["model_config"].get("use_lora", False):
        cmd.append("--use_lora")
        cmd.append(f"--lora_r {config['model_config']['lora_r']}")

    # Training settings
    cmd.append(f"--learning_rate {config['sft_config']['learning_rate']}")
    cmd.append(f"--num_epochs {config['sft_config']['num_train_epochs']}")
    cmd.append(f"--batch_size {config['data_config']['train_batch_size']}")
    cmd.append(
        f"--gradient_accumulation_steps {config['sft_config']['gradient_accumulation_steps']}"
    )

    # Output directories
    cmd.append(f"--sft_output_dir {output_dir}/sft")
    cmd.append(f"--grpo_output_dir {output_dir}/grpo")

    # Other settings
    if config["sft_config"].get("bf16", False):
        cmd.append("--bf16")

    return " \\\n    ".join(cmd)


def main():
    parser = argparse.ArgumentParser(description="Configure Qwen 2.5 VL training")
    parser.add_argument("--dataset_size", type=int, required=True, help="Number of training images")
    parser.add_argument("--gpu_memory", type=int, required=True, help="Available GPU memory in GB")
    parser.add_argument("--force_full_ft", action="store_true", help="Force full fine-tuning")
    parser.add_argument("--output_config", type=str, help="Save configuration to JSON file")
    parser.add_argument("--show_command", action="store_true", help="Show training command")

    args = parser.parse_args()

    # Get recommended configuration
    config = get_recommended_config(
        dataset_size=args.dataset_size,
        gpu_memory_gb=args.gpu_memory,
        use_full_ft=args.force_full_ft,
    )

    # Print recommendations
    print("🚀 Qwen 2.5 VL Training Configuration Recommendations")
    print("=" * 60)
    print(f"Dataset Size: {args.dataset_size:,} images")
    print(f"GPU Memory: {args.gpu_memory} GB")
    print(f"Recommended Approach: {config['training_approach'].replace('_', ' ').title()}")
    print()

    if config["model_config"].get("use_lora", False):
        print("LoRA Configuration:")
        print(f"  - Rank (r): {config['model_config']['lora_r']}")
        print(f"  - Alpha: {config['model_config']['lora_alpha']}")
        print(f"  - Learning Rate: {config['sft_config']['learning_rate']}")
    else:
        print("Full Fine-tuning Configuration:")
        print(f"  - Learning Rate: {config['sft_config']['learning_rate']}")

    print()
    print("Training Configuration:")
    print(f"  - Batch Size: {config['data_config']['train_batch_size']}")
    print(f"  - Gradient Accumulation: {config['sft_config']['gradient_accumulation_steps']}")
    print(
        f"  - Effective Batch Size: {config['data_config']['train_batch_size'] * config['sft_config']['gradient_accumulation_steps']}"
    )
    print()

    print("Hardware Requirements:")
    print(
        f"  - Minimum GPU Memory: {config['hardware_recommendations']['minimum_gpu_memory_gb']} GB"
    )
    print(
        f"  - Recommended GPU Memory: {config['hardware_recommendations']['recommended_gpu_memory_gb']} GB"
    )
    print(
        f"  - Estimated Training Time: {config['hardware_recommendations']['estimated_training_time_hours']} hours"
    )
    print(
        f"  - Storage Requirements: {config['hardware_recommendations']['storage_requirements_gb']} GB"
    )
    print()

    # Memory warning
    if args.gpu_memory < config["hardware_recommendations"]["minimum_gpu_memory_gb"]:
        print("⚠️  WARNING: Your GPU memory is below the minimum recommended.")
        print("   Consider using gradient checkpointing or reducing batch size.")
        print()

    # Show training command
    if args.show_command:
        print("Training Command:")
        print("-" * 60)
        print(generate_command_line(config))
        print()

    # Save configuration
    if args.output_config:
        with open(args.output_config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {args.output_config}")


if __name__ == "__main__":
    main()
