"""
Main training script for Ovis2.5-9B
Coordinates two-stage training: SFT and R-GRPO
Based on official guide: https://huggingface.co/AIDC-AI/Ovis2.5-9B
"""

import argparse
import json
import logging

from config import DataConfig, GRPOTrainingConfig, ModelConfig, SFTTrainingConfig
from train_grpo import OvisGRPOTrainer
from train_sft import OvisSFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config_from_json(config_path: str, config_class):
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return config_class(**config_dict)


def main():
    parser = argparse.ArgumentParser(description="Train Ovis2.5-9B with two-stage training")

    # Training stage selection
    parser.add_argument(
        "--stage",
        type=str,
        choices=["sft", "grpo", "both"],
        default="both",
        help="Training stage to run: sft (Stage 1), grpo (Stage 2), or both",
    )

    # Configuration files
    parser.add_argument("--model_config", type=str, help="Path to model configuration JSON file")
    parser.add_argument("--data_config", type=str, help="Path to data configuration JSON file")
    parser.add_argument(
        "--sft_config", type=str, help="Path to SFT training configuration JSON file"
    )
    parser.add_argument(
        "--grpo_config", type=str, help="Path to GRPO training configuration JSON file"
    )

    # Data paths (override config)
    parser.add_argument("--train_data", type=str, help="Path to training data JSONL file")
    parser.add_argument("--val_data", type=str, help="Path to validation data JSONL file")
    parser.add_argument("--image_base_path", type=str, help="Base path for images")

    # Model settings (override config)
    parser.add_argument(
        "--model_name",
        type=str,
        default="AIDC-AI/Ovis2.5-9B",
        help="Hugging Face model name or path",
    )
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient training")
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank")

    # Training hyperparameters (override config)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, help="Gradient accumulation steps"
    )

    # Ovis-specific parameters
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=896 * 896,
        help="Maximum pixels for image processing (Ovis native resolution)",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=True,
        help="Enable Ovis thinking mode during training",
    )
    parser.add_argument(
        "--thinking_budget", type=int, default=2048, help="Thinking budget for Ovis generation"
    )

    # Output directories
    parser.add_argument(
        "--sft_output_dir", type=str, default="outputs/sft", help="Output directory for SFT stage"
    )
    parser.add_argument(
        "--grpo_output_dir",
        type=str,
        default="outputs/grpo",
        help="Output directory for GRPO stage",
    )

    # GRPO specific
    parser.add_argument(
        "--sft_checkpoint", type=str, help="Path to SFT checkpoint for GRPO training"
    )
    parser.add_argument("--beta", type=float, help="KL penalty coefficient for GRPO")

    # Other settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,  # Default to bf16 for Ovis as per official guide
        help="Use BF16 training",
    )

    args = parser.parse_args()

    # Load configurations
    if args.model_config:
        model_config = load_config_from_json(args.model_config, ModelConfig)
    else:
        model_config = ModelConfig()

    if args.data_config:
        data_config = load_config_from_json(args.data_config, DataConfig)
    else:
        data_config = DataConfig()

    if args.sft_config:
        sft_config = load_config_from_json(args.sft_config, SFTTrainingConfig)
    else:
        sft_config = SFTTrainingConfig()

    if args.grpo_config:
        grpo_config = load_config_from_json(args.grpo_config, GRPOTrainingConfig)
    else:
        grpo_config = GRPOTrainingConfig()

    # Override configurations with command line arguments
    if args.model_name:
        model_config.model_name = args.model_name

    if args.train_data:
        data_config.train_data_path = args.train_data
    if args.val_data:
        data_config.val_data_path = args.val_data
    if args.image_base_path:
        data_config.image_base_path = args.image_base_path

    # Ovis-specific data config
    if args.max_pixels:
        data_config.max_pixels = args.max_pixels

    if args.learning_rate:
        sft_config.learning_rate = args.learning_rate
        grpo_config.learning_rate = args.learning_rate / 4  # GRPO typically uses lower LR

    if args.num_epochs:
        sft_config.num_train_epochs = args.num_epochs

    if args.batch_size:
        data_config.train_batch_size = args.batch_size
        grpo_config.mini_batch_size = args.batch_size

    if args.gradient_accumulation_steps:
        sft_config.gradient_accumulation_steps = args.gradient_accumulation_steps
        grpo_config.gradient_accumulation_steps = args.gradient_accumulation_steps

    if args.use_lora:
        sft_config.use_lora = True
        sft_config.lora_r = args.lora_r

    # Ovis generation parameters
    if args.thinking_budget:
        grpo_config.generation_kwargs["thinking_budget"] = args.thinking_budget

    grpo_config.generation_kwargs["enable_thinking"] = args.enable_thinking

    sft_config.output_dir = args.sft_output_dir
    grpo_config.output_dir = args.grpo_output_dir

    if args.sft_checkpoint:
        grpo_config.sft_checkpoint_path = args.sft_checkpoint
    else:
        grpo_config.sft_checkpoint_path = f"{args.sft_output_dir}/final_model"

    if args.beta:
        grpo_config.beta = args.beta

    sft_config.seed = args.seed
    grpo_config.seed = args.seed

    if args.fp16:
        sft_config.fp16 = True
        sft_config.bf16 = False
        grpo_config.fp16 = True
        grpo_config.bf16 = False
    elif args.bf16:
        sft_config.bf16 = True
        sft_config.fp16 = False
        grpo_config.bf16 = True
        grpo_config.fp16 = False

    # Print configuration summary
    logger.info("=" * 60)
    logger.info("OVIS2.5-9B TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model: {model_config.model_name}")
    logger.info(f"Training stages: {args.stage}")
    logger.info(f"Max pixels: {data_config.max_pixels}")
    logger.info(f"Thinking mode: {args.enable_thinking}")
    logger.info(f"Thinking budget: {args.thinking_budget}")
    logger.info(f"Use LoRA: {sft_config.use_lora}")
    if sft_config.use_lora:
        logger.info(f"LoRA rank: {sft_config.lora_r}")
    logger.info(f"Batch size: {data_config.train_batch_size}")
    logger.info(f"Learning rate: {sft_config.learning_rate}")
    logger.info(
        f"Mixed precision: {'bf16' if sft_config.bf16 else 'fp16' if sft_config.fp16 else 'fp32'}"
    )
    logger.info("=" * 60)

    # Run training stages
    if args.stage in ["sft", "both"]:
        logger.info("=" * 50)
        logger.info("Starting Stage 1: Supervised Fine-Tuning (SFT)")
        logger.info("=" * 50)

        sft_trainer = OvisSFTTrainer(
            model_config=model_config,
            data_config=data_config,
            training_config=sft_config,
        )
        sft_trainer.train()

        logger.info("Stage 1 completed successfully")

    if args.stage in ["grpo", "both"]:
        logger.info("=" * 50)
        logger.info("Starting Stage 2: Regional GRPO (R-GRPO)")
        logger.info("=" * 50)

        grpo_trainer = OvisGRPOTrainer(
            model_config=model_config,
            data_config=data_config,
            training_config=grpo_config,
        )
        grpo_trainer.train()

        logger.info("Stage 2 completed successfully")

    logger.info("=" * 50)
    logger.info("Ovis2.5-9B Training completed successfully!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
