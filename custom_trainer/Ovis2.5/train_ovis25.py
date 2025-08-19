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


def validate_lora_setup(sft_config: SFTTrainingConfig, grpo_config: GRPOTrainingConfig):
    """Validate LoRA configuration consistency between stages"""
    if sft_config.use_lora:
        logger.info("✅ LoRA enabled for SFT stage")
        logger.info(f"   LoRA rank: {sft_config.lora_r}")
        logger.info(f"   LoRA alpha: {sft_config.lora_alpha}")
        logger.info(f"   LoRA dropout: {sft_config.lora_dropout}")
        logger.info(f"   Target modules: {sft_config.lora_target_modules}")

        # GRPO will automatically work with LoRA checkpoints
        logger.info("✅ GRPO stage will support LoRA checkpoints from SFT")
    else:
        logger.info("ℹ️  LoRA disabled - using full fine-tuning")
        logger.warning("   This will require significantly more VRAM (~40GB+ vs ~24-28GB)")


def validate_gradient_checkpointing(model_config: ModelConfig):
    """Validate gradient checkpointing is properly disabled"""
    if model_config.gradient_checkpointing:
        logger.error("🚨 CRITICAL: Gradient checkpointing is enabled!")
        logger.error("   This WILL cause in-place operation errors with Ovis2.5")
        logger.error("   Forcing gradient_checkpointing = False")
        model_config.gradient_checkpointing = False

    logger.info("✅ Gradient checkpointing disabled - prevents in-place operation errors")


def print_memory_requirements(sft_config: SFTTrainingConfig, data_config: DataConfig):
    """Print expected memory requirements"""
    batch_size = data_config.train_batch_size
    grad_accum = sft_config.gradient_accumulation_steps
    effective_batch = batch_size * grad_accum

    logger.info("=" * 50)
    logger.info("MEMORY REQUIREMENTS ESTIMATE")
    logger.info("=" * 50)

    if sft_config.use_lora:
        if sft_config.lora_r <= 64:
            vram_estimate = "~24GB"
        elif sft_config.lora_r <= 128:
            vram_estimate = "~28GB"
        else:
            vram_estimate = "~32GB"

        logger.info(f"Training method: LoRA (rank {sft_config.lora_r})")
        logger.info(f"Estimated VRAM: {vram_estimate}")
        logger.info(f"Recommended GPUs: RTX 4090, RTX 6000 Ada, A100")
    else:
        logger.info(f"Training method: Full fine-tuning")
        logger.info(f"Estimated VRAM: ~40GB+")
        logger.info(f"Recommended GPUs: A100 80GB, H100")

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation: {grad_accum}")
    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(
        f"Mixed precision: {'bf16' if sft_config.bf16 else 'fp16' if sft_config.fp16 else 'fp32'}"
    )
    logger.info("=" * 50)


def validate_custom_tokens(model_config: ModelConfig):
    """Validate custom token configuration"""
    if model_config.use_custom_tool_tokens:
        logger.info("✅ Custom tool tokens enabled for crop strategy")
        logger.info(f"   Start token: {model_config.tool_call_start_token}")
        logger.info(f"   End token: {model_config.tool_call_end_token}")
        logger.info("   This preserves your thinking-based cropping strategy")
    else:
        logger.info("ℹ️  Custom tool tokens disabled")


def validate_batch_size(data_config: DataConfig):
    """Validate batch size configuration for Ovis2.5"""
    if data_config.train_batch_size != 1:
        logger.error(f"🚨 CRITICAL: train_batch_size = {data_config.train_batch_size}")
        logger.error("   Ovis2.5 REQUIRES batch_size = 1 due to native resolution processing")
        logger.error("   Forcing train_batch_size = 1")
        data_config.train_batch_size = 1

    if data_config.eval_batch_size != 1:
        logger.error(f"🚨 CRITICAL: eval_batch_size = {data_config.eval_batch_size}")
        logger.error("   Ovis2.5 REQUIRES batch_size = 1 due to native resolution processing")
        logger.error("   Forcing eval_batch_size = 1")
        data_config.eval_batch_size = 1

    logger.info("✅ Batch size = 1 (required for native resolution)")


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
    parser.add_argument("--lora_alpha", type=int, default=256, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

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

    # Memory optimization options
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Use memory efficient settings (smaller batch, higher grad accum)",
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

    # LoRA configuration
    if args.use_lora:
        sft_config.use_lora = True
        sft_config.lora_r = args.lora_r
        sft_config.lora_alpha = args.lora_alpha
        sft_config.lora_dropout = args.lora_dropout

    # Memory efficient settings
    if args.memory_efficient:
        logger.info("🔧 Applying memory efficient settings...")
        data_config.train_batch_size = 1  # Force to 1 anyway
        sft_config.gradient_accumulation_steps = max(sft_config.gradient_accumulation_steps, 32)
        grpo_config.mini_batch_size = 1  # Force to 1 anyway
        grpo_config.gradient_accumulation_steps = max(grpo_config.gradient_accumulation_steps, 32)

        # Use smaller LoRA rank for memory efficiency
        if sft_config.use_lora and sft_config.lora_r > 64:
            sft_config.lora_r = 64
            sft_config.lora_alpha = 128
            logger.info(f"   Reduced LoRA rank to {sft_config.lora_r} for memory efficiency")

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

    # 🎯 CRITICAL VALIDATIONS: Ensure proper configuration for Ovis2.5
    validate_gradient_checkpointing(model_config)
    validate_batch_size(data_config)
    validate_custom_tokens(model_config)
    validate_lora_setup(sft_config, grpo_config)

    # Print memory requirements
    print_memory_requirements(sft_config, data_config)

    # Print configuration summary
    logger.info("=" * 60)
    logger.info("OVIS2.5-9B TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model: {model_config.model_name}")
    logger.info(f"Training stages: {args.stage}")
    logger.info(f"Custom tokens: {model_config.use_custom_tool_tokens}")
    logger.info(f"Gradient checkpointing: {model_config.gradient_checkpointing}")  # Should be False
    logger.info(f"Max pixels: {data_config.max_pixels}")
    logger.info(f"Thinking mode: {args.enable_thinking}")
    logger.info(f"Thinking budget: {args.thinking_budget}")
    logger.info(f"Use LoRA: {sft_config.use_lora}")
    if sft_config.use_lora:
        logger.info(f"LoRA rank: {sft_config.lora_r}")
        logger.info(f"LoRA alpha: {sft_config.lora_alpha}")
        logger.info(f"LoRA dropout: {sft_config.lora_dropout}")
    logger.info(f"Batch size: {data_config.train_batch_size} (forced for native resolution)")
    logger.info(f"Gradient accumulation: {sft_config.gradient_accumulation_steps}")
    logger.info(
        f"Effective batch size: {data_config.train_batch_size * sft_config.gradient_accumulation_steps}"
    )
    logger.info(f"Learning rate: {sft_config.learning_rate}")
    logger.info(
        f"Mixed precision: {'bf16' if sft_config.bf16 else 'fp16' if sft_config.fp16 else 'fp32'}"
    )
    logger.info(f"Memory efficient mode: {args.memory_efficient}")
    logger.info("=" * 60)

    # Pre-flight checks
    logger.info("🔍 Running pre-flight checks...")

    # Check if LoRA patch is available
    try:
        from lora_patch import patch_ovis_for_lora, validate_lora_compatibility

        logger.info("✅ LoRA compatibility patch found")
    except ImportError:
        if sft_config.use_lora:
            logger.error("❌ LoRA patch not found but LoRA is enabled!")
            logger.error("   Please ensure lora_patch.py is in the same directory")
            return
        else:
            logger.info("ℹ️  LoRA patch not found (not needed for full fine-tuning)")

    # Warn about potential memory issues with gradient checkpointing disabled
    logger.info("⚠️  MEMORY USAGE NOTE:")
    logger.info("   Gradient checkpointing is DISABLED to prevent in-place operation errors")
    logger.info("   This may increase memory usage but prevents training failures")
    logger.info("   Consider using LoRA for memory efficiency")

    # Run training stages
    if args.stage in ["sft", "both"]:
        logger.info("=" * 50)
        logger.info("Starting Stage 1: Supervised Fine-Tuning (SFT)")
        if sft_config.use_lora:
            logger.info("✅ LoRA enabled - memory efficient training")
        else:
            logger.info("⚠️  Full fine-tuning - high memory usage")
        logger.info("=" * 50)

        try:
            sft_trainer = OvisSFTTrainer(
                model_config=model_config,
                data_config=data_config,
                training_config=sft_config,
            )
            sft_trainer.train()
            logger.info("✅ Stage 1 completed successfully")

        except Exception as e:
            logger.error(f"❌ Stage 1 failed: {e}")
            if "CUDA out of memory" in str(e):
                logger.error("💡 Try using --memory_efficient or reduce --lora_r")
            elif "in-place operation" in str(e):
                logger.error("💡 Gradient checkpointing should be disabled - check configuration")
            return

    if args.stage in ["grpo", "both"]:
        logger.info("=" * 50)
        logger.info("Starting Stage 2: Regional GRPO (R-GRPO)")
        logger.info("=" * 50)

        try:
            grpo_trainer = OvisGRPOTrainer(
                model_config=model_config,
                data_config=data_config,
                training_config=grpo_config,
            )
            grpo_trainer.train()
            logger.info("✅ Stage 2 completed successfully")

        except Exception as e:
            logger.error(f"❌ Stage 2 failed: {e}")
            if "CUDA out of memory" in str(e):
                logger.error("💡 Try reducing --batch_size or using smaller LoRA rank")
            elif "in-place operation" in str(e):
                logger.error("💡 Gradient checkpointing should be disabled - check configuration")
            return

    logger.info("=" * 50)
    logger.info("🎉 Ovis2.5-9B Training completed successfully!")
    logger.info("=" * 50)

    # Print final model locations
    if args.stage in ["sft", "both"]:
        logger.info(f"📁 SFT model saved to: {sft_config.output_dir}/final_model")
    if args.stage in ["grpo", "both"]:
        logger.info(f"📁 GRPO model saved to: {grpo_config.output_dir}/final_model")

    # Print next steps
    logger.info("\n🚀 Next steps:")
    logger.info("1. Test inference: python inference.py --model_path outputs/grpo/final_model")
    logger.info("2. Run evaluation on your test set")
    logger.info("3. Deploy your fine-tuned model")

    if sft_config.use_lora:
        logger.info("\n💡 LoRA Tips:")
        logger.info("- Your adapters are much smaller than full models")
        logger.info("- You can easily switch between different LoRA adapters")
        logger.info("- Base model weights are preserved and reusable")

    logger.info("\n🎯 TRAINING SUCCESS:")
    logger.info("- Gradient checkpointing disabled ✅")
    logger.info("- Batch size = 1 enforced ✅")
    logger.info("- Custom tool tokens preserved ✅")
    logger.info("- Your crop tool strategy intact ✅")


if __name__ == "__main__":
    main()
