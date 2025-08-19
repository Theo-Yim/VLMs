"""
Stage 2: Regional GRPO (R-GRPO) Training for Ovis2.5-9B
🎯 FIXED: Disable gradient checkpointing to prevent in-place operation errors
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from config import DataConfig, GRPOTrainingConfig, ModelConfig
from data_utils import GroundingParser, OvisDataset
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM  # Ovis uses AutoModelForCausalLM

sys.path.append("..")
from crop_tool import CropTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegionalReward:
    """Container for regional reward computation"""

    global_score: float
    regional_scores: List[float]
    regions: List[List[float]]  # [x1, y1, x2, y2] for each region
    grounding_elements: Dict[str, List]  # Parsed grounding elements

    @property
    def combined_score(self) -> float:
        """Compute combined reward score"""
        if not self.regional_scores:
            return self.global_score
        return 0.5 * self.global_score + 0.5 * np.mean(self.regional_scores)


class OvisRegionalRewardModel:
    """
    Regional reward model for Ovis2.5-9B R-GRPO
    Computes rewards based on text quality and grounding accuracy
    """

    def __init__(self, config: GRPOTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grounding_parser = GroundingParser()
        self.crop_tool = CropTool()  # Use existing CropTool from QwenVL2.5

    def compute_regional_reward(
        self, generated_text: str, original_image: Optional[Image.Image] = None
    ) -> RegionalReward:
        """
        Compute reward based on grounding quality and text structure
        Uses Ovis grounding format: <ref>, <box>, <point>
        """
        # Parse grounding elements from generated text
        grounding_elements = self.grounding_parser.parse_grounding(generated_text)

        # Compute regional scores based on grounding validity
        regional_scores = []
        regions = []

        # Check box validity
        for box in grounding_elements["boxes"]:
            if len(box) == 4:
                x1, y1, x2, y2 = box
                # Basic validity check for normalized coordinates [0,1)
                if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
                    regional_scores.append(1.0)
                    regions.append(box)
                else:
                    regional_scores.append(0.0)

        # Check point validity
        for point in grounding_elements["points"]:
            if len(point) == 2:
                x, y = point
                # Basic validity check for normalized coordinates [0,1)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    regional_scores.append(1.0)
                    regions.append([x, y, x, y])  # Convert point to box format
                else:
                    regional_scores.append(0.0)

        # Compute global score based on overall text quality
        global_score = self.compute_global_score(generated_text, grounding_elements)

        return RegionalReward(
            global_score=global_score,
            regional_scores=regional_scores,
            regions=regions,
            grounding_elements=grounding_elements,
        )

    def compute_global_score(self, text: str, grounding_elements: Dict[str, List]) -> float:
        """
        Compute global score for generated text
        Enhanced for Ovis grounding format
        """
        # Check for presence of think and answer sections
        has_think = "<think>" in text and "</think>" in text
        has_answer = "<answer>" in text and "</answer>" in text

        # Check for proper grounding elements
        has_refs = len(grounding_elements["refs"]) > 0
        has_boxes = len(grounding_elements["boxes"]) > 0
        has_points = len(grounding_elements["points"]) > 0
        has_grounding = has_refs and (has_boxes or has_points)

        # Check for reference-grounding alignment
        ref_grounding_aligned = len(grounding_elements["refs"]) == (
            len(grounding_elements["boxes"]) + len(grounding_elements["points"])
        )

        # Simple scoring based on structure
        score = 0.0
        if has_think:
            score += 0.25
        if has_answer:
            score += 0.25
        if has_grounding:
            score += 0.3
        if ref_grounding_aligned:
            score += 0.2

        # Penalize very short or very long responses
        text_len = len(text.split())
        if text_len < 10:
            score *= 0.5
        elif text_len > 500:
            score *= 0.8

        return min(1.0, score)


class OvisGRPOTrainer:
    """Trainer for Ovis2.5-9B Regional GRPO"""

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: GRPOTrainingConfig,
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision="bf16"
            if training_config.bf16
            else "fp16"
            if training_config.fp16
            else "no",
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        )

        # Initialize reward model
        self.reward_model = OvisRegionalRewardModel(training_config)

        # Setup directories
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories"""
        Path(self.training_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.training_config.output_dir}/logs").mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Load Ovis2.5-9B model from SFT checkpoint"""
        logger.info(
            f"Loading model from SFT checkpoint: {self.training_config.sft_checkpoint_path}"
        )

        # Model loading configuration
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.training_config.bf16 else torch.float16,
            "device_map": "auto",
        }

        if self.model_config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Check if loading from LoRA checkpoint or full model
        checkpoint_path = Path(self.training_config.sft_checkpoint_path)
        adapter_path = checkpoint_path / "adapter_model.safetensors"

        if adapter_path.exists():
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_name, **model_kwargs
            )

            # Apply LoRA compatibility patch before loading adapter
            from lora_patch import patch_ovis_for_lora, validate_lora_compatibility

            patch_ovis_for_lora(base_model)

            # Validate the patch worked
            if not validate_lora_compatibility(base_model):
                logger.error("LoRA compatibility validation failed")
                raise RuntimeError("Failed to make Ovis2.5 compatible with LoRA")

            # Load LoRA weights
            self.model = PeftModel.from_pretrained(
                base_model, self.training_config.sft_checkpoint_path, is_trainable=True
            )
        else:
            # Load full fine-tuned model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.training_config.sft_checkpoint_path, **model_kwargs
            )

            # Apply LoRA compatibility patch for potential future use
            from lora_patch import patch_ovis_for_lora

            patch_ovis_for_lora(self.model)

        # 🎯 CRITICAL FIX: Do NOT enable gradient checkpointing - causes in-place operation errors
        logger.info("🚨 Gradient checkpointing DISABLED to prevent in-place operation errors")

        logger.info("Model loaded successfully")

    def prepare_datasets(self):
        """Prepare training dataset for GRPO"""
        logger.info("Preparing dataset for GRPO")

        # Training dataset with GRPO stage flag
        self.train_dataset = OvisDataset(
            data_path=self.data_config.train_data_path,
            image_base_path=self.data_config.image_base_path,
            model=self.model,
            max_length=self.data_config.max_length,
            max_pixels=self.data_config.max_pixels,
            stage="grpo",  # Important: set stage to grpo
        )

        # Create DataLoader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.mini_batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for batch_size=1
        )

        logger.info(f"Train dataset size: {len(self.train_dataset)}")

    def compute_rewards(
        self, generated_texts: List[str], original_images: Optional[List[Image.Image]] = None
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of generated texts
        """
        rewards = []

        for i, gen_text in enumerate(generated_texts):
            # Get original image if available
            orig_img = original_images[i] if original_images else None

            # Compute regional reward
            regional_reward = self.reward_model.compute_regional_reward(gen_text, orig_img)

            # Compute combined reward
            if self.training_config.use_regional_rewards:
                reward = (
                    self.training_config.global_reward_weight * regional_reward.global_score
                    + self.training_config.region_reward_weight
                    * np.mean(regional_reward.regional_scores)
                    if regional_reward.regional_scores
                    else regional_reward.global_score
                )
            else:
                reward = regional_reward.global_score

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def grpo_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform one GRPO training step using Ovis generation
        """
        # 🎯 CRITICAL FIX: Use torch.no_grad() for generation to prevent gradient issues
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs=batch["input_ids"],
                pixel_values=batch.get("pixel_values"),
                grid_thws=batch.get("grid_thws"),
                # Use Ovis-specific generation parameters
                enable_thinking=self.training_config.generation_kwargs.get("enable_thinking", True),
                enable_thinking_budget=self.training_config.generation_kwargs.get(
                    "enable_thinking_budget", True
                ),
                max_new_tokens=self.training_config.generation_kwargs.get("max_new_tokens", 3072),
                thinking_budget=self.training_config.generation_kwargs.get("thinking_budget", 2048),
                temperature=self.training_config.generation_kwargs.get("temperature", 0.7),
                top_p=self.training_config.generation_kwargs.get("top_p", 0.9),
                do_sample=self.training_config.generation_kwargs.get("do_sample", True),
            )

        # Decode generated texts using model's text tokenizer
        generated_texts = [
            self.model.text_tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_ids
        ]

        # Compute rewards
        rewards = self.compute_rewards(generated_texts, batch.get("original_images", None))

        # Compute log probabilities of generated sequences
        with torch.cuda.amp.autocast(enabled=self.training_config.bf16):
            outputs = self.model(
                input_ids=generated_ids,
                pixel_values=batch.get("pixel_values"),
                grid_thws=batch.get("grid_thws"),
                labels=generated_ids,
            )

        # Compute policy gradient loss
        # Simplified version - in practice, you'd want more sophisticated PPO-style updates
        log_probs = -outputs.loss  # Negative loss as proxy for log probability

        # Compute advantages (rewards - baseline)
        baseline = rewards.mean()
        advantages = rewards - baseline

        # Policy gradient loss
        pg_loss = -(log_probs * advantages.detach()).mean()

        # Add KL penalty if specified
        if self.training_config.beta > 0:
            # Compute KL divergence from reference model
            # This is simplified - you'd need to store reference model outputs
            kl_penalty = self.training_config.beta * outputs.loss
            total_loss = pg_loss + kl_penalty
        else:
            total_loss = pg_loss

        # Backward pass
        self.accelerator.backward(total_loss)

        return {
            "loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std().item(),
        }

    def train(self):
        """Main GRPO training loop"""
        logger.info("Starting Ovis2.5-9B R-GRPO training")

        # Load model
        self.load_model()

        # Prepare datasets
        self.prepare_datasets()

        # Prepare model and dataloader with accelerator
        self.model, self.train_dataloader = self.accelerator.prepare(
            self.model, self.train_dataloader
        )

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.training_config.learning_rate, weight_decay=0.01
        )

        # Training loop
        global_step = 0
        for epoch in range(self.training_config.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.training_config.num_train_epochs}")

            epoch_losses = []
            epoch_rewards = []

            for batch_idx, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {
                    k: v.to(self.accelerator.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                # Perform GRPO step
                metrics = self.grpo_step(batch)

                # Optimizer step
                if (batch_idx + 1) % self.training_config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    global_step += 1

                    # Logging
                    if global_step % self.training_config.logging_steps == 0:
                        logger.info(
                            f"Step {global_step}: "
                            f"Loss = {metrics['loss']:.4f}, "
                            f"Mean Reward = {metrics['mean_reward']:.4f}"
                        )

                    # Save checkpoint
                    if global_step % self.training_config.save_steps == 0:
                        self.save_checkpoint(global_step)

                epoch_losses.append(metrics["loss"])
                epoch_rewards.append(metrics["mean_reward"])

            # Log epoch metrics
            avg_loss = np.mean(epoch_losses)
            avg_reward = np.mean(epoch_rewards)
            logger.info(
                f"Epoch {epoch + 1} completed: "
                f"Avg Loss = {avg_loss:.4f}, "
                f"Avg Reward = {avg_reward:.4f}"
            )

        # Save final model
        logger.info("Saving final model")
        self.save_checkpoint("final")

        logger.info("Ovis2.5-9B R-GRPO training completed")

    def save_checkpoint(self, step: Any):
        """Save model checkpoint"""
        save_dir = f"{self.training_config.output_dir}/checkpoint-{step}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Save model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_dir)

        logger.info(f"Checkpoint saved to {save_dir}")


def main():
    """Main training entry point"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Ovis2.5-9B R-GRPO Training")

    # Configuration files
    parser.add_argument("--model_config", type=str, help="Path to model configuration JSON file")
    parser.add_argument("--data_config", type=str, help="Path to data configuration JSON file")
    parser.add_argument(
        "--grpo_config", type=str, help="Path to GRPO training configuration JSON file"
    )

    # Data paths (override config)
    parser.add_argument("--train_data", type=str, help="Path to training data JSONL file")
    parser.add_argument("--image_base_path", type=str, help="Base path for images")

    # Model settings (override config)
    parser.add_argument("--model_name", type=str, help="Hugging Face model name or path")
    parser.add_argument(
        "--sft_checkpoint", type=str, help="Path to SFT checkpoint for GRPO training"
    )

    # Training hyperparameters (override config)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, help="Gradient accumulation steps"
    )

    # Ovis-specific parameters
    parser.add_argument(
        "--max_pixels", type=int, default=896 * 896, help="Maximum pixels for image processing"
    )
    parser.add_argument(
        "--enable_thinking", action="store_true", default=True, help="Enable Ovis thinking mode"
    )
    parser.add_argument(
        "--thinking_budget", type=int, default=2048, help="Thinking budget for Ovis generation"
    )

    # GRPO specific parameters
    parser.add_argument("--beta", type=float, help="KL penalty coefficient for GRPO")
    parser.add_argument(
        "--use_regional_rewards", action="store_true", default=True, help="Use regional rewards"
    )
    parser.add_argument(
        "--global_reward_weight", type=float, default=0.5, help="Weight for global reward"
    )
    parser.add_argument(
        "--region_reward_weight", type=float, default=0.5, help="Weight for regional reward"
    )

    # Output directory
    parser.add_argument(
        "--output_dir", type=str, default="outputs/grpo", help="Output directory for GRPO stage"
    )

    # Other settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use BF16 training")

    args = parser.parse_args()

    # Load configurations
    if args.model_config:
        with open(args.model_config, "r") as f:
            config_dict = json.load(f)
        model_config = ModelConfig(**config_dict)
    else:
        model_config = ModelConfig()

    if args.data_config:
        with open(args.data_config, "r") as f:
            config_dict = json.load(f)
        data_config = DataConfig(**config_dict)
    else:
        data_config = DataConfig()

    if args.grpo_config:
        with open(args.grpo_config, "r") as f:
            config_dict = json.load(f)
        training_config = GRPOTrainingConfig(**config_dict)
    else:
        training_config = GRPOTrainingConfig()

    # Override configurations with command line arguments
    if args.model_name:
        model_config.model_name = args.model_name

    if args.train_data:
        data_config.train_data_path = args.train_data
    if args.image_base_path:
        data_config.image_base_path = args.image_base_path

    if args.max_pixels:
        data_config.max_pixels = args.max_pixels

    if args.learning_rate:
        training_config.learning_rate = args.learning_rate

    if args.num_epochs:
        training_config.num_train_epochs = args.num_epochs

    if args.batch_size:
        training_config.mini_batch_size = args.batch_size

    if args.gradient_accumulation_steps:
        training_config.gradient_accumulation_steps = args.gradient_accumulation_steps

    # Ovis generation parameters
    if args.thinking_budget:
        training_config.generation_kwargs["thinking_budget"] = args.thinking_budget

    training_config.generation_kwargs["enable_thinking"] = args.enable_thinking

    training_config.output_dir = args.output_dir

    if args.sft_checkpoint:
        training_config.sft_checkpoint_path = args.sft_checkpoint

    if args.beta:
        training_config.beta = args.beta

    training_config.seed = args.seed

    if args.fp16:
        training_config.fp16 = True
        training_config.bf16 = False
    elif args.bf16:
        training_config.bf16 = True
        training_config.fp16 = False

    # GRPO specific settings
    if args.use_regional_rewards:
        training_config.use_regional_rewards = True
        training_config.global_reward_weight = args.global_reward_weight
        training_config.region_reward_weight = args.region_reward_weight

    # Print configuration summary
    logger.info("=" * 60)
    logger.info("OVIS2.5-9B R-GRPO TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model: {model_config.model_name}")
    logger.info(f"SFT Checkpoint: {training_config.sft_checkpoint_path}")
    logger.info(f"Gradient checkpointing: {model_config.gradient_checkpointing}")  # Should be False
    logger.info(f"Max pixels: {data_config.max_pixels}")
    logger.info(f"Thinking mode: {args.enable_thinking}")
    logger.info(f"Thinking budget: {args.thinking_budget}")
    logger.info(f"Batch size: {training_config.mini_batch_size}")
    logger.info(f"Learning rate: {training_config.learning_rate}")
    logger.info(f"Beta (KL penalty): {training_config.beta}")
    logger.info(f"Use regional rewards: {training_config.use_regional_rewards}")
    logger.info(
        f"Mixed precision: {'bf16' if training_config.bf16 else 'fp16' if training_config.fp16 else 'fp32'}"
    )
    logger.info("=" * 60)

    # Initialize trainer
    trainer = OvisGRPOTrainer(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
