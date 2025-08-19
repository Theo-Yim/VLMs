"""
Stage 2: Regional GRPO (R-GRPO) Training for Qwen 2.5 VL
Based on VLM-R3 paper (https://arxiv.org/abs/2505.16192)
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from config import DataConfig, GRPOTrainingConfig, ModelConfig
from data_utils import QwenVLDataset
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

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

    @property
    def combined_score(self) -> float:
        """Compute combined reward score"""
        if not self.regional_scores:
            return self.global_score
        return 0.5 * self.global_score + 0.5 * np.mean(self.regional_scores)


class RegionalRewardModel:
    """
    Simplified regional reward model for R-GRPO
    Computes rewards based on text quality and tool call presence
    """

    def __init__(self, config: GRPOTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crop_tool = CropTool()

    def compute_regional_reward(
        self, generated_text: str, original_image: Optional[Image.Image] = None
    ) -> RegionalReward:
        """
        Compute reward based on tool call presence and text structure
        Since we don't have ground truth regions in the training data,
        we focus on structural quality of the response
        """
        # Extract tool calls from generated text
        tool_calls = self.crop_tool.extract_tool_calls(generated_text)

        # Compute regional scores based on tool call validity
        regional_scores = []
        for tool_call in tool_calls:
            # Check if coordinates are valid (within reasonable bounds)
            coords = tool_call["coordinates"]
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                # Basic validity check
                if x2 > x1 and y2 > y1:
                    regional_scores.append(1.0)
                else:
                    regional_scores.append(0.0)

        # Compute global score based on overall text quality
        global_score = self.compute_global_score(generated_text)

        return RegionalReward(
            global_score=global_score,
            regional_scores=regional_scores,
            regions=[tc["coordinates"] for tc in tool_calls],
        )

    def compute_global_score(self, text: str) -> float:
        """
        Compute global score for generated text
        This is a simplified version - you might want to use a more sophisticated method
        """
        # Check for presence of think and answer sections
        has_think = "<think>" in text and "</think>" in text
        has_answer = "<answer>" in text and "</answer>" in text

        # Check for tool calls
        has_tool_calls = "<tool_call>" in text and "</tool_call>" in text

        # Simple scoring based on structure
        score = 0.0
        if has_think:
            score += 0.3
        if has_answer:
            score += 0.3
        if has_tool_calls:
            score += 0.4

        # Penalize very short or very long responses
        text_len = len(text.split())
        if text_len < 10:
            score *= 0.5
        elif text_len > 500:
            score *= 0.8

        return min(1.0, score)


class QwenVLGRPOTrainer:
    """Trainer for Qwen 2.5 VL Regional GRPO"""

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
        self.reward_model = RegionalRewardModel(training_config)

        # Setup directories
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories"""
        Path(self.training_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.training_config.output_dir}/logs").mkdir(parents=True, exist_ok=True)

    def load_model_and_processor(self):
        """Load Qwen 2.5 VL model from SFT checkpoint"""
        logger.info(
            f"Loading model from SFT checkpoint: {self.training_config.sft_checkpoint_path}"
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.training_config.sft_checkpoint_path,
            trust_remote_code=True,
        )

        # Model loading configuration
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.training_config.bf16 else torch.float16,
            "device_map": "auto",
        }

        if self.model_config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load base model and apply LoRA from checkpoint
        base_model = AutoModelForVision2Seq.from_pretrained(
            self.model_config.model_name, **model_kwargs
        )

        # Load LoRA weights if they exist
        lora_path = Path(self.training_config.sft_checkpoint_path) / "adapter_model.bin"
        if lora_path.exists():
            self.model = PeftModel.from_pretrained(
                base_model, self.training_config.sft_checkpoint_path, is_trainable=True
            )
        else:
            self.model = base_model

        # Enable gradient checkpointing if specified
        if self.model_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        logger.info("Model and processor loaded successfully")

    def prepare_datasets(self):
        """Prepare training dataset for GRPO"""
        logger.info("Preparing dataset for GRPO")

        # Training dataset with GRPO stage flag
        self.train_dataset = QwenVLDataset(
            data_path=self.data_config.train_data_path,
            image_base_path=self.data_config.image_base_path,
            processor=self.processor,
            max_length=self.data_config.max_length,
            image_size=self.data_config.image_size,
            stage="grpo",  # Important: set stage to grpo
        )

        # Create DataLoader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.mini_batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
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
        Perform one GRPO training step
        """
        # Generate responses
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                **self.training_config.generation_kwargs,
            )

        # Decode generated texts
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Compute rewards
        rewards = self.compute_rewards(generated_texts, batch.get("original_images", None))

        # Compute log probabilities of generated sequences
        with torch.cuda.amp.autocast(enabled=self.training_config.bf16):
            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                pixel_values=batch.get("pixel_values"),
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
        logger.info("Starting R-GRPO training")

        # Load model and processor
        self.load_model_and_processor()

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

        logger.info("R-GRPO training completed")

    def save_checkpoint(self, step: Any):
        """Save model checkpoint"""
        save_dir = f"{self.training_config.output_dir}/checkpoint-{step}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Save model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_dir)

        # Save processor
        self.processor.save_pretrained(save_dir)

        logger.info(f"Checkpoint saved to {save_dir}")


def main():
    """Main training entry point"""

    # Initialize configurations
    model_config = ModelConfig()
    data_config = DataConfig()
    training_config = GRPOTrainingConfig()

    # Initialize trainer
    trainer = QwenVLGRPOTrainer(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
