"""
Stage 1: Supervised Fine-Tuning (SFT) for Ovis2.5-9B
Based on official guide: https://huggingface.co/AIDC-AI/Ovis2.5-9B
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import wandb
from accelerate import Accelerator
from config import DataConfig, ModelConfig, SFTTrainingConfig
from data_utils import OvisDataCollator, OvisDataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OvisSFTTrainer:
    """Trainer for Ovis2.5-9B Supervised Fine-Tuning"""

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: SFTTrainingConfig,
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

        # Setup directories
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories"""
        Path(self.training_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.training_config.output_dir}/logs").mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Load Ovis2.5-9B model"""
        logger.info(f"Loading model: {self.model_config.model_name}")

        # Model loading configuration
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.training_config.bf16 else torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        if self.model_config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load model using AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name, **model_kwargs
        )

        # Patch model for LoRA compatibility BEFORE adding tokens
        if self.training_config.use_lora:
            from lora_patch import patch_ovis_for_lora, validate_lora_compatibility

            patch_ovis_for_lora(self.model)

            if not validate_lora_compatibility(self.model):
                logger.error("LoRA compatibility validation failed")
                raise RuntimeError("Failed to make Ovis2.5 compatible with LoRA")

        # Handle custom tokens if specified
        if self.model_config.use_custom_tool_tokens:
            self.add_custom_tokens_safely()

        logger.info("Model loaded successfully")

        # 🎯 CRITICAL FIX: Do NOT enable gradient checkpointing - causes in-place operation errors
        logger.info("🚨 Gradient checkpointing DISABLED to prevent in-place operation errors")

        # Apply LoRA if specified
        if self.training_config.use_lora:
            self.apply_lora()

    def add_custom_tokens_safely(self):
        """Add custom tokens without breaking embeddings"""
        logger.info("Adding custom tool call tokens")

        # Get current sizes
        current_embedding_size = self.model.get_input_embeddings().weight.shape[0]
        current_vocab_size = len(self.model.text_tokenizer)

        logger.info(f"Current embedding size: {current_embedding_size}")
        logger.info(f"Current vocab size: {current_vocab_size}")

        # Add custom tokens
        new_tokens = [
            self.model_config.tool_call_start_token,
            self.model_config.tool_call_end_token,
        ]

        # Check if tokens already exist
        existing_tokens = set(self.model.text_tokenizer.get_vocab().keys())
        tokens_to_add = [token for token in new_tokens if token not in existing_tokens]

        if not tokens_to_add:
            logger.info("Custom tokens already exist in tokenizer")
            vocab = self.model.text_tokenizer.get_vocab()
            self.model_config.tool_call_start_token_id = vocab.get(new_tokens[0])
            self.model_config.tool_call_end_token_id = vocab.get(new_tokens[1])
            return

        # Add tokens
        num_added_tokens = self.model.text_tokenizer.add_tokens(tokens_to_add)
        logger.info(f"Added {num_added_tokens} new tokens: {tokens_to_add}")

        if num_added_tokens > 0:
            new_vocab_size = len(self.model.text_tokenizer)
            logger.info(f"New vocab size: {new_vocab_size}")

            # Only resize if absolutely necessary
            if new_vocab_size > current_embedding_size:
                logger.info(
                    f"Resizing embeddings from {current_embedding_size} to {new_vocab_size}"
                )
                try:
                    self.model.resize_token_embeddings(new_vocab_size)
                    logger.info("✅ Token embeddings resized successfully")
                except Exception as e:
                    logger.error(f"Embedding resize failed: {e}")
                    logger.warning("Continuing without embedding resize")
            else:
                logger.info(
                    f"No resize needed: vocab {new_vocab_size} <= embedding {current_embedding_size}"
                )

        # Update token IDs
        vocab = self.model.text_tokenizer.get_vocab()
        self.model_config.tool_call_start_token_id = vocab.get(new_tokens[0])
        self.model_config.tool_call_end_token_id = vocab.get(new_tokens[1])

        logger.info(f"✅ Custom tokens configured:")
        logger.info(f"   {new_tokens[0]} -> ID: {self.model_config.tool_call_start_token_id}")
        logger.info(f"   {new_tokens[1]} -> ID: {self.model_config.tool_call_end_token_id}")

    def apply_lora(self):
        """Apply LoRA configuration to the model"""
        logger.info("Applying LoRA configuration")

        # Prepare model for k-bit training if using quantization
        if self.model_config.load_in_8bit or self.model_config.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.training_config.lora_r,
            lora_alpha=self.training_config.lora_alpha,
            lora_dropout=self.training_config.lora_dropout,
            target_modules=self.training_config.lora_target_modules,
            bias="none",
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        logger.info("Preparing datasets")

        # Training dataset
        self.train_dataset = OvisDataset(
            data_path=self.data_config.train_data_path,
            image_base_path=self.data_config.image_base_path,
            model=self.model,
            max_length=self.data_config.max_length,
            max_pixels=self.data_config.max_pixels,
            stage="sft",
        )

        # Validation dataset
        self.val_dataset = OvisDataset(
            data_path=self.data_config.val_data_path,
            image_base_path=self.data_config.image_base_path,
            model=self.model,
            max_length=self.data_config.max_length,
            max_pixels=self.data_config.max_pixels,
            stage="sft",
        )

        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Validation dataset size: {len(self.val_dataset)}")

    def get_training_arguments(self) -> TrainingArguments:
        """Get training arguments for HuggingFace Trainer"""
        return TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.data_config.train_batch_size,
            per_device_eval_batch_size=self.data_config.eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_ratio=self.training_config.warmup_ratio,
            weight_decay=self.training_config.weight_decay,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            eval_strategy=self.training_config.eval_strategy,
            save_strategy=self.training_config.save_strategy,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            optim=self.training_config.optim,
            adam_epsilon=self.training_config.adam_epsilon,
            max_grad_norm=self.training_config.max_grad_norm,
            seed=self.training_config.seed,
            push_to_hub=self.training_config.push_to_hub,
            hub_model_id=self.training_config.hub_model_id,
            report_to=self.training_config.report_to,
            logging_dir=f"{self.training_config.output_dir}/logs",
            dataloader_num_workers=0,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            dataloader_drop_last=False,
            # 🎯 CRITICAL FIX: Explicitly disable gradient checkpointing
            gradient_checkpointing=False,  # MUST be False to avoid in-place operation errors
        )

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred

        # Calculate perplexity
        loss = torch.nn.functional.cross_entropy(
            predictions.view(-1, predictions.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        perplexity = torch.exp(loss)

        return {
            "perplexity": perplexity.item(),
        }

    def train(self):
        """Main training function"""
        logger.info("Starting Ovis2.5-9B SFT training")

        # Load model
        self.load_model()

        # Prepare datasets
        self.prepare_datasets()

        # Data collator using Ovis-specific collator
        data_collator = OvisDataCollator(
            model=self.model,
            padding=True,
        )

        # Training arguments
        training_args = self.get_training_arguments()

        # 🎯 FINAL CHECK: Ensure gradient checkpointing is disabled
        if (
            hasattr(training_args, "gradient_checkpointing")
            and training_args.gradient_checkpointing
        ):
            logger.warning(
                "⚠️  Forcing gradient_checkpointing = False to prevent in-place operation errors"
            )
            training_args.gradient_checkpointing = False

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
            ],
        )

        # Train
        train_result = trainer.train()

        # Save the final model
        logger.info("Saving final model")
        trainer.save_model(f"{self.training_config.output_dir}/final_model")

        # Save training metrics
        with open(f"{self.training_config.output_dir}/train_results.txt", "w") as f:
            f.write(str(train_result))

        logger.info("Ovis2.5-9B SFT training completed")

        return trainer, train_result


def main():
    """Main training entry point"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Ovis2.5-9B SFT Training")
    parser.add_argument("--model_config", type=str, help="Path to model configuration JSON file")
    parser.add_argument("--data_config", type=str, help="Path to data configuration JSON file")
    parser.add_argument(
        "--sft_config", type=str, help="Path to SFT training configuration JSON file"
    )
    parser.add_argument("--train_data", type=str, help="Path to training data JSONL file")
    parser.add_argument("--val_data", type=str, help="Path to validation data JSONL file")
    parser.add_argument("--image_base_path", type=str, help="Base path for images")
    parser.add_argument("--model_name", type=str, help="Hugging Face model name or path")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient training")
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_pixels", type=int, default=896 * 896, help="Maximum pixels for image processing"
    )
    parser.add_argument(
        "--enable_thinking", action="store_true", default=True, help="Enable Ovis thinking mode"
    )
    parser.add_argument(
        "--thinking_budget", type=int, default=2048, help="Thinking budget for Ovis generation"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/sft", help="Output directory for SFT stage"
    )
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

    if args.sft_config:
        with open(args.sft_config, "r") as f:
            config_dict = json.load(f)
        training_config = SFTTrainingConfig(**config_dict)
    else:
        training_config = SFTTrainingConfig()

    # Override configurations with command line arguments
    if args.model_name:
        model_config.model_name = args.model_name
    if args.train_data:
        data_config.train_data_path = args.train_data
    if args.val_data:
        data_config.val_data_path = args.val_data
    if args.image_base_path:
        data_config.image_base_path = args.image_base_path
    if args.max_pixels:
        data_config.max_pixels = args.max_pixels
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    if args.num_epochs:
        training_config.num_train_epochs = args.num_epochs
    if args.batch_size:
        data_config.train_batch_size = args.batch_size
    if args.gradient_accumulation_steps:
        training_config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.use_lora:
        training_config.use_lora = True
        training_config.lora_r = args.lora_r

    training_config.output_dir = args.output_dir
    training_config.seed = args.seed

    if args.fp16:
        training_config.fp16 = True
        training_config.bf16 = False
    elif args.bf16:
        training_config.bf16 = True
        training_config.fp16 = False

    # Print configuration summary
    logger.info("=" * 60)
    logger.info("OVIS2.5-9B SFT TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model: {model_config.model_name}")
    logger.info(f"Custom tokens: {model_config.use_custom_tool_tokens}")
    logger.info(f"Gradient checkpointing: {model_config.gradient_checkpointing}")  # Should be False
    logger.info(f"Max pixels: {data_config.max_pixels}")
    logger.info(f"Use LoRA: {training_config.use_lora}")
    if training_config.use_lora:
        logger.info(f"LoRA rank: {training_config.lora_r}")
    logger.info(f"Batch size: {data_config.train_batch_size}")
    logger.info(f"Learning rate: {training_config.learning_rate}")
    logger.info(
        f"Mixed precision: {'bf16' if training_config.bf16 else 'fp16' if training_config.fp16 else 'fp32'}"
    )
    logger.info("=" * 60)

    # Initialize trainer
    trainer = OvisSFTTrainer(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
