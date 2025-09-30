"""
Ovis2.5 Fine-tuning Script using TRL SFTTrainer
Based on original Ovis training framework with custom modeling implementation
"""

import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from ovis.model.modeling_ovis2_5 import Ovis2_5
from ovis.train.dataset.conversation_dataset import ConversationDataset
from ovis.train.dataset.multimodal_dataset import DataCollatorForMultimodalDataset
from transformers import (
    EarlyStoppingCallback,
    HfArgumentParser,
    set_seed,
)

# Import TRL components
from trl import SFTConfig, SFTTrainer

try:
    import flash_attn

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


@dataclass
class ModelArguments:
    """Arguments for model configuration"""

    model_path: str = field(default="AIDC-AI/Ovis2.5-9B")
    visual_vocab_size: int = field(default=65536)


@dataclass
class CustomSFTConfig(SFTConfig):
    """Extended SFT configuration for Ovis2.5 based on TRL SFTConfig"""

    # Ovis-specific arguments
    train_modules: str = field(default="all")  # all, llm, visual_tokenizer, vte, etc.
    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)

    # Override some defaults for Ovis2.5
    ovis_pretrained_path: Optional[str] = field(default="AIDC-AI/Ovis2.5-9B")
    stage: Optional[int] = field(default=3)  # Stage 3 for full training

    # Data paths
    data_path: str = field(default="./data/train_data.json")
    eval_data_path: Optional[str] = field(default=None)
    image_folder: str = field(default="./data/images")
    data_name: str = field(default="test_dataset")
    data_type: str = field(default="conversation")

    # Multimodal-specific parameters (from original Ovis training args)
    multimodal_max_length: int = field(default=8192)
    text_max_length: Optional[int] = field(default=4096)
    single_image_min_pixels: int = field(default=448 * 448)
    single_image_max_pixels: int = field(default=1792 * 1792)
    multiple_image_min_pixels: int = field(default=448 * 448)
    multiple_image_max_pixels: int = field(default=896 * 896)
    video_min_pixels: int = field(default=448 * 448)
    video_max_pixels: int = field(default=896 * 896)
    min_frames: int = field(default=8)
    max_frames: int = field(default=8)

    # SFT-specific parameters (correct placement)
    packing: bool = field(default=False)  # Set to False for multimodal data
    max_seq_length: Optional[int] = field(
        default=8192
    )  # Controls sequence length TODO Check if it is needed
    dataset_text_field: Optional[str] = field(default=None)  # For text-only datasets
    formatting_func: Optional[str] = field(default=None)  # Alternative to dataset_text_field

    def __post_init__(self):
        super().__post_init__()
        # Sync max_seq_length with multimodal_max_length
        if self.max_seq_length != self.multimodal_max_length:
            self.max_seq_length = self.multimodal_max_length

        # For custom multimodal datasets, we need these settings
        self.remove_unused_columns = False  # Important for multimodal data
        self.dataset_kwargs = {"skip_prepare_dataset": True}  # Skip default text processing


def create_dataset_info(training_args: CustomSFTConfig) -> Dict[str, Dict]:
    """Create dataset info structure expected by original ConversationDataset"""
    return {
        training_args.data_name: {
            "meta_file": training_args.data_path,
            "storage_type": "hybrid",
            "data_format": "conversation",
            "image_dir": training_args.image_folder,
        }
    }


def setup_model_for_training(model: Ovis2_5, training_args: CustomSFTConfig):
    """Setup model parameters for training"""

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Selectively unfreeze based on train_modules
    train_modules = training_args.train_modules.lower()

    if "all" in train_modules:
        for param in model.parameters():
            param.requires_grad = True
    else:
        if "llm" in train_modules and not training_args.freeze_llm:
            for param in model.llm.parameters():
                param.requires_grad = True

        if "visual_tokenizer" in train_modules and not training_args.freeze_vision_tower:
            for param in model.visual_tokenizer.parameters():
                param.requires_grad = True

        if "vte" in train_modules:
            for param in model.vte.parameters():
                param.requires_grad = True

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")


def main():
    """Main training function using TRL SFTTrainer"""

    config_file = (
        os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "./Ovis/src_theo/train_config.json"
    )

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, CustomSFTConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(
            json_file=config_file, allow_extra_keys=True
        )
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed
    set_seed(training_args.seed)

    # Set device for model loading (DeepSpeed will handle distribution)
    device_map = f"cuda:{training_args.local_rank}" if training_args.local_rank != -1 else "cuda:0"

    # Load model with device map
    print(f"Loading model from {model_args.model_path}...")
    model = Ovis2_5.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    )

    # Setup model for training
    setup_model_for_training(model, training_args)

    # Prepare tokenizer
    tokenizer = model.text_tokenizer
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset info structure for original ConversationDataset
    dataset_info = create_dataset_info(training_args)

    # Create dataset using original ConversationDataset
    print(f"Loading training dataset from {training_args.data_path}...")
    train_dataset = ConversationDataset(
        name=training_args.data_name,
        info=dataset_info[training_args.data_name],
        model=model,
        training_args=training_args,
    )
    print(f"Training dataset loaded: {len(train_dataset)} samples")

    # Calculate max_steps to avoid dataloader length issues
    dataset_size = len(train_dataset)
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )
    steps_per_epoch = dataset_size // effective_batch_size
    if dataset_size % effective_batch_size != 0:
        steps_per_epoch += 1  # Round up for partial batches

    print("Dataset analysis:")
    print(f"  - Dataset size: {dataset_size}")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Steps per epoch: {steps_per_epoch}")

    # If max_steps is not set or is -1, calculate it
    if training_args.max_steps <= 0:
        calculated_max_steps = steps_per_epoch * training_args.num_train_epochs
        training_args.max_steps = calculated_max_steps
        print(f"  - Auto-calculated max_steps: {calculated_max_steps}")
    else:
        print(f"  - Using configured max_steps: {training_args.max_steps}")

    # Update eval and save steps if they're still fractional
    if training_args.eval_steps and training_args.eval_steps < 1:
        training_args.eval_steps = max(1, int(training_args.eval_steps * training_args.max_steps))
    if training_args.save_steps and training_args.save_steps < 1:
        training_args.save_steps = max(1, int(training_args.save_steps * training_args.max_steps))
    if training_args.warmup_steps < 1 and training_args.warmup_steps > 0:
        training_args.warmup_steps = max(
            1, int(training_args.warmup_steps * training_args.max_steps)
        )

    # Load validation dataset if provided
    eval_dataset = None
    if training_args.eval_data_path and os.path.exists(training_args.eval_data_path):
        print(f"Loading validation dataset from {training_args.eval_data_path}...")
        eval_dataset_info = {
            f"{training_args.data_name}_eval": {
                "meta_file": training_args.eval_data_path,
                "storage_type": "hybrid",
                "data_format": "conversation",
                "image_dir": training_args.image_folder,
            }
        }
        eval_dataset = ConversationDataset(
            name=f"{training_args.data_name}_eval",
            info=eval_dataset_info[f"{training_args.data_name}_eval"],
            model=model,
            training_args=training_args,
        )
        print(f"Validation dataset loaded: {len(eval_dataset)} samples")
    elif training_args.eval_data_path:
        print(f"Warning: Validation dataset {training_args.eval_data_path} not found. Skipping it.")

    # Use original data collator (SFTTrainer can work with custom data collators)
    data_collator = DataCollatorForMultimodalDataset(model.text_tokenizer)

    # Initialize SFTTrainer with correct parameters
    print("Initializing SFTTrainer...")

    # Add early stopping callback if eval dataset is provided
    callbacks = []
    if eval_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = SFTTrainer(
        model=model,
        args=training_args,  # SFTConfig containing all training parameters
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        # Note: tokenizer, packing, max_seq_length are handled by the SFTConfig
        # processing_class can be added if needed for multimodal processing
    )

    # Training
    print("Starting SFT training...")
    print(f"Training with early stopping: {'enabled' if eval_dataset else 'disabled'}")
    # Check for existing checkpoints and resume automatically
    checkpoint_dir = pathlib.Path(training_args.output_dir)
    existing_checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if existing_checkpoints:
        # Find the latest checkpoint
        latest_checkpoint = max(existing_checkpoints, key=lambda x: int(x.name.split("-")[1]))
        print(f"Found existing checkpoint: {latest_checkpoint}")
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=str(latest_checkpoint))
    else:
        print("No existing checkpoints found. Starting training from scratch.")
        trainer.train()

    # Save model
    print("Saving model...")
    trainer.save_model()
    trainer.tokenizer.save_pretrained(training_args.output_dir)
    if hasattr(model, "visual_tokenizer") and hasattr(model.visual_tokenizer, "image_processor"):
        model.visual_tokenizer.image_processor.save_pretrained(training_args.output_dir)
    trainer.save_state()

    print("SFT training completed!")


if __name__ == "__main__":
    main()
