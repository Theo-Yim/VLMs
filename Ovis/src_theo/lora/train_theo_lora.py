"""
Ovis2.5 LoRA Fine-tuning Script using TRL SFTTrainer
Based on original Ovis training framework with PEFT LoRA integration
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

# Import custom Ovis2.5 implementation
from ovis.model.modeling_ovis2_5 import Ovis2_5

# Import original Ovis training components
from ovis.train.dataset.conversation_dataset import ConversationDataset
from ovis.train.dataset.conversation_bbox_dataset import ConversationBboxDataset
from ovis.train.dataset.multimodal_dataset import DataCollatorForMultimodalDataset
from transformers import (
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
)

# Import TRL and PEFT components
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType

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
class LoRAArguments:
    """Arguments for LoRA configuration"""
    # LoRA hyperparameters
    lora_r: int = field(default=32, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    
    # Target modules - focusing on LLM attention and MLP layers
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )
    
    # Whether to apply LoRA to visual components
    apply_lora_to_vision: bool = field(
        default=False,
        metadata={"help": "Whether to apply LoRA to vision encoder"}
    )


@dataclass
class CustomSFTConfig(SFTConfig):
    """Extended SFT configuration for Ovis2.5 LoRA training"""
    
    # Data paths
    data_path: str = field(default="./data/train_data.json")
    eval_data_path: Optional[str] = field(default=None)
    image_folder: str = field(default="./data/images")
    data_name: str = field(default="test_dataset")
    data_type: str = field(default="conversation")
    
    # Ovis-specific parameters
    ovis_pretrained_path: Optional[str] = field(default="AIDC-AI/Ovis2.5-9B")
    stage: Optional[int] = field(default=3)
    
    # Multimodal-specific parameters
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
    
    # Override defaults for LoRA training
    learning_rate: float = field(default=1e-4)  # Higher LR for LoRA
    
    # SFT-specific parameters
    packing: bool = field(default=False)
    max_seq_length: Optional[int] = field(default=8192)
    dataset_text_field: Optional[str] = field(default=None)
    formatting_func: Optional[str] = field(default=None)
    
    def __post_init__(self):
        super().__post_init__()
        # Sync max_seq_length with multimodal_max_length
        if self.max_seq_length != self.multimodal_max_length:
            self.max_seq_length = self.multimodal_max_length
        
        # For custom multimodal datasets
        self.remove_unused_columns = False
        self.dataset_kwargs = {"skip_prepare_dataset": True}


def create_lora_config(lora_args: LoRAArguments) -> LoraConfig:
    """Create LoRA configuration for Ovis2.5"""
    
    # Parse target modules
    target_modules = [module.strip() for module in lora_args.lora_target_modules.split(",")]
    
    # Create LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",  # Don't apply LoRA to bias terms
    )
    
    return lora_config


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


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model"""
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"All parameters: {all_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / all_params:.4f}%")


def main():
    """Main LoRA training function using TRL SFTTrainer"""
    
    config_file = (
        os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "./Ovis/src_theo/lora/train_config_lora.json"
    )
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, LoRAArguments, CustomSFTConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, lora_args, training_args = parser.parse_json_file(
            json_file=config_file, allow_extra_keys=True
        )
    else:
        model_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    set_seed(training_args.seed)
    
    # Load model
    print(f"Loading model from {model_args.model_path}...")
    
    # Handle device mapping for distributed training
    device_map = (
        f"cuda:{training_args.local_rank}" if training_args.local_rank != -1 else "cuda:0"
    )
    print(f"Using device map: {device_map}")
    
    model = Ovis2_5.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    )
    
    # Create LoRA configuration
    print("Setting up LoRA configuration...")
    peft_config = create_lora_config(lora_args)
    print(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}, dropout={peft_config.lora_dropout}")
    print(f"Target modules: {peft_config.target_modules}")
    
    # Prepare tokenizer
    tokenizer = model.text_tokenizer
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset info structure for original ConversationDataset
    dataset_info = create_dataset_info(training_args)
    
    # Create dataset using original ConversationDataset
    print(f"Loading training dataset from {training_args.data_path}...")
    train_dataset = ConversationBboxDataset(
        name=training_args.data_name,
        info=dataset_info[training_args.data_name],
        model=model,
        training_args=training_args,
    )
    
    print(f"Training dataset loaded: {len(train_dataset)} samples")
    
    # Calculate max_steps to avoid dataloader length issues
    dataset_size = len(train_dataset)
    effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    steps_per_epoch = dataset_size // effective_batch_size
    if dataset_size % effective_batch_size != 0:
        steps_per_epoch += 1  # Round up for partial batches
    
    print(f"Dataset analysis:")
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
        training_args.warmup_steps = max(1, int(training_args.warmup_steps * training_args.max_steps))
    
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
        eval_dataset = ConversationBboxDataset(
            name=f"{training_args.data_name}_eval",
            info=eval_dataset_info[f"{training_args.data_name}_eval"],
            model=model,
            training_args=training_args,
        )
        print(f"Validation dataset loaded: {len(eval_dataset)} samples")
    elif training_args.eval_data_path:
        print(f"Warning: Validation dataset path {training_args.eval_data_path} not found. Skipping validation.")
    
    # Use original data collator
    data_collator = DataCollatorForMultimodalDataset(model.text_tokenizer)
    
    # Initialize SFTTrainer with LoRA
    print("Initializing SFTTrainer with LoRA...")
    
    # Add early stopping callback if eval dataset is provided
    callbacks = []
    if eval_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        peft_config=peft_config,  # This enables LoRA training
        callbacks=callbacks,
    )
    
    # Print trainable parameters after PEFT setup
    print_trainable_parameters(trainer.model)
    
    # Training
    print("Starting LoRA training...")
    print(f"Training with early stopping: {'enabled' if eval_dataset else 'disabled'}")
    trainer.train()
    
    # Save model and adapters
    print("Saving LoRA adapters...")
    trainer.save_model()
    trainer.save_state()
    
    # Optionally save the merged model (requires more memory)
    if training_args.output_dir:
        print("Note: To merge LoRA adapters with base model, run the merge script separately.")
        print(f"LoRA adapters saved to: {training_args.output_dir}")
    
    print("LoRA training completed!")


if __name__ == "__main__":
    main()