"""
InternVL 3.5 LoRA Fine-tuning Script with PEFT LoRA
Complete implementation with thinking mode support and multimodal capabilities
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List

import torch

# Dataset handling
from internvl_dataset import MultimodalDataCollator, create_internvl_dataset
from peft import LoraConfig, TaskType

# Core transformers and training
from transformers import (
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# =============================================================================
# Training Arguments
# =============================================================================


@dataclass
class ModelArguments:
    """Model configuration arguments."""

    model_name_or_path: str = field(default="OpenGVLab/InternVL3_5-8B")
    use_flash_attn: bool = field(default=True)


@dataclass
class LoRAArguments:
    """LoRA configuration arguments."""

    lora_enable: bool = field(default=True)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "mlp1.1",
            "mlp1.3",
            "qkv",
            "fc1",
            "fc2",
        ]
    )


@dataclass
class DataArguments:
    """Data configuration arguments."""

    data_path: str = field(default="./data/train_data.json")
    eval_data_path: str = field(default=None)
    image_folder: str = field(default="./data/images")
    image_size: int = field(default=448)
    max_dynamic_patches: int = field(default=12)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extended TrainingArguments with early stopping support."""

    early_stopping_patience: int = field(
        default=0,
        metadata={
            "help": "Number of evaluations with no improvement before stopping (0 = disabled)"
        },
    )


# =============================================================================
# Custom Callback
# =============================================================================


class EvalLossCallback(TrainerCallback):
    """Callback to log evaluation loss to a text file"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "eval_loss.txt")
        # Create output directory if needed, but don't clear the file to preserve history
        os.makedirs(output_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics and (not args.local_rank or args.local_rank <= 0):
            with open(self.log_file, "a") as f:
                f.write(f"iteration {state.global_step}:\n")
                f.write(f"{metrics}\n\n")
            print(f"\n{'=' * 60}")
            print(f"Evaluation at step {state.global_step}:")
            if "eval_loss" in metrics:
                print(f"  eval_loss: {metrics['eval_loss']:.6f}")
            if "eval_runtime" in metrics:
                print(f"  eval_runtime: {metrics['eval_runtime']:.2f}s")
            if "eval_samples_per_second" in metrics:
                print(f"  eval_samples_per_second: {metrics['eval_samples_per_second']:.3f}")
            print(f"{'=' * 60}\n")

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training to log final training metrics"""
        if state.log_history and (not args.local_rank or args.local_rank <= 0):
            # Find the last training metrics
            final_metrics = {}
            for log_entry in reversed(state.log_history):
                if "train_runtime" in log_entry:
                    final_metrics = {
                        k: v
                        for k, v in log_entry.items()
                        if k
                        in [
                            "train_runtime",
                            "train_samples_per_second",
                            "train_steps_per_second",
                            "train_loss",
                            "epoch",
                        ]
                    }
                    break

            if final_metrics:
                with open(self.log_file, "a") as f:
                    f.write(f"{final_metrics}\n")
                print(f"\n{'=' * 60}")
                print("Training completed!")
                if "train_loss" in final_metrics:
                    print(f"  train_loss: {final_metrics['train_loss']:.6f}")
                if "train_runtime" in final_metrics:
                    print(f"  train_runtime: {final_metrics['train_runtime']:.2f}s")
                print(f"{'=' * 60}\n")


# =============================================================================
# Custom Trainer
# =============================================================================


class InternVLTrainer(Trainer):
    """
    Custom trainer for InternVL.
    Explicitly passes only the arguments InternVL's forward accepts.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for InternVL.
        Ensures image_flags are present.
        """
        # Add image_flags if not present
        if "image_flags" not in inputs:
            batch_size = inputs["input_ids"].shape[0]
            inputs["image_flags"] = torch.ones(
                batch_size, 1, dtype=torch.long, device=inputs["input_ids"].device
            )

        # Call model - InternVL's forward now has **kwargs to catch extras
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# Model Setup Functions
# =============================================================================


def load_model_and_tokenizer(model_args: ModelArguments, training_args, is_main_process):
    """Load InternVL 3.5 model and tokenizer."""
    if is_main_process:
        print(f"Loading model from {model_args.model_name_or_path}")

    # Handle device mapping for distributed training
    device_map = f"cuda:{training_args.local_rank}" if training_args.local_rank != -1 else "cuda:0"
    if is_main_process:
        print(f"Using device map: {device_map}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True, use_fast=False
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=model_args.use_flash_attn,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    # Set img_context_token_id (InternVL needs this for image placeholder)
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if is_main_process:
        print(f"✓ Set img_context_token_id = {model.img_context_token_id} for '{IMG_CONTEXT_TOKEN}'")

    # Patch the forward method to accept **kwargs (fixes inputs_embeds issue)
    original_forward = model.forward

    def forward_with_kwargs(self, *args, **kwargs):
        kwargs.pop("inputs_embeds", None)
        return original_forward(*args, **kwargs)

    import types

    model.forward = types.MethodType(forward_with_kwargs, model)
    if is_main_process:
        print("✓ Patched model.forward() to accept **kwargs")

    return model, tokenizer


def create_dataset(data_args: DataArguments, tokenizer):
    """Create training dataset."""
    return create_internvl_dataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        tokenizer=tokenizer,
        image_size=data_args.image_size,
        max_dynamic_patches=data_args.max_dynamic_patches,
    )


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"All parameters: {all_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / all_params:.4f}%")


# =============================================================================
# Main Training Function
# =============================================================================


def train():
    """Main training function."""
    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, LoRAArguments, CustomTrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from JSON config file
        model_args, data_args, lora_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # Set random seed
    set_seed(training_args.seed if hasattr(training_args, "seed") else 42)

    # Determine if this is the main process (to avoid duplicate logs)
    is_main_process = training_args.local_rank in [-1, 0]

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, training_args, is_main_process)

    # Apply LoRA manually
    if lora_args.lora_enable:
        from peft import get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if is_main_process:
            print(f"LoRA applied: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
            print_trainable_parameters(model)

    # Create datasets
    train_dataset = create_dataset(data_args, tokenizer)

    # Create eval dataset if eval_data_path is provided
    eval_dataset = None
    if data_args.eval_data_path is not None:
        if is_main_process:
            print(f"Loading evaluation dataset from: {data_args.eval_data_path}")
        eval_data_args = DataArguments(
            data_path=data_args.eval_data_path,
            eval_data_path=None,
            image_folder=data_args.image_folder,
            image_size=data_args.image_size,
            max_dynamic_patches=data_args.max_dynamic_patches,
        )
        eval_dataset = create_dataset(eval_data_args, tokenizer)
        if is_main_process:
            print(f"Evaluation dataset size: {len(eval_dataset)}")

    # Calculate max_steps to avoid dataloader length issues (from Ovis implementation)
    dataset_size = len(train_dataset)

    # Calculate effective batch size accounting for DDP (multiple GPUs)
    # In DDP, each GPU processes per_device_train_batch_size samples
    # With gradient_accumulation_steps, that's per_device * grad_accum per GPU
    # With world_size GPUs, total = per_device * grad_accum * world_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    effective_batch_size_per_gpu = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )
    effective_batch_size_total = effective_batch_size_per_gpu * world_size

    steps_per_epoch = dataset_size // effective_batch_size_total
    if dataset_size % effective_batch_size_total != 0:
        steps_per_epoch += 1  # Round up for partial batches

    if is_main_process:
        print("Dataset analysis:")
        print(f"  - Dataset size: {dataset_size}")
        print(f"  - Per-device batch size: {training_args.per_device_train_batch_size}")
        print(f"  - Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print(f"  - Number of GPUs (world_size): {world_size}")
        print(f"  - Effective batch size per GPU: {effective_batch_size_per_gpu}")
        print(f"  - Effective batch size (total): {effective_batch_size_total}")
        print(f"  - Steps per epoch: {steps_per_epoch}")

    # If max_steps is not set or is -1, calculate it
    if training_args.max_steps <= 0:
        calculated_max_steps = steps_per_epoch * training_args.num_train_epochs
        training_args.max_steps = calculated_max_steps
        if is_main_process:
            print(f"  - Auto-calculated max_steps: {calculated_max_steps}")
    else:
        if is_main_process:
            print(f"  - Using configured max_steps: {training_args.max_steps}")

    # Update eval and save steps if they're still fractional
    if (
        hasattr(training_args, "eval_steps")
        and training_args.eval_steps
        and training_args.eval_steps < 1
    ):
        training_args.eval_steps = max(1, int(training_args.eval_steps * training_args.max_steps))
    if (
        hasattr(training_args, "save_steps")
        and training_args.save_steps
        and training_args.save_steps < 1
    ):
        training_args.save_steps = max(1, int(training_args.save_steps * training_args.max_steps))
    if (
        hasattr(training_args, "warmup_steps")
        and training_args.warmup_steps < 1
        and training_args.warmup_steps > 0
    ):
        training_args.warmup_steps = max(
            1, int(training_args.warmup_steps * training_args.max_steps)
        )

    # Setup data collator
    data_collator = MultimodalDataCollator(tokenizer)

    # Configure training arguments
    training_args.remove_unused_columns = False  # Keep pixel_values and image_flags

    # Handle gradient checkpointing configuration
    if training_args.gradient_checkpointing:
        # Check if DeepSpeed is being used
        using_deepspeed = (
            hasattr(training_args, "deepspeed") and training_args.deepspeed is not None
        )

        if using_deepspeed:
            # DeepSpeed handles gradient checkpointing well
            if is_main_process:
                print("✓ Using DeepSpeed with gradient checkpointing (recommended)")
        else:
            # Native DDP + gradient checkpointing + LoRA is problematic
            if is_main_process:
                print("⚠️  WARNING: Gradient checkpointing with native DDP may cause errors.")
                print("   Consider using DeepSpeed ZeRO-2 or ZeRO-3 instead.")
            # Try to enable anyway for non-DeepSpeed case
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            training_args.ddp_find_unused_parameters = True
            training_args.ddp_broadcast_buffers = False

    # Create InternVLTrainer (minimal wrapper handling inputs_embeds issue)
    trainer = InternVLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Add eval loss logging callback
    trainer.add_callback(EvalLossCallback(training_args.output_dir))

    # Add callback for early stopping if specified
    if eval_dataset and training_args.early_stopping_patience > 0:
        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)
        )
        if is_main_process:
            print(f"Early stopping enabled with patience={training_args.early_stopping_patience}")

    # Start training
    if is_main_process:
        print("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Run final evaluation if eval dataset exists
    if eval_dataset and (training_args.local_rank <= 0):
        if is_main_process:
            print("\nRunning final evaluation...")
        final_metrics = trainer.evaluate()
        if is_main_process:
            print(f"Final evaluation metrics: {final_metrics}")

    # Save final model
    if is_main_process:
        print("Saving final model...")
    trainer.save_model()

    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)

    if is_main_process:
        print(f"Training completed! Model saved to {training_args.output_dir}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    train()
