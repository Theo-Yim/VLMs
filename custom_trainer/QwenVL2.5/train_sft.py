"""
Stage 1: Supervised Fine-Tuning (SFT) for Qwen 2.5 VL
"""

import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from config import DataConfig, ModelConfig, SFTTrainingConfig
from data_utils import QwenVLDataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenVLSFTTrainer:
    """Trainer for Qwen 2.5 VL Supervised Fine-Tuning"""

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

    def load_model_and_processor(self):
        """Load Qwen 2.5 VL model and processor"""
        logger.info(f"Loading model: {self.model_config.model_name}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=True,
        )

        # Add custom tokens
        new_tokens = [
            self.model_config.tool_call_start_token,
            self.model_config.tool_call_end_token,
        ]
        num_added_tokens = self.processor.tokenizer.add_tokens(new_tokens)
        logger.info(f"Added {num_added_tokens} new tokens")

        # Update token IDs in processor
        self.processor.tokenizer.additional_special_tokens_ids = [
            self.model_config.tool_call_start_token_id,
            self.model_config.tool_call_end_token_id,
        ]

        # Model loading configuration
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.training_config.bf16 else torch.float16,
            "device_map": "auto",
        }

        if self.model_config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_config.model_name, **model_kwargs
        )

        # Resize token embeddings
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

        # Enable gradient checkpointing if specified
        if self.model_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Apply LoRA if specified
        if self.training_config.use_lora:
            self.apply_lora()

        logger.info("Model and processor loaded successfully")

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
        self.train_dataset = QwenVLDataset(
            data_path=self.data_config.train_data_path,
            image_base_path=self.data_config.image_base_path,
            processor=self.processor,
            max_length=self.data_config.max_length,
            image_size=self.data_config.image_size,
            stage="sft",
        )

        # Validation dataset
        self.val_dataset = QwenVLDataset(
            data_path=self.data_config.val_data_path,
            image_base_path=self.data_config.image_base_path,
            processor=self.processor,
            max_length=self.data_config.max_length,
            image_size=self.data_config.image_size,
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
            dataloader_num_workers=self.data_config.num_workers,
            remove_unused_columns=False,
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
        logger.info("Starting SFT training")

        # Load model and processor
        self.load_model_and_processor()

        # Prepare datasets
        self.prepare_datasets()

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.processor.tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8,
        )

        # Training arguments
        training_args = self.get_training_arguments()

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
        self.processor.save_pretrained(f"{self.training_config.output_dir}/final_model")

        # Save training metrics
        with open(f"{self.training_config.output_dir}/train_results.txt", "w") as f:
            f.write(str(train_result))

        logger.info("SFT training completed")

        return trainer, train_result


def main():
    """Main training entry point"""

    # Initialize configurations
    model_config = ModelConfig()
    data_config = DataConfig()
    training_config = SFTTrainingConfig()

    # You can override configurations here or load from a config file
    # For example:
    # training_config.num_train_epochs = 5
    # data_config.train_batch_size = 8

    # Initialize trainer
    trainer = QwenVLSFTTrainer(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
