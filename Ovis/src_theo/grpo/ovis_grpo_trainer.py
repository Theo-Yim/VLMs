"""
Custom GRPO Trainer for Ovis2.5 Multimodal Model

Extends TRL's GRPOTrainer with multimodal generation support
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel
from trl import GRPOConfig, GRPOTrainer

from Ovis.src_theo.grpo.ovis_grpo_generator import (
    OvisMultimodalGenerator,
    collate_multimodal_grpo_batch,
)

logger = logging.getLogger(__name__)


class OvisGRPOTrainer(GRPOTrainer):
    """
    GRPO Trainer adapted for Ovis2.5 multimodal generation

    Key differences from standard GRPOTrainer:
    - Uses custom OvisMultimodalGenerator for generation
    - Handles variable-sized image tensors (batch_size=1)
    - Supports tool-calling reward functions
    """

    def __init__(
        self,
        model: PreTrainedModel,
        config: GRPOConfig,
        processing_class,
        train_dataset: Dataset,
        reward_funcs: List[Callable],
        eval_dataset: Optional[Dataset] = None,
        **kwargs,
    ):
        """
        Initialize Ovis GRPO Trainer

        Args:
            model: Ovis2.5 model
            config: GRPO training configuration
            processing_class: Tokenizer (for compatibility, not used for images)
            train_dataset: ToolCallingGRPODataset
            reward_funcs: List of reward functions
            eval_dataset: Optional evaluation dataset
        """
        # Initialize parent GRPOTrainer
        super().__init__(
            model=model,
            config=config,
            processing_class=processing_class,
            train_dataset=train_dataset,
            reward_funcs=reward_funcs,
            eval_dataset=eval_dataset,
            **kwargs,
        )

        # Create custom multimodal generator
        self.multimodal_generator = OvisMultimodalGenerator(
            model=self.model,
            max_new_tokens=config.max_completion_length,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
        )

        logger.info("Initialized OvisGRPOTrainer with multimodal generation support")

    def _generate_completions(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Override generation method to use Ovis2.5's multimodal preprocessing

        Args:
            batch: Batch from dataloader (contains 'prompt_messages_batch')

        Returns:
            Dictionary with generated completions and metadata
        """
        prompt_messages_batch = batch["prompt_messages_batch"]

        # Generate completions using custom generator
        all_completions = self.multimodal_generator.generate_batch(
            prompt_messages_batch=prompt_messages_batch,
            num_generations=self.config.num_generations,
        )

        # Flatten completions for reward computation
        # Shape: (batch_size * num_generations,)
        flat_completions = [comp for completions in all_completions for comp in completions]

        return {
            "completions": flat_completions,
            "batch_size": len(prompt_messages_batch),
            "num_generations": self.config.num_generations,
        }

    def get_train_dataloader(self):
        """
        Override to use custom collate function for multimodal data
        """
        from torch.utils.data import DataLoader

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            collate_fn=collate_multimodal_grpo_batch,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True,
        )

    def compute_rewards(
        self,
        completions: List[str],
        prompts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute rewards for generated completions

        Args:
            completions: List of generated texts
            prompts: Optional list of prompts (for context-aware rewards)

        Returns:
            Tensor of rewards (shape: [num_completions])
        """
        all_rewards = []

        # Apply each reward function
        for reward_func in self.reward_funcs:
            rewards = reward_func(completions, prompts=prompts)
            all_rewards.append(torch.tensor(rewards, device=self.model.device))

        # Average rewards from all functions
        if len(all_rewards) > 1:
            rewards_tensor = torch.stack(all_rewards).mean(dim=0)
        else:
            rewards_tensor = all_rewards[0]

        return rewards_tensor


def create_ovis_grpo_trainer(
    model,
    train_dataset,
    reward_model,
    args,
    data_args,
    eval_dataset=None,
):
    """
    Factory function to create OvisGRPOTrainer

    Args:
        model: Ovis2.5 model
        train_dataset: ToolCallingGRPODataset
        reward_model: ToolCallingRewardModel instance
        args: GRPOTrainingArguments
        data_args: DataArguments
        eval_dataset: Optional evaluation dataset

    Returns:
        OvisGRPOTrainer instance
    """
    from trl import GRPOConfig

    # Create GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # GRPO specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=data_args.max_prompt_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        # Logging
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        # Mixed precision
        bf16=args.bf16,
        fp16=args.fp16,
        # Other
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
    )

    # Define reward function wrapper
    def reward_function(completions: List[str], **kwargs) -> List[float]:
        """Wrapper for reward model"""
        return reward_model.compute_rewards(completions)

    # Create trainer
    trainer = OvisGRPOTrainer(
        model=model,
        processing_class=model.text_tokenizer,
        config=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[reward_function],
        eval_dataset=eval_dataset,
    )

    return trainer
