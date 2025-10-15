"""
Custom multimodal generation wrapper for Ovis2.5 GRPO training

Handles:
- Image preprocessing with model.preprocess_inputs()
- Variable-sized tensors (batch_size=1 requirement)
- Tool execution during generation (if needed)
"""

import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class OvisMultimodalGenerator:
    """
    Wrapper for generating completions from Ovis2.5 with proper multimodal handling
    """

    def __init__(
        self,
        model,
        max_new_tokens: int = 3072,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

    def generate_batch(
        self,
        prompt_messages_batch: List[List[Dict]],
        num_generations: int = 4,
    ) -> List[List[str]]:
        """
        Generate multiple completions for a batch of prompts

        Args:
            prompt_messages_batch: List of prompt message lists (Ovis format)
            num_generations: Number of completions per prompt

        Returns:
            List of lists of generated texts (batch_size x num_generations)
        """
        all_completions = []

        # Ovis2.5 requires batch_size=1 due to variable-sized tensors
        for prompt_messages in prompt_messages_batch:
            completions = self.generate_single_prompt(
                prompt_messages, num_generations=num_generations
            )
            all_completions.append(completions)

        return all_completions

    def generate_single_prompt(
        self,
        prompt_messages: List[Dict],
        num_generations: int = 4,
    ) -> List[str]:
        """
        Generate multiple completions for a single prompt

        Args:
            prompt_messages: Ovis-format messages [{'role': 'user', 'content': [...]}]
            num_generations: Number of completions to generate

        Returns:
            List of generated completion texts
        """
        completions = []

        try:
            # Preprocess once (images → visual tokens)
            inputs = self.model.preprocess_inputs(prompt_messages)

            # Generate multiple completions
            for i in range(num_generations):
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature if i > 0 else 1e-8,  # First is greedy
                        top_p=self.top_p,
                        do_sample=self.do_sample and i > 0,
                        pad_token_id=self.model.text_tokenizer.eos_token_id,
                    )

                # Decode only the generated part (exclude prompt)
                prompt_length = inputs["input_ids"].shape[1]
                generated_ids = output_ids[0, prompt_length:]
                completion_text = self.model.text_tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=False,  # Keep <tool_call> tags
                    clean_up_tokenization_spaces=True,
                )

                completions.append(completion_text)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return empty completions on error
            completions = [""] * num_generations

        return completions

    def generate_with_tool_execution(
        self,
        prompt_messages: List[Dict],
        num_generations: int = 4,
        tool_registry=None,
    ) -> List[str]:
        """
        Generate with real-time tool execution (for inference simulation during GRPO)

        This allows GRPO to train on realistic tool-calling scenarios where:
        - Model generates <tool_call>...</tool_call>
        - System executes tool
        - Model continues with tool result

        Args:
            prompt_messages: Ovis-format messages
            num_generations: Number of completions
            tool_registry: ToolRegistry instance for tool execution

        Returns:
            List of completions with executed tool results
        """
        if tool_registry is None:
            # Fallback to basic generation
            return self.generate_single_prompt(prompt_messages, num_generations)

        completions = []

        # TODO: Implement streaming generation with tool execution
        # This is advanced functionality for Phase 2 GRPO
        # For now, use basic generation (tools are in training data, not executed)

        logger.warning(
            "Tool execution during GRPO generation not yet implemented. "
            "Using standard generation (tools must be in training data)."
        )

        return self.generate_single_prompt(prompt_messages, num_generations)


def collate_multimodal_grpo_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for GRPO with multimodal data

    Args:
        batch: List of samples from ToolCallingGRPODataset

    Returns:
        Collated batch suitable for OvisMultimodalGenerator
    """
    prompt_messages_batch = [sample["prompt_messages"] for sample in batch]
    queries = [sample["query"] for sample in batch]
    image_paths = [sample["image_path"] for sample in batch]

    return {
        "prompt_messages_batch": prompt_messages_batch,
        "queries": queries,
        "image_paths": image_paths,
    }
