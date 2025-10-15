"""
Ovis2.5 GRPO Training for Tool-Calling (Crop + Identify)

Based on:
- VLM-R� R-GRPO (Region-Conditioned Reinforcement Policy Optimization)
- TRL GRPOTrainer best practices
- Tool-calling reward design for crop and identify tools

Features:
- Multi-reward system (tool usage correctness, bbox validity, reasoning quality)
- Proper training-inference alignment with tool masking
- Memory-efficient with gradient checkpointing support
- Support for both full fine-tuning and LoRA
"""

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from ovis.model.modeling_ovis2_5 import Ovis2_5
from PIL import Image
from transformers import HfArgumentParser, set_seed

from Ovis.src_theo.grpo.ovis_grpo_trainer import create_ovis_grpo_trainer
from Ovis.src_theo.tools.tool_base import ToolRegistry

# Try to import TRL GRPO
try:
    from trl import GRPOConfig, GRPOTrainer

    HAS_TRL_GRPO = True
except ImportError:
    logging.warning("TRL GRPOTrainer not available - install with: pip install trl>=0.13.0")
    HAS_TRL_GRPO = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class ModelArguments:
    """Model configuration"""

    model_path: str = field(default="AIDC-AI/Ovis2.5-9B")
    sft_model_path: Optional[str] = field(
        default=None, metadata={"help": "Path to SFT checkpoint (LoRA or full model)"}
    )
    visual_vocab_size: int = field(default=65536)
    use_flash_attention: bool = field(default=True)


@dataclass
class DataArguments:
    """Data configuration"""

    train_data_path: str = field(
        default="/mnt/nas1/data/coco/refcoco_vlm_results_theo_ready_to_train/refcoco_toolcall_merged_train.json"
    )
    eval_data_path: Optional[str] = field(default=None)
    image_folder: str = field(default="/mnt/nas1/data/")

    # Pixel settings
    single_image_min_pixels: int = field(default=200704)
    single_image_max_pixels: int = field(default=3211264)
    max_prompt_length: int = field(default=2048)


@dataclass
class GRPOTrainingArguments:
    """GRPO-specific training arguments"""

    # Basic training
    output_dir: str = field(default="./Ovis/checkpoints/ovis25_grpo_toolcall")
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=1e-5)

    # GRPO specific
    num_generations: int = field(
        default=4, metadata={"help": "Number of completions to generate per prompt"}
    )
    max_completion_length: int = field(
        default=3072, metadata={"help": "Maximum length for generated completions"}
    )

    # Reward weights
    tool_usage_weight: float = field(
        default=0.4, metadata={"help": "Weight for tool usage correctness reward"}
    )
    bbox_validity_weight: float = field(
        default=0.3, metadata={"help": "Weight for bounding box validity reward"}
    )
    reasoning_quality_weight: float = field(
        default=0.3, metadata={"help": "Weight for reasoning quality reward"}
    )

    # KL penalty
    beta: float = field(default=0.01, metadata={"help": "KL penalty coefficient"})

    # Generation parameters
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.9)
    enable_thinking: bool = field(default=True)
    thinking_budget: int = field(default=2048)

    # Logging and saving
    logging_steps: int = field(default=10)
    save_steps: int = field(default=100)
    save_total_limit: int = field(default=3)

    # Mixed precision
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)

    # Other
    seed: int = field(default=42)
    dataloader_num_workers: int = field(default=4)


# ============================================================================
# Reward Model
# ============================================================================


class ToolCallingRewardModel:
    """
    Reward model for tool-calling evaluation

    Computes rewards based on:
    1. Tool usage correctness (crop/identify calls with proper format)
    2. Bounding box validity (coordinates in valid range)
    3. Reasoning quality (proper <think>/<answer> structure)
    """

    def __init__(self, config: GRPOTrainingArguments):
        self.config = config
        self.tool_registry = ToolRegistry()

        # Reward weights
        self.tool_usage_weight = config.tool_usage_weight
        self.bbox_validity_weight = config.bbox_validity_weight
        self.reasoning_quality_weight = config.reasoning_quality_weight

    def compute_rewards(
        self, completions: List[str], prompts: Optional[List[str]] = None
    ) -> List[float]:
        """Compute rewards for a batch of completions"""
        rewards = []

        for completion in completions:
            # Compute individual reward components
            tool_usage_reward = self._compute_tool_usage_reward(completion)
            bbox_validity_reward = self._compute_bbox_validity_reward(completion)
            reasoning_quality_reward = self._compute_reasoning_quality_reward(completion)

            # Weighted combination
            total_reward = (
                self.tool_usage_weight * tool_usage_reward
                + self.bbox_validity_weight * bbox_validity_reward
                + self.reasoning_quality_weight * reasoning_quality_reward
            )

            rewards.append(total_reward)

        return rewards

    def _compute_tool_usage_reward(self, text: str) -> float:
        """
        Reward for correct tool usage

        Checks:
        - Presence of tool calls with correct format
        - Tool calls have corresponding responses (for identify)
        - Tool calls are in appropriate context (within <think> tags)
        """
        score = 0.0

        # Check for tool calls
        crop_pattern = r"<tool_call>Crop \[[\d.,\s]+\]</tool_call>"
        identify_pattern = r"<tool_call>Identify \[[\d.,\s]+\]</tool_call>"

        crop_calls = re.findall(crop_pattern, text)
        identify_calls = re.findall(identify_pattern, text)

        total_tools = len(crop_calls) + len(identify_calls)

        if total_tools == 0:
            # No tools used - check if question requires tools
            if self._requires_tools(text):
                return 0.0  # Should have used tools
            else:
                return 0.5  # Acceptable for simple questions

        # Reward for using tools
        score += 0.3

        # Check identify tool responses
        identify_response_pattern = (
            r"<tool_call>Identify \[[\d.,\s]+\]</tool_call><tool_response>[^<]+</tool_response>"
        )
        identify_with_response = re.findall(identify_response_pattern, text)

        if len(identify_calls) > 0:
            response_ratio = len(identify_with_response) / len(identify_calls)
            score += 0.4 * response_ratio  # Reward for proper tool response format

        # Check if tools are used within <think> tags
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            think_content = think_match.group(1)
            tools_in_think = len(re.findall(crop_pattern, think_content)) + len(
                re.findall(identify_pattern, think_content)
            )
            if tools_in_think == total_tools:
                score += 0.3  # All tools in proper context

        return min(1.0, score)

    def _compute_bbox_validity_reward(self, text: str) -> float:
        """
        Reward for valid bounding boxes

        Checks:
        - Coordinates are numeric
        - Coordinates in valid range (normalized [0,1] or pixel values)
        - x1 < x2, y1 < y2
        """
        score = 1.0  # Start with perfect score

        # Extract all bounding boxes
        bbox_pattern = r"\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]"
        bboxes = re.findall(bbox_pattern, text)

        if not bboxes:
            return 1.0  # No boxes to validate

        invalid_count = 0
        for bbox_str in bboxes:
            try:
                x1, y1, x2, y2 = [float(x) for x in bbox_str]

                # Check validity
                # Accept both normalized [0,1] and pixel coordinates (large values)
                if x1 >= x2 or y1 >= y2:
                    invalid_count += 1
                elif x1 < 0 or y1 < 0:
                    invalid_count += 1
                # If normalized coordinates
                elif max(x1, y1, x2, y2) <= 1.0:
                    if x2 > 1.0 or y2 > 1.0:
                        invalid_count += 1

            except ValueError:
                invalid_count += 1

        # Penalize invalid boxes
        if len(bboxes) > 0:
            score = 1.0 - (invalid_count / len(bboxes))

        return max(0.0, score)

    def _compute_reasoning_quality_reward(self, text: str) -> float:
        """
        Reward for reasoning quality

        Checks:
        - Proper structure with <think> and <answer> tags
        - Reasoning before tool use
        - Integration of tool results
        - Answer coherence
        """
        score = 0.0

        # Check for proper structure
        has_think = "<think>" in text and "</think>" in text
        has_answer = "<answer>" in text and "</answer>" in text

        if has_think:
            score += 0.3
        if has_answer:
            score += 0.3

        # Check reasoning flow (text before tool call)
        if has_think:
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            if think_match:
                think_content = think_match.group(1)

                # Look for reasoning before first tool call
                tool_call_match = re.search(r"<tool_call>", think_content)
                if tool_call_match:
                    reasoning_before_tool = think_content[: tool_call_match.start()]
                    # Reward substantial reasoning before tools
                    if len(reasoning_before_tool.split()) > 10:
                        score += 0.2

        # Check length (penalize too short or too long)
        word_count = len(text.split())
        if word_count < 20:
            score *= 0.5  # Too short
        elif word_count > 1000:
            score *= 0.8  # Too verbose

        # Check for integration of tool results (text after tool response)
        if "<tool_response>" in text:
            # Look for text analyzing tool response
            response_pattern = r"</tool_response>(.*?)(?:<tool_call>|</think>)"
            follow_ups = re.findall(response_pattern, text, re.DOTALL)
            if any(len(f.split()) > 5 for f in follow_ups):
                score += 0.2  # Reward for analyzing tool results

        return min(1.0, score)

    def _requires_tools(self, text: str) -> bool:
        """Heuristic to determine if question requires tools"""
        # Simple heuristic: check for keywords
        tool_keywords = [
            "who",
            "identify",
            "person",
            "people",
            "crop",
            "examine",
            "look at",
            "focus on",
            "region",
            "area",
            "section",
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in tool_keywords)


# ============================================================================
# Dataset
# ============================================================================


class ToolCallingGRPODataset(torch.utils.data.Dataset):
    """
    Dataset for GRPO training with tool-calling data

    Returns prompts only (no completions) for generation during training
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        model: Ovis2_5,
        max_prompt_length: int = 2048,
        min_pixels: int = 200704,
        max_pixels: int = 3211264,
    ):
        self.data_path = data_path
        self.image_folder = image_folder
        self.model = model
        self.max_prompt_length = max_prompt_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Load data
        with open(data_path, "r") as f:
            self.samples = json.load(f)

        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load image
        image_path = os.path.join(self.image_folder, sample["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return dummy data
            return {
                "prompt": sample["conversations"][1]["value"]
                if len(sample["conversations"]) > 1
                else "",
                "image": None,
                "query": "",
            }

        # Extract query (human message)
        query = ""
        for conv in sample["conversations"]:
            if conv["from"] == "human":
                query = conv["value"].replace("<image>\n", "").replace("<image>", "").strip()
                break

        # Prepare prompt for generation (system + query)
        prompt_messages = []
        for conv in sample["conversations"]:
            if conv["from"] == "system":
                prompt_messages.append(
                    {"role": "system", "content": [{"type": "text", "text": conv["value"]}]}
                )
            elif conv["from"] == "human":
                prompt_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": query},
                        ],
                    }
                )
                break  # Stop at first human message

        return {
            "prompt_messages": prompt_messages,
            "query": query,
            "image": image,
            "image_path": image_path,
        }


# ============================================================================
# GRPO Trainer
# ============================================================================


def create_grpo_trainer(
    model: Ovis2_5,
    train_dataset: ToolCallingGRPODataset,
    reward_model: ToolCallingRewardModel,
    args: GRPOTrainingArguments,
    data_args: DataArguments,
):
    """
    Create GRPO trainer with multimodal support

    Uses custom OvisGRPOTrainer that handles:
    - Multimodal generation (images + text)
    - Variable-sized tensors (Ovis2.5 requirement)
    - Tool-calling reward functions
    """

    if not HAS_TRL_GRPO:
        raise ImportError(
            "TRL with GRPO support is required. Install with: pip install trl>=0.13.0"
        )

    # Use custom trainer factory (handles multimodal generation)
    trainer = create_ovis_grpo_trainer(
        model=model,
        train_dataset=train_dataset,
        reward_model=reward_model,
        args=args,
        data_args=data_args,
    )

    return trainer


# ============================================================================
# Main
# ============================================================================


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("=" * 80)
    logger.info("Ovis2.5 GRPO Training for Tool-Calling")
    logger.info("=" * 80)
    logger.info(f"Model: {model_args.model_path}")
    logger.info(f"SFT checkpoint: {model_args.sft_model_path}")
    logger.info(f"Train data: {data_args.train_data_path}")
    logger.info(f"Output dir: {training_args.output_dir}")
    logger.info(f"Num generations: {training_args.num_generations}")
    logger.info(
        f"Reward weights: tool={training_args.tool_usage_weight}, "
        f"bbox={training_args.bbox_validity_weight}, "
        f"reasoning={training_args.reasoning_quality_weight}"
    )
    logger.info("=" * 80)

    # Load model
    logger.info("Loading model...")

    # Determine model path (SFT checkpoint or base model)
    load_path = model_args.sft_model_path if model_args.sft_model_path else model_args.model_path

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if training_args.bf16 else torch.float16,
    }

    if model_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = Ovis2_5.from_pretrained(load_path, **model_kwargs)
        logger.info(f"Model loaded from {load_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Prepare dataset
    logger.info("Preparing dataset...")
    train_dataset = ToolCallingGRPODataset(
        data_path=data_args.train_data_path,
        image_folder=data_args.image_folder,
        model=model,
        max_prompt_length=data_args.max_prompt_length,
        min_pixels=data_args.single_image_min_pixels,
        max_pixels=data_args.single_image_max_pixels,
    )

    # Create reward model
    logger.info("Creating reward model...")
    reward_model = ToolCallingRewardModel(training_args)

    # Create trainer
    logger.info("Creating GRPO trainer...")
    trainer = create_grpo_trainer(
        model=model,
        train_dataset=train_dataset,
        reward_model=reward_model,
        args=training_args,
        data_args=data_args,
    )

    # Train
    logger.info("Starting GRPO training...")
    trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
