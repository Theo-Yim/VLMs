"""
Configuration for Ovis2.5-9B training
Based on official guide: https://huggingface.co/AIDC-AI/Ovis2.5-9B
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration for Ovis2.5-9B"""

    model_name: str = "AIDC-AI/Ovis2.5-9B"
    use_flash_attention: bool = True

    # 🎯 CRITICAL FIX: Disable gradient checkpointing to prevent in-place operation errors
    gradient_checkpointing: bool = False  # DISABLED - causes in-place operation issues with Ovis

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Keep custom tokens for your crop strategy
    use_custom_tool_tokens: bool = True
    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"
    tool_call_start_token_id: Optional[int] = None
    tool_call_end_token_id: Optional[int] = None

    use_native_grounding: bool = False


@dataclass
class DataConfig:
    """Data configuration for Ovis2.5-9B training"""

    train_data_path: str = "../data/refcoco_qa_pairs_train.jsonl"
    val_data_path: str = "../data/refcoco_qa_pairs_val.jsonl"
    image_base_path: str = "/mnt/nas3/Data/"

    max_pixels: int = 896 * 896
    max_length: int = 2048

    # Batch sizes MUST be 1 for native resolution
    train_batch_size: int = 1
    eval_batch_size: int = 1
    num_workers: int = 0


@dataclass
class SFTTrainingConfig:
    """Stage 1: Supervised Fine-Tuning configuration for Ovis2.5-9B"""

    output_dir: str = "outputs/sft"

    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Higher gradient accumulation for batch_size=1
    gradient_accumulation_steps: int = 16

    fp16: bool = False
    bf16: bool = True

    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    optim: str = "adamw_torch"
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class GRPOTrainingConfig:
    """Stage 2: Regional GRPO configuration for Ovis2.5-9B"""

    output_dir: str = "outputs/grpo"
    sft_checkpoint_path: str = "outputs/sft/final_model"

    beta: float = 0.1
    reward_model_path: Optional[str] = None

    use_regional_rewards: bool = True
    region_reward_weight: float = 0.5
    global_reward_weight: float = 0.5

    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    warmup_ratio: float = 0.1

    ppo_epochs: int = 4
    chunk_size: int = 128
    mini_batch_size: int = 1

    generation_kwargs: dict = field(
        default_factory=lambda: {
            "max_new_tokens": 3072,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "enable_thinking": True,
            "enable_thinking_budget": True,
            "thinking_budget": 2048,
        }
    )

    gradient_accumulation_steps: int = 8

    fp16: bool = False
    bf16: bool = True

    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3

    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])


@dataclass
class InferenceConfig:
    """Inference configuration for Ovis2.5-9B"""

    model_path: str = "outputs/grpo/final_model"

    max_new_tokens: int = 3072
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    enable_thinking: bool = True
    enable_thinking_budget: bool = True
    thinking_budget: int = 2048

    max_pixels: int = 896 * 896

    device: str = "cuda"

    parse_tool_calls: bool = True
    extract_think_answer: bool = True
    parse_grounding: bool = True
