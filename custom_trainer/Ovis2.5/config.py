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
    gradient_checkpointing: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Token strategy for crop tool integration
    # Two options:
    # 1. Use custom <tool_call> tokens (consistent with Qwen data)
    # 2. Use native Ovis grounding format: <ref>, <box>, <point>

    # Option 1: Custom tokens (set to True to match Qwen implementation)
    use_custom_tool_tokens: bool = True
    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"
    tool_call_start_token_id: Optional[int] = None  # Set dynamically
    tool_call_end_token_id: Optional[int] = None  # Set dynamically

    # Option 2: Native Ovis grounding (set use_custom_tool_tokens=False)
    use_native_grounding: bool = False  # Enable to use <ref>, <box>, <point>


@dataclass
class DataConfig:
    """Data configuration for Ovis2.5-9B training"""

    train_data_path: str = "../data/refcoco_qa_pairs_train.jsonl"
    val_data_path: str = "../data/refcoco_qa_pairs_val.jsonl"
    image_base_path: str = "/mnt/nas3/Data/"

    # Ovis2.5 processes images at native resolution
    # max_pixels parameter controls resolution instead
    max_pixels: int = 896 * 896  # Default as per official guide
    max_length: int = 2048

    # Data loading
    train_batch_size: int = 4
    eval_batch_size: int = 8
    num_workers: int = 4


@dataclass
class SFTTrainingConfig:
    """Stage 1: Supervised Fine-Tuning configuration for Ovis2.5-9B"""

    output_dir: str = "outputs/sft"

    # Training hyperparameters
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Gradient accumulation
    gradient_accumulation_steps: int = 4

    # Mixed precision
    fp16: bool = False
    bf16: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # Optimization
    optim: str = "adamw_torch"
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Evaluation
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Others
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    # LoRA configuration - Updated for Ovis2.5-9B architecture
    use_lora: bool = False  # Disabled by default due to PEFT compatibility issues. TODO: check if we can use lora
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            # Language model attention modules
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # Language model MLP modules
            "gate_proj",
            "up_proj",
            "down_proj",
            # Optional: Vision components (uncomment if needed)
            # "fc1", "fc2", "out_proj",  # Vision transformer modules
        ]
    )


@dataclass
class GRPOTrainingConfig:
    """Stage 2: Regional GRPO configuration for Ovis2.5-9B"""

    output_dir: str = "outputs/grpo"

    # Load from SFT checkpoint
    sft_checkpoint_path: str = "outputs/sft/final_model"

    # GRPO specific parameters
    beta: float = 0.1  # KL penalty coefficient
    reward_model_path: Optional[str] = None  # If using external reward model

    # Regional reward computation
    use_regional_rewards: bool = True
    region_reward_weight: float = 0.5
    global_reward_weight: float = 0.5

    # Training hyperparameters
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    warmup_ratio: float = 0.1

    # PPO-style parameters
    ppo_epochs: int = 4
    chunk_size: int = 128
    mini_batch_size: int = 4

    # Ovis2.5-specific generation parameters (from official guide)
    generation_kwargs: dict = field(
        default_factory=lambda: {
            "max_new_tokens": 3072,  # As per official guide
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            # Ovis thinking mode parameters
            "enable_thinking": True,
            "enable_thinking_budget": True,
            "thinking_budget": 2048,  # As per official guide
        }
    )

    # Gradient accumulation
    gradient_accumulation_steps: int = 8

    # Mixed precision
    fp16: bool = False
    bf16: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3

    # Others
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])


@dataclass
class InferenceConfig:
    """Inference configuration for Ovis2.5-9B"""

    model_path: str = "outputs/grpo/final_model"  # Or "outputs/sft/final_model"

    # Ovis2.5 generation parameters (matching official guide)
    max_new_tokens: int = 3072
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Ovis thinking mode parameters
    enable_thinking: bool = True
    enable_thinking_budget: bool = True
    thinking_budget: int = 2048

    # Image processing - Ovis handles native resolution
    max_pixels: int = 896 * 896

    # Device
    device: str = "cuda"

    # Output parsing
    parse_tool_calls: bool = True
    extract_think_answer: bool = True
    parse_grounding: bool = True  # Parse <ref>, <box>, <point> tags
