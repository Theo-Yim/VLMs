"""
Configuration for Qwen 2.5 VL training
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration"""

    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Custom tokens
    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"
    tool_call_start_token_id: int = 151657
    tool_call_end_token_id: int = 151658


@dataclass
class DataConfig:
    """Data configuration"""

    train_data_path: str = "../data/refcoco_qa_pairs_train.jsonl"
    val_data_path: str = "../data/refcoco_qa_pairs_val.jsonl"
    image_base_path: str = "/mnt/nas3/Data/"

    # Image processing
    image_size: int = 448  # Shorter side
    max_length: int = 2048

    # Data loading
    train_batch_size: int = 4
    eval_batch_size: int = 8
    num_workers: int = 4


@dataclass
class SFTTrainingConfig:
    """Stage 1: Supervised Fine-Tuning configuration"""

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

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 128  # Increased for larger dataset and VLM complexity
    lora_alpha: int = 256  # 2x rank for stable training
    lora_dropout: float = 0.05  # Reduced for larger dataset
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            # Language model attention and MLP
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # Vision-language connector modules (adjust based on model architecture)
            "c_attn",
            "c_proj",  # Additional attention projections
            # Vision encoder (if accessible)
            # "vision.transformer.resblocks.*.attn.in_proj_weight",  # Uncomment if targeting vision encoder
        ]
    )

    # Alternative: Full fine-tuning configuration (for comparison)
    # Set use_lora = False and adjust learning rate to 5e-6 for full fine-tuning
    # Recommended for datasets >100k images or when LoRA results are insufficient


@dataclass
class GRPOTrainingConfig:
    """Stage 2: Regional GRPO configuration"""

    output_dir: str = "outputs/grpo"

    # Load from SFT checkpoint
    sft_checkpoint_path: str = "outputs/sft/best_model"

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

    # Generation parameters for sampling
    generation_kwargs: dict = field(
        default_factory=lambda: {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
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
    """Inference configuration"""

    model_path: str = "outputs/grpo/best_model"  # Or "outputs/sft/best_model"

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1

    # Image processing
    image_size: int = 448

    # Device
    device: str = "cuda"

    # Output parsing
    parse_tool_calls: bool = True
    extract_think_answer: bool = True
