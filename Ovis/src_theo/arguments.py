"""
Training arguments for Ovis model training
Based on original Ovis training structure - clean and simple
"""

import dataclasses
from typing import Optional
from transformers import TrainingArguments as HfTrainingArguments


@dataclasses.dataclass
class ModelArguments:
    """Arguments for model configuration - matches original Ovis ModelArguments exactly"""
    
    llm_name_or_path: Optional[str] = dataclasses.field(default="Qwen/Qwen3-8B")
    vit_name_or_path: Optional[str] = dataclasses.field(default="google/siglip2-so400m-patch16-512")
    visual_vocab_size: int = dataclasses.field(default=65536)
    conversation_formatter_class: str = dataclasses.field(default="Qwen3ConversationFormatter")
    attn_implementation: Optional[str] = dataclasses.field(default=None)
    accepts_loss_kwargs: bool = dataclasses.field(default=True)
    
    # ViT Configuration (original defaults)
    vit_hidden_stride: int = dataclasses.field(default=2)
    vit_window_size: int = dataclasses.field(default=112)
    vit_temporal_patch_size: int = dataclasses.field(default=1)
    vit_fullatt_block_indexes: Optional[str] = dataclasses.field(default=None)
    vit_preserve_original_pe: Optional[bool] = dataclasses.field(default=True)
    vit_use_rope: Optional[bool] = dataclasses.field(default=True)


@dataclasses.dataclass
class TrainingArguments(HfTrainingArguments):
    """Extended training arguments for Ovis - matches original structure exactly"""
    
    # Original Ovis training arguments
    data_info_version: Optional[str] = dataclasses.field(default=None)
    data_name: Optional[str] = dataclasses.field(default=None)
    data_type: Optional[str] = dataclasses.field(default="conversation")
    ovis_pretrained_path: Optional[str] = dataclasses.field(default=None)
    stage: Optional[int] = dataclasses.field(default=3)
    train_modules: Optional[str] = dataclasses.field(default="all")
    cache_dir: Optional[str] = dataclasses.field(default=None)
    optim: str = dataclasses.field(default="adamw_torch")
    save_safetensors: bool = dataclasses.field(default=True)
    monitor_step: int = dataclasses.field(default=100)
    model_init_seed: int = dataclasses.field(default=0)
    
    # Length configuration (original defaults)
    multimodal_max_length: int = dataclasses.field(default=4096)
    text_max_length: Optional[int] = dataclasses.field(default=4096)
    min_frames: int = dataclasses.field(default=8)
    max_frames: int = dataclasses.field(default=8)
    
    # Image pixel configuration (original defaults)
    single_image_min_pixels: int = dataclasses.field(default=448*448)
    single_image_max_pixels: int = dataclasses.field(default=1792*1344)
    multiple_image_min_pixels: int = dataclasses.field(default=448*448)
    multiple_image_max_pixels: int = dataclasses.field(default=896*896)
    video_min_pixels: int = dataclasses.field(default=448*448)
    video_max_pixels: int = dataclasses.field(default=896*896)
    
    # Data mixing (original)
    overall_ratio: Optional[str] = dataclasses.field(default=None)
    mix_data_name: Optional[str] = dataclasses.field(default=None)
    mix_ratio: Optional[float] = dataclasses.field(default=None)
    min_lr_rate: Optional[float] = dataclasses.field(default=None)
    
    # Custom paths for this implementation
    data_path: Optional[str] = dataclasses.field(default=None)
    image_folder: Optional[str] = dataclasses.field(default=None)
    
    def __post_init__(self):
        if self.min_lr_rate is not None:
            self.lr_scheduler_kwargs = {"min_lr_rate": self.min_lr_rate}
        if self.gradient_checkpointing:
            self.gradient_checkpointing_kwargs = {"use_reentrant": False}
        if self.stage < 3:
            self.save_safetensors = False
        super().__post_init__()
        assert self.model_init_seed != self.seed, "`model_init_seed` should be different from `seed`"