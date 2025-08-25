"""
Ovis2.5 Fine-tuning Script
Based on original Ovis training framework with custom modeling implementation
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from ovis.train.arguments import TrainingArguments as OvisTrainingArguments

# Import original Ovis training components
from ovis.train.dataset.conversation_dataset import ConversationDataset
from ovis.train.dataset.multimodal_dataset import DataCollatorForMultimodalDataset
from transformers import (
    HfArgumentParser,
    Trainer,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

# Import custom Ovis2.5 implementation
from ovis.model.modeling_ovis2_5 import Ovis2_5

try:
    import flash_attn

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# Import constants from the model
# from ovis.utils.constants import INDICATOR_IDS, VISUAL_ATOM_ID


@dataclass
class ModelArguments:
    """Arguments for model configuration"""

    model_path: str = field(default="AIDC-AI/Ovis2.5-9B")
    visual_vocab_size: int = field(default=65536)
    # attn_implementation: Optional[str] = field(default="flash_attention_2")


@dataclass
class CustomTrainingArguments(OvisTrainingArguments):
    """Extended training arguments for Ovis2.5 based on original Ovis arguments"""

    train_modules: str = field(default="all")  # all, llm, visual_tokenizer, vte, etc.
    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)

    # Override some defaults for Ovis2.5
    ovis_pretrained_path: Optional[str] = field(default="AIDC-AI/Ovis2.5-9B")
    stage: Optional[int] = field(default=3)  # Stage 3 for full training

    # Data paths
    data_path: str = field(default="./data/train_data.json")
    image_folder: str = field(default="./data/images")

    distribute_gpus: bool = field(default=False)


def create_dataset_info(training_args: CustomTrainingArguments) -> Dict[str, Dict]:
    """Create dataset info structure expected by original ConversationDataset"""
    return {
        training_args.data_name: {
            "meta_file": training_args.data_path,
            "storage_type": "hybrid",
            "data_format": "conversation",
            "image_dir": training_args.image_folder,
        }
    }


# def create_device_map(gpu_ids):
#     """
#     Create device map for Ovis2_5 model components across multiple GPUs.
#     Args:
#         gpu_ids: List of GPU IDs to use
#     Returns:
#         device_map: Dictionary mapping model components to GPUs
#     """
#     if len(gpu_ids) < 2:
#         return f"cuda:{gpu_ids[0]}"
#     # id 0: Vision components (vte, visual_tokenizer)
#     # id 1~: LLM components (llm)
#     device_map = {"vte": gpu_ids[0], "visual_tokenizer": gpu_ids[0]}
#     if len(gpu_ids) < 3:
#         device_map["llm"] = gpu_ids[1]
#     else:
#         device_map["llm"] = gpu_ids[1]
#         # TODO: For 3+ GPUs, we should distribute LLM layers further
#     return device_map


def split_ovis25_model():
    """
    Create device mapping for Ovis2.5 model to efficiently distribute across multiple GPUs.
    Architecture:
    - Visual processing: ViT (27 layers) + visual tokenizer head + VTE
    - LLM: Qwen3-8B (36 layers) + embeddings + norm + lm_head
    Data flow: Images -> ViT -> visual_head -> VTE -> merge with text embeddings -> LLM layers -> output
    """
    device_map = {}
    world_size = torch.cuda.device_count()
    if world_size < 2:
        # Single GPU - put everything on GPU 0
        return "auto"
    # Load config to get layer counts
    llm_layers = 36  # config.llm_config.num_hidden_layers  # 36 for Qwen3-8B
    device_map["visual_tokenizer.vit"] = 0  # Entire SigLIP ViT (27 layers)
    device_map["visual_tokenizer.head"] = 0  # Visual projection head
    device_map["vte"] = 0  # Visual token embeddings
    device_map["llm.model.embed_tokens"] = 0  # Text embeddings (need to merge with visual)
    # Distribute LLM layers across available GPUs
    if world_size == 2:
        # GPU 0: Vision + embeddings + first few LLM layers
        # GPU 1: Remaining LLM layers + norm + lm_head
        layers_gpu0 = 8  # Keep some LLM layers with embeddings
        for i in range(layers_gpu0):
            device_map[f"llm.model.layers.{i}"] = 0
        for i in range(layers_gpu0, llm_layers):
            device_map[f"llm.model.layers.{i}"] = 1
        device_map["llm.model.norm"] = 1
        device_map["llm.lm_head"] = 1
    elif world_size == 3:
        layers_per_gpu = [10, 13, 13]  # Slightly front-loaded due to vision processing
        layer_cnt = 0
        for gpu_id, num_layers in enumerate(layers_per_gpu):
            for i in range(num_layers):
                device_map[f"llm.model.layers.{layer_cnt}"] = gpu_id
                layer_cnt += 1
        device_map["llm.model.norm"] = 2
        device_map["llm.lm_head"] = 2
    else:  # world_size >= 4
        # Reserve some layers for GPU 0 (vision processing) and last GPU (output)
        layers_gpu0 = max(4, llm_layers // (world_size + 1))  # Front-load slightly
        layers_last_gpu = max(4, llm_layers // (world_size + 1))
        remaining_layers = llm_layers - layers_gpu0 - layers_last_gpu
        middle_gpus = world_size - 2
        if middle_gpus > 0:
            layers_per_middle_gpu = remaining_layers // middle_gpus
            extra_layers = remaining_layers % middle_gpus
        else:
            layers_per_middle_gpu = 0
            extra_layers = 0
            layers_last_gpu += remaining_layers
        # GPU 0: First layers
        for i in range(layers_gpu0):
            device_map[f"llm.model.layers.{i}"] = 0
        # Middle GPUs: Distribute remaining layers
        layer_cnt = layers_gpu0
        for gpu_id in range(1, world_size - 1):
            num_layers = layers_per_middle_gpu + (1 if gpu_id - 1 < extra_layers else 0)
            for i in range(num_layers):
                device_map[f"llm.model.layers.{layer_cnt}"] = gpu_id
                layer_cnt += 1
        # Last GPU: Final layers + norm + lm_head
        for i in range(layer_cnt, llm_layers):
            device_map[f"llm.model.layers.{i}"] = world_size - 1
        device_map["llm.model.norm"] = world_size - 1
        device_map["llm.lm_head"] = world_size - 1
    return device_map


def setup_model_for_training(model: Ovis2_5, training_args: CustomTrainingArguments):
    """Setup model parameters for training"""

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Selectively unfreeze based on train_modules
    train_modules = training_args.train_modules.lower()

    if "all" in train_modules:
        for param in model.parameters():
            param.requires_grad = True
    else:
        if "llm" in train_modules and not training_args.freeze_llm:
            for param in model.llm.parameters():
                param.requires_grad = True

        if "visual_tokenizer" in train_modules and not training_args.freeze_vision_tower:
            for param in model.visual_tokenizer.parameters():
                param.requires_grad = True

        if "vte" in train_modules:
            for param in model.vte.parameters():
                param.requires_grad = True

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")


def main():
    """Main training function"""

    config_file = (
        os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "./Ovis/src_theo/train_config.json"
    )

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(
            json_file=config_file, allow_extra_keys=True
        )
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed
    set_seed(training_args.seed)

    # Create device map for multi-GPU distribution
    device_map = None
    if training_args.distribute_gpus:
        device_map = split_ovis25_model()  # create_device_map(training_args.gpu_ids)
    else:
        device_map = f"cuda:{training_args.local_rank}" if training_args.local_rank != -1 else "cuda:0"

    # Load model with device map
    print(f"Loading model from {model_args.model_path}...")
    model = Ovis2_5.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2" if HAS_FLASH_ATTN else "eager",
        device_map=device_map,
    )

    # Setup model for training
    setup_model_for_training(model, training_args)

    # Prepare tokenizer
    tokenizer = model.text_tokenizer
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset info structure for original ConversationDataset
    dataset_info = create_dataset_info(training_args)

    # Create dataset using original ConversationDataset
    print(f"Loading dataset from {training_args.data_path}...")
    train_dataset = ConversationDataset(
        name=training_args.data_name,
        info=dataset_info[training_args.data_name],
        model=model,
        training_args=training_args,
    )

    # Use original data collator
    data_collator = DataCollatorForMultimodalDataset(model.text_tokenizer)

    # Initialize trainer
    # trainer = SFTTrainer(
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Training
    print("Starting training...")
    trainer.train()

    # Save model
    print("Saving model...")
    trainer.save_model()
    trainer.save_state()

    print("Training completed!")


if __name__ == "__main__":
    main()
