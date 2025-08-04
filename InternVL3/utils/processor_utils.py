import math
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    # TODO: check this Manually adjusted numbers
    if world_size >= 4:
        num_layers_per_gpu[0] -= (world_size - 2)
        for i in range(1, world_size - 1):
            num_layers_per_gpu[i] += 1

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0  # seems not exist in 78B
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0  # seems not exist in 78B
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    # device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


def split_model_for_group(model_path, gpu_group: list):
    """Create device map for a specific group of 4 GPUs"""
    device_map = {}
    world_size = len(gpu_group)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        gpu_id = gpu_group[i]
        for _ in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = gpu_id
            layer_cnt += 1
    first_gpu = gpu_group[0]
    device_map["vision_model"] = first_gpu
    device_map["mlp1"] = first_gpu
    device_map["language_model.model.tok_embeddings"] = first_gpu
    device_map["language_model.model.embed_tokens"] = first_gpu
    device_map["language_model.output"] = first_gpu
    device_map["language_model.model.norm"] = first_gpu
    device_map["language_model.model.rotary_emb"] = first_gpu
    device_map["language_model.lm_head"] = first_gpu

    return device_map


def load_models(model_path="OpenGVLab/InternVL3-78B", device_map: list = None):
    # If you set `load_in_8bit=True`, you will need two 80GB GPUs.
    # If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
    # model_path = "OpenGVLab/InternVL3-78B"
    # device_map = split_model(model_path)
    if device_map is None:
        device_map = split_model(model_path)

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        print("Failed to compile model. Continuing without compilation.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    return model, tokenizer
