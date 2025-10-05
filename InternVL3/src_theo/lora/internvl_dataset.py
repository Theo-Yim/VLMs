"""
InternVL 3.5 Dataset and Data Collator for Training
Handles multimodal data with thinking mode support
"""

import json
import os
from typing import Any, Dict

import torch
from torch.utils.data import Dataset

from InternVL3.utils.preprocess import load_image

# InternVL 3.5 Thinking Mode System Prompt
THINKING_SYSTEM_PROMPT = """You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.""".strip()


class InternVLDataset(Dataset):
    """Dataset class for InternVL 3.5 training data."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_size: int = 448,
        max_dynamic_patches: int = 12,
    ):
        super().__init__()
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_dynamic_patches = max_dynamic_patches

        # Load data
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} training samples")

    def __len__(self):
        return len(self.data)

    def has_thinking_mode(self, response: str) -> bool:
        """Check if response contains thinking tags."""
        return "<think>" in response and "</think>" in response

    def format_conversation(self, sample: Dict[str, Any], num_patches: int = 0) -> str:
        """Format conversation with appropriate system prompt and expand image tokens."""
        conversations = sample["conversations"]

        # Determine if we need thinking mode system prompt
        assistant_responses = [conv["value"] for conv in conversations if conv["from"] == "gpt"]
        use_thinking = any(self.has_thinking_mode(resp) for resp in assistant_responses)

        # Build conversation
        formatted = ""

        # Add system prompt if using thinking mode
        if use_thinking:
            formatted += f"<|im_start|>system\n{THINKING_SYSTEM_PROMPT}<|im_end|>\n"

        # Calculate image tokens: InternVL uses 256 tokens per patch (downsample_ratio=0.5, so 16x16 patches)
        # For image_size=448, patch_size=14: (448/14)^2 = 1024 base tokens, with downsample 0.5: 1024*0.25 = 256
        num_image_token = 256
        
        # Add conversation turns  
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            message = conv["value"]
            
            # Expand <image> to multiple <IMG_CONTEXT> tokens if we have patches
            if "<image>" in message and num_patches > 0:
                # Format: <img><IMG_CONTEXT>*N</img> where N = num_image_token * num_patches
                img_context_tokens = "<IMG_CONTEXT>" * (num_image_token * num_patches)
                image_tokens = f"<img>{img_context_tokens}</img>"
                message = message.replace("<image>", image_tokens)
            
            formatted += f"<|im_start|>{role}\n{message}<|im_end|>\n"

        return formatted

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample."""
        sample = self.data[idx]

        # Load image if present
        pixel_values = None
        num_patches = 0
        if "image" in sample and sample["image"]:
            image_path = os.path.join(self.image_folder, sample["image"])
            pixel_values = load_image(
                image_path, max_num=self.max_dynamic_patches, input_size=self.image_size
            )
            if pixel_values is not None:
                num_patches = pixel_values.shape[0]  # First dim is number of patches

        # Format conversation with image token expansion
        text = self.format_conversation(sample, num_patches)

        return {"text": text, "pixel_values": pixel_values, "sample_id": sample.get("id", str(idx))}


class MultimodalDataCollator:
    """Custom data collator for handling both text and images."""

    def __init__(self, tokenizer, response_template: str = "<|im_start|>assistant\n"):
        self.tokenizer = tokenizer
        self.response_template = response_template

    def __call__(self, features):
        texts = [f["text"] for f in features]
        pixel_values_list = [f.get("pixel_values") for f in features]

        batch = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
        )
        batch["labels"] = batch["input_ids"].clone()

        # Concatenate all patches and create flags (one per patch)
        all_pixel_values = []
        image_flags = []

        for pv in pixel_values_list:
            if pv is not None:
                num_patches = pv.shape[0]
                all_pixel_values.append(pv)  # (num_patches, 3, H, W)
                image_flags.extend([1] * num_patches)  # One flag per patch
            # Note: text-only samples don't add patches or flags

        if all_pixel_values:
            batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)  # (total_patches, 3, H, W)
            batch["image_flags"] = torch.tensor(image_flags, dtype=torch.long).unsqueeze(
                -1
            )  # (total_patches, 1)
        else:
            # No images in batch - provide dummy
            batch["pixel_values"] = torch.zeros(1, 3, 448, 448)
            batch["image_flags"] = torch.zeros(1, 1, dtype=torch.long)

        return batch


def create_internvl_dataset(
    data_path: str,
    image_folder: str,
    tokenizer,
    image_size: int = 448,
    max_dynamic_patches: int = 12,
):
    """Create InternVL dataset."""
    dataset = InternVLDataset(
        data_path=data_path,
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_size=image_size,
        max_dynamic_patches=max_dynamic_patches,
    )
    return dataset
