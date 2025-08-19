"""
Data utilities for Qwen 2.5 VL training
Handles data loading, preprocessing, and tool call parsing with crop execution
"""

import json
from typing import Dict, List, Any
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLProcessor
import sys

sys.path.append("..")
from crop_tool import CropTool, parse_and_replace_tool_calls


def resize_image_shortest_side(image: Image.Image, target_size: int = 448) -> Image.Image:
    """
    Resize image so that the shorter side equals target_size while maintaining aspect ratio
    """
    width, height = image.size

    if width < height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))

    return image.resize((new_width, new_height), Image.LANCZOS)


def resize_crop_smart(image: Image.Image, target_size: int = 448) -> Image.Image:
    """
    Smart resize for cropped regions:
    - Only resize if shorter side > target_size (avoid upscaling)
    - Preserve original size if shorter side <= target_size
    - Maintains aspect ratio
    """
    width, height = image.size
    shorter_side = min(width, height)

    # Only resize if the image is larger than target size
    if shorter_side <= target_size:
        return image  # Keep original size

    # Resize down to target size
    if width < height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))

    return image.resize((new_width, new_height), Image.LANCZOS)


class QwenVLDataset(Dataset):
    """Dataset for Qwen 2.5 VL training with crop tool support"""

    def __init__(
        self,
        data_path: str,
        image_base_path: str,
        processor: Qwen2VLProcessor,
        max_length: int = 2048,
        image_size: int = 448,
        stage: str = "sft",  # "sft" or "grpo"
    ):
        self.data_path = Path(data_path)
        self.image_base_path = Path(image_base_path)
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.stage = stage
        self.crop_tool = CropTool()

        # Load data
        with open(self.data_path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def format_conversation_with_crops(self, item: Dict, image: Image.Image) -> List[Dict]:
        """
        Format QnA pairs into conversation format with crop tool execution

        For each QnA pair:
        1. Parse tool calls in the answer
        2. Execute crops and interleave cropped images in the response
        """
        messages = []

        for qa in item.get("QnA", []):
            answer = qa.get("A3", "")

            # Parse and replace tool call format
            if answer:
                answer = parse_and_replace_tool_calls(answer)

            # Add user question with the main image
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": qa["Q"]},
                    ],
                }
            )

            # Process answer with crop tool to create multi-modal response
            if answer:
                content_items = self.crop_tool.format_for_training(answer, image)

                # Convert to message format
                assistant_content = []
                for item in content_items:
                    if item["type"] == "text":
                        assistant_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image":
                        # Smart resize for cropped image (avoid upscaling small crops)
                        cropped_resized = resize_crop_smart(item["image"], self.image_size)
                        assistant_content.append({"type": "image", "image": cropped_resized})

                messages.append({"role": "assistant", "content": assistant_content})
            else:
                # No answer provided
                messages.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})

        return messages

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Load and process image
        image_path = self.image_base_path / item["image_path"]
        image = Image.open(image_path).convert("RGB")

        # Format conversation with crop tool execution
        messages = self.format_conversation_with_crops(item, image)

        # Extract all images from messages
        images = []
        for msg in messages:
            for content_item in msg.get("content", []):
                if content_item.get("type") == "image":
                    img = content_item.get("image")
                    if img is not None:
                        images.append(img)

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process inputs with all images
        inputs = self.processor(
            text=text,
            images=images if images else None,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        # For GRPO stage, store minimal additional info
        if self.stage == "grpo":
            inputs["image_id"] = item.get("image_id", f"img_{idx}")
            inputs["original_image"] = image  # Keep original for reward computation

        # Flatten batch dimension
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].squeeze(0)

        return inputs
