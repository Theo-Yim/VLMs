"""
Data utilities for Ovis2.5-9B training
Based on official guide: https://huggingface.co/AIDC-AI/Ovis2.5-9B
"""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.append("..")
from crop_tool import CropTool, parse_and_replace_tool_calls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OvisDataset(Dataset):
    """Dataset for Ovis2.5-9B training"""

    def __init__(
        self,
        data_path: str,
        image_base_path: str,
        model: Any,
        max_length: int = 2048,
        max_pixels: int = 896 * 896,
        stage: str = "sft",
    ):
        self.data_path = data_path
        self.image_base_path = image_base_path
        self.model = model
        self.max_length = max_length
        self.max_pixels = max_pixels
        self.stage = stage
        self.crop_tool = CropTool()

        self.load_data()
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def load_data(self):
        """Load data from JSONL file"""
        self.data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if "image_path" in item and "QnA" in item:
                        self.data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {line.strip()}, error: {e}")

    def format_conversation_with_crops(self, item: Dict, image: Image.Image) -> List[Dict]:
        """Format QnA pairs with crop tool execution"""
        messages = []

        for qa in item.get("QnA", []):
            question = qa["Q"]
            answer = qa.get("A3", qa.get("A", ""))

            # Add user question with the main image
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            )

            # Simple processing that preserves <tool_call> tokens
            if answer and isinstance(answer, str):
                try:
                    # Just parse and replace tool calls - keep it simple
                    processed_answer = parse_and_replace_tool_calls(answer)
                    messages.append({"role": "assistant", "content": processed_answer})
                except Exception as e:
                    logger.warning(f"Error processing answer: {e}")
                    messages.append({"role": "assistant", "content": str(answer)})
            elif answer:
                messages.append({"role": "assistant", "content": str(answer)})

        return messages

    def process_item_with_ovis(self, messages: List[Dict]) -> Dict:
        """
        🎯 CRITICAL FIX: Return only the exact keys Ovis expects
        """
        try:
            input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=False,
                enable_thinking=True,
                max_pixels=self.max_pixels,
            )

            # 🎯 CRITICAL FIX: Return ONLY the standard keys that Ovis expects
            # Do NOT include any extra keys that might conflict
            result = {}

            if input_ids is not None:
                input_ids = input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids
                result["input_ids"] = input_ids

            if pixel_values is not None:
                pixel_values = pixel_values.squeeze(0) if pixel_values.dim() > 4 else pixel_values
                result["pixel_values"] = pixel_values

            if grid_thws is not None:
                grid_thws = grid_thws.squeeze(0) if grid_thws.dim() > 1 else grid_thws
                result["grid_thws"] = grid_thws

            # 🎯 CRITICAL: Do NOT add any other keys that might conflict with model arguments
            return result

        except Exception as e:
            logger.error(f"Error processing with Ovis: {e}")
            return self.fallback_processing(messages)

    def fallback_processing(self, messages: List[Dict]) -> Dict:
        """Fallback processing if Ovis preprocessing fails"""
        # Extract text content
        full_text = ""
        for msg in messages:
            if msg["role"] == "user":
                for content in msg["content"]:
                    if content["type"] == "text":
                        full_text += content["text"] + " "
            elif msg["role"] == "assistant":
                if isinstance(msg["content"], str):
                    full_text += msg["content"] + " "

        # Use model's text tokenizer
        tokenizer = getattr(self.model, "text_tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(self.model, "tokenizer", None)

        if tokenizer is None:
            logger.error("Could not find tokenizer")
            return {
                "input_ids": torch.tensor([0], dtype=torch.long),
                "labels": torch.tensor([0], dtype=torch.long),
            }

        try:
            encoding = tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return {
                "input_ids": torch.tensor([0], dtype=torch.long),
                "labels": torch.tensor([0], dtype=torch.long),
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get dataset item with proper tensor handling"""
        try:
            item = self.data[idx]
            image_path = os.path.join(self.image_base_path, item["image_path"])

            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return {
                    "input_ids": torch.tensor([0], dtype=torch.long),
                    "labels": torch.tensor([0], dtype=torch.long),
                }

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Format conversation
            messages = self.format_conversation_with_crops(item, image)

            if not messages:
                logger.warning(f"No valid messages for item {idx}")
                return {
                    "input_ids": torch.tensor([0], dtype=torch.long),
                    "labels": torch.tensor([0], dtype=torch.long),
                }

            # Process with Ovis
            processed = self.process_item_with_ovis(messages)

            # 🎯 CRITICAL FIX: Add labels only after processing
            if "input_ids" in processed and processed["input_ids"] is not None:
                processed["labels"] = processed["input_ids"].clone()

            return processed

        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            return {
                "input_ids": torch.tensor([0], dtype=torch.long),
                "labels": torch.tensor([0], dtype=torch.long),
            }


class OvisDataCollator:
    """
    Data collator for Ovis2.5-9B
    🎯 CRITICAL FIX: Clean input structure to prevent keyword conflicts
    """

    def __init__(self, model, padding=True, max_length=None):
        self.tokenizer = (
            model.text_tokenizer if hasattr(model, "text_tokenizer") else model.tokenizer
        )
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch):
        """
        🎯 CRITICAL FIX: Return clean input dict with only expected keys
        """
        # For Ovis2.5, batch_size=1 is required
        if len(batch) > 1:
            logger.warning(f"Batch size {len(batch)} > 1 detected. Using only first item.")
            batch = batch[:1]

        item = batch[0]

        # 🎯 CRITICAL FIX: Create clean batch dict with only the keys Ovis expects
        batch_dict = {}

        # Essential keys for Ovis
        if "input_ids" in item and item["input_ids"] is not None:
            batch_dict["input_ids"] = item["input_ids"].unsqueeze(0)

        if "labels" in item and item["labels"] is not None:
            batch_dict["labels"] = item["labels"].unsqueeze(0)

        # Visual inputs for Ovis
        if "pixel_values" in item and item["pixel_values"] is not None:
            batch_dict["pixel_values"] = item["pixel_values"].unsqueeze(0)

        if "grid_thws" in item and item["grid_thws"] is not None:
            batch_dict["grid_thws"] = item["grid_thws"].unsqueeze(0)

        # Optional attention mask
        if "attention_mask" in item and item["attention_mask"] is not None:
            batch_dict["attention_mask"] = item["attention_mask"].unsqueeze(0)

        # 🎯 CRITICAL: Do NOT include any other keys that might conflict
        # Specifically avoid: inputs_embeds, position_ids, etc.

        return batch_dict


class GroundingParser:
    """Parser for Ovis2.5 grounding format"""

    def __init__(self):
        self.ref_pattern = r"<ref>(.*?)</ref>"
        self.box_pattern = r"<box>\(([^)]+)\),\(([^)]+)\)</box>"
        self.point_pattern = r"<point>\(([^)]+)\)</point>"

    def parse_grounding(self, text: str) -> Dict[str, List]:
        """Parse grounding elements from text"""
        refs = re.findall(self.ref_pattern, text)

        box_matches = re.finditer(self.box_pattern, text)
        boxes = []
        for match in box_matches:
            try:
                x1, y1 = map(float, match.group(1).split(","))
                x2, y2 = map(float, match.group(2).split(","))
                boxes.append([x1, y1, x2, y2])
            except:
                continue

        point_matches = re.finditer(self.point_pattern, text)
        points = []
        for match in point_matches:
            try:
                x, y = map(float, match.group(1).split(","))
                points.append([x, y])
            except:
                continue

        return {"refs": refs, "boxes": boxes, "points": points}
