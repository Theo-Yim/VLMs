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
    """
    Dataset for Ovis2.5-9B training
    Key corrections from official guide:
    1. Uses model.text_tokenizer for decoding
    2. Uses proper max_pixels parameter
    3. Supports Ovis grounding format: <ref>, <box>, <point>
    """

    def __init__(
        self,
        data_path: str,
        image_base_path: str,
        model: Any,  # Ovis model for preprocessing
        max_length: int = 2048,
        max_pixels: int = 896 * 896,  # As per official guide
        stage: str = "sft",
    ):
        self.data_path = data_path
        self.image_base_path = image_base_path
        self.model = model
        self.max_length = max_length
        self.max_pixels = max_pixels
        self.stage = stage
        self.crop_tool = CropTool()  # Use existing CropTool from QwenVL2.5

        # Load data
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
        """
        Format QnA pairs into conversation format with crop tool execution
        Supports both custom <tool_call> tokens and native Ovis grounding

        For each QnA pair:
        1. Parse tool calls in the answer using crop tool
        2. Execute crops and interleave cropped images in the response
        3. Optionally convert to native Ovis grounding format
        """
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

            # Process answer with crop tool to create multi-modal response
            if answer:
                # First, parse and replace tool calls (same as QwenVL2.5 does)
                processed_answer = parse_and_replace_tool_calls(answer)

                # Then use existing CropTool.format_for_training method (same as QwenVL2.5)
                content_items = self.crop_tool.format_for_training(processed_answer, image)

                # Convert to Ovis message format
                assistant_content = []
                for item in content_items:
                    if item["type"] == "text":
                        # Optionally convert <tool_call> to native Ovis grounding
                        text_content = item["text"]
                        if hasattr(self, "use_native_grounding") and self.use_native_grounding:
                            text_content = self.convert_tool_calls_to_grounding(text_content)
                        assistant_content.append({"type": "text", "text": text_content})
                    elif item["type"] == "image":
                        assistant_content.append({"type": "image", "image": item["image"]})

                if assistant_content:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_content
                            if len(assistant_content) > 1
                            else assistant_content[0]["text"]
                            if assistant_content[0]["type"] == "text"
                            else assistant_content,
                        }
                    )

        return messages

    def convert_tool_calls_to_grounding(self, text: str) -> str:
        """
        Convert <tool_call>[x1,y1,x2,y2]</tool_call> to <ref>description</ref><box>(x1,y1),(x2,y2)</box>
        This converts from custom tokens to native Ovis grounding format
        """
        # Extract tool calls
        tool_calls = self.crop_tool.extract_tool_calls(text)

        if not tool_calls:
            return text

        converted_text = text

        # Process in reverse order to maintain positions
        for tool_call in reversed(tool_calls):
            coords = tool_call["coordinates"]
            if len(coords) == 4:
                x1, y1, x2, y2 = coords

                # Use generic description (would need original descriptions for better results)
                description = "region"

                # Convert to Ovis grounding format (normalized coordinates)
                if max(coords) > 1.0:
                    # If pixel coordinates, normalize (approximate)
                    x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000

                grounding_text = (
                    f"<ref>{description}</ref><box>({x1:.3f},{y1:.3f}),({x2:.3f},{y2:.3f})</box>"
                )

                # Replace the tool call with grounding format
                converted_text = (
                    converted_text[: tool_call["start_pos"]]
                    + grounding_text
                    + converted_text[tool_call["end_pos"] :]
                )

        return converted_text

    def extract_think_answer(self, text: str) -> Tuple[str, str]:
        """
        Extract <think> and <answer> content from response
        Returns (think_content, answer_content)
        """
        think_pattern = r"<think>(.*?)</think>"
        answer_pattern = r"<answer>(.*?)</answer>"

        think_match = re.search(think_pattern, text, re.DOTALL)
        answer_match = re.search(answer_pattern, text, re.DOTALL)

        think_content = think_match.group(1).strip() if think_match else ""
        answer_content = answer_match.group(1).strip() if answer_match else text

        return think_content, answer_content

    def process_item_with_ovis(self, messages: List[Dict]) -> Dict:
        """
        Process item using Ovis's preprocessing pipeline
        Using parameters from official guide
        """
        try:
            # Use Ovis's preprocessing method with proper parameters
            input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=False,  # We want the full conversation for training
                enable_thinking=True,  # Enable thinking mode during training
                max_pixels=self.max_pixels,  # Control resolution
            )

            return {
                "input_ids": input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids,
                "pixel_values": pixel_values.squeeze(0)
                if pixel_values is not None and pixel_values.dim() > 4
                else pixel_values,
                "grid_thws": grid_thws.squeeze(0)
                if grid_thws is not None and grid_thws.dim() > 1
                else grid_thws,
            }

        except Exception as e:
            logger.error(f"Error processing with Ovis: {e}")
            # Fallback to basic tokenization
            return self.fallback_processing(messages)

    def fallback_processing(self, messages: List[Dict]) -> Dict:
        """
        Fallback processing if Ovis preprocessing fails
        """
        # Extract text content for basic tokenization
        full_text = ""
        for msg in messages:
            if msg["role"] == "user":
                # Find text content in user message
                for content in msg["content"]:
                    if content["type"] == "text":
                        full_text += content["text"] + " "
            elif msg["role"] == "assistant":
                full_text += msg["content"]

        # Use model's text tokenizer (key correction from official guide)
        if hasattr(self.model, "text_tokenizer"):
            tokenizer = self.model.text_tokenizer
        else:
            # Fallback to model tokenizer
            tokenizer = getattr(self.model, "tokenizer", None)
            if tokenizer is None:
                logger.error("Could not find tokenizer in model")
                raise RuntimeError("No tokenizer found")

        # Tokenize
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
            "pixel_values": None,
            "grid_thws": None,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get image path
        image_path = os.path.join(self.image_base_path, item["image_path"])

        # Check if image exists
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            # Return a placeholder or skip
            return self.__getitem__((idx + 1) % len(self.data))

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Format conversation with crop tool (like QwenVL2.5 does)
            messages = self.format_conversation_with_crops(item, image)

            # Process with Ovis
            processed = self.process_item_with_ovis(messages)

            # Create labels (same as input_ids for causal LM training)
            if "input_ids" in processed:
                processed["labels"] = processed["input_ids"].clone()

            return processed

        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            # Skip to next item
            return self.__getitem__((idx + 1) % len(self.data))


class OvisDataCollator:
    """
    Data collator for Ovis2.5-9B
    Handles the multimodal nature of Ovis inputs
    """

    def __init__(self, model, padding=True, max_length=None):
        # Use model's text tokenizer (key correction)
        self.tokenizer = (
            model.text_tokenizer if hasattr(model, "text_tokenizer") else model.tokenizer
        )
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch):
        # Separate different types of data
        input_ids = []
        labels = []
        pixel_values = []
        grid_thws = []
        attention_masks = []

        for item in batch:
            if "input_ids" in item:
                input_ids.append(item["input_ids"])

            if "labels" in item:
                labels.append(item["labels"])

            if "attention_mask" in item:
                attention_masks.append(item["attention_mask"])

            if "pixel_values" in item and item["pixel_values"] is not None:
                pixel_values.append(item["pixel_values"])

            if "grid_thws" in item and item["grid_thws"] is not None:
                grid_thws.append(item["grid_thws"])

        # Create batch dictionary
        batch_dict = {}

        # Handle input_ids and labels
        if input_ids:
            # Pad sequences
            padded_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            batch_dict["input_ids"] = padded_ids

        if labels:
            padded_labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            )
            batch_dict["labels"] = padded_labels

        if attention_masks:
            padded_attention = torch.nn.utils.rnn.pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            )
            batch_dict["attention_mask"] = padded_attention

        # Handle visual data
        if pixel_values:
            # Stack pixel values
            batch_dict["pixel_values"] = torch.stack(pixel_values)

        if grid_thws:
            # Stack grid_thws
            batch_dict["grid_thws"] = torch.stack(grid_thws)

        return batch_dict


class GroundingParser:
    """
    Parser for Ovis2.5 grounding format
    Handles <ref>, <box>, <point> tags as per official guide
    """

    def __init__(self):
        self.ref_pattern = r"<ref>(.*?)</ref>"
        self.box_pattern = r"<box>\(([^)]+)\),\(([^)]+)\)</box>"
        self.point_pattern = r"<point>\(([^)]+)\)</point>"

    def parse_grounding(self, text: str) -> Dict[str, List]:
        """
        Parse grounding elements from text
        Returns dict with refs, boxes, and points
        """
        # Find all references
        refs = re.findall(self.ref_pattern, text)

        # Find all boxes
        box_matches = re.finditer(self.box_pattern, text)
        boxes = []
        for match in box_matches:
            try:
                x1, y1 = map(float, match.group(1).split(","))
                x2, y2 = map(float, match.group(2).split(","))
                boxes.append([x1, y1, x2, y2])
            except:
                continue

        # Find all points
        point_matches = re.finditer(self.point_pattern, text)
        points = []
        for match in point_matches:
            try:
                x, y = map(float, match.group(1).split(","))
                points.append([x, y])
            except:
                continue

        return {"refs": refs, "boxes": boxes, "points": points}
