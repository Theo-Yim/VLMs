"""
Crop tool implementation for Ovis2.5 training.
Handles image cropping based on tool call coordinates and creates multimodal sequences.

Key functions:
- extract_tool_calls(): Parse <tool_call>Crop [x1,y1,x2,y2]</tool_call> patterns
- crop_image(): Crop image regions based on coordinates
- create_multimodal_content(): Create text+image sequences for training
- parse_and_replace_tool_calls(): Convert {Crop ...} to <tool_call>...</tool_call> format
"""

import re
from typing import Any, Dict, List

import numpy as np
from PIL import Image


class CropTool:
    """
    Tool for cropping image regions based on coordinates.
    Handles conversion from raw format to training format and creates multimodal sequences.
    """

    def __init__(self):
        self.tool_call_pattern = r"<tool_call>Crop \[([0-9.,\s]+)\]</tool_call>"

    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all tool calls from text
        Returns list of tool calls with their positions and coordinates
        """
        tool_calls = []

        for match in re.finditer(self.tool_call_pattern, text):
            coords_str = match.group(1)
            coords = [float(x.strip()) for x in coords_str.split(",")]

            if len(coords) == 4:
                tool_calls.append(
                    {
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "full_match": match.group(0),
                        "coordinates": coords,
                        "bbox": {
                            "x1": coords[0],
                            "y1": coords[1],
                            "x2": coords[2],
                            "y2": coords[3],
                        },
                    }
                )

        return tool_calls

    def crop_image(
        self, image: Image.Image, bbox: List[float], return_pil: bool = True
    ) -> Image.Image:
        """
        Crop image based on bounding box coordinates

        Args:
            image: PIL Image to crop
            bbox: [x1, y1, x2, y2] coordinates
            return_pil: Whether to return PIL Image (True) or numpy array (False)

        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = bbox
        width, height = image.size

        # Convert to pixel coordinates
        x1_px = int(max(0, min(x1, width)))
        y1_px = int(max(0, min(y1, height)))
        x2_px = int(max(0, min(x2, width)))
        y2_px = int(max(0, min(y2, height)))

        # Ensure valid crop region
        if x2_px <= x1_px or y2_px <= y1_px:
            # Return a small valid region if invalid coordinates
            x2_px = min(x1_px + 10, width)
            y2_px = min(y1_px + 10, height)

        # Crop the image
        cropped = image.crop((x1_px, y1_px, x2_px, y2_px))

        if return_pil:
            return cropped
        else:
            return np.array(cropped)

    def create_multimodal_content(self, text: str, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Create multimodal content list with text and cropped images interspersed

        Args:
            text: Text containing <tool_call>Crop [...]</tool_call> patterns
            image: Original image to crop from

        Returns:
            List of content items: [{"type": "text", "text": "..."}, {"type": "image", "image": PIL.Image}, ...]
        """
        tool_calls = self.extract_tool_calls(text)

        if not tool_calls:
            # No tool calls, return simple text content
            return [{"type": "text", "text": text}]

        content = []
        last_pos = 0

        for tool_call in tool_calls:
            # Add text up to and including the tool call
            text_segment = text[last_pos : tool_call["end_pos"]]
            if text_segment.strip():
                content.append({"type": "text", "text": text_segment})

            # Add cropped image right after the tool call
            try:
                cropped_image = self.crop_image(image, tool_call["coordinates"])
                content.append({"type": "image", "image": cropped_image})
            except Exception as e:
                print(
                    f"Warning: Failed to crop image with coordinates {tool_call['coordinates']}: {e}"
                )
                # Continue without the cropped image

            last_pos = tool_call["end_pos"]

        # Add remaining text after the last tool call
        if last_pos < len(text):
            remaining_text = text[last_pos:]
            if remaining_text.strip():
                content.append({"type": "text", "text": remaining_text})

        return content


def parse_and_replace_tool_calls(text: str) -> str:
    """
    Parse tool calls in the format {Crop person 1 [0.00, 141.43, 79.23, 480.00]}
    and replace with <tool_call>Crop [0.00, 141.43, 79.23, 480.00]</tool_call>

    Example Usage:
    When a3_answer is "<think>Let me closely look at the person.\n\n{Crop person 2 [181, 16, 220, 191]}\n\nUpon closer inspection, the person is engaging with the camera.</think>\n\n<answer>The person is engaging with the camera.\n</answer>"
    a3_answer_processed = parse_and_replace_tool_calls(a3_answer)
    a3_answer_processed is "<think>Let me closely look at the person.\n\n<tool_call>Crop [181, 16, 220, 191]</tool_call>\n\nUpon closer inspection, the person is engaging with the camera.</think>\n\n<answer>The person is engaging with the camera.\n</answer>"
    messages.append({"role": "assistant", "content": a3_answer_processed})
    """
    pattern = r"\{Crop[^}]*\[([\d.,\s]+)\]\}"

    def replace_func(match):
        coords = match.group(1)
        return f"<tool_call>Crop [{coords}]</tool_call>"

    return re.sub(pattern, replace_func, text)
