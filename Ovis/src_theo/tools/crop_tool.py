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
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .tool_base import ToolBase


class CropTool(ToolBase):
    """
    Tool for cropping image regions based on coordinates.
    You can call this tool to crop and investigate the details of the image region.
    Usage: <tool_call>Crop [x1, y1, x2, y2]</tool_call>
    parameters: [x1, y1, x2, y2] : bbox coordinates of the region to crop
    Example: "To investigate the person in the center, I should call crop tool.\n<tool_call>Crop [300,200,600,700]</tool_call>\n"
    """

    def __init__(self):
        self.tool_call_pattern = r"<tool_call>Crop \[([0-9.,\s]+)\]</tool_call>"

    # def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
    #     """
    #     Extract all tool calls from text (legacy method for compatibility)
    #     Returns list of tool calls with their positions and coordinates
    #     """
    #     tool_calls = []

    #     for match in re.finditer(self.tool_call_pattern, text):
    #         coords_str = match.group(1)
    #         coords = [float(x.strip()) for x in coords_str.split(",")]

    #         if len(coords) == 4:
    #             tool_calls.append(
    #                 {
    #                     "start_pos": match.start(),
    #                     "end_pos": match.end(),
    #                     "full_match": match.group(0),
    #                     "parameters": [coords, True],
    #                 }
    #             )

    #     return tool_calls

    def extract_tool_call(self, text: str) -> Dict[str, Any]:
        """
        Optimized extraction for single tool call.
        Returns parameters only since that's all we need now.
        """
        match = re.search(self.tool_call_pattern, text)
        if match:
            coords_str = match.group(1)
            coords = [float(x.strip()) for x in coords_str.split(",")]
            if len(coords) == 4:
                return {
                    "parameters": [coords, True],
                    "tool_name": "crop",  # Each tool manages its own name
                    "tool_instance": self,  # Each tool manages its own instance
                }
        return None

    def execute(self, image: Optional[Image.Image], parameters: Any) -> Optional[dict]:
        """Execute crop tool and return structured result or None if no image."""
        if image is None:
            # Cannot crop without an image - return None (nothing to append)
            return None

        cropped_image = self.crop_image(image, parameters[0], parameters[1])
        return {"type": "image", "content": cropped_image}

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
