import re
from typing import Any, Dict, List, Optional

from PIL import Image

from .tool_base import ToolBase


class IdentifyTool(ToolBase):
    """
    Tool for identifying people or objects in image regions
    You can call this tool to identify the name of the people or objects in the image region.
    Usage: <tool_call>Identify [x1, y1, x2, y2]</tool_call>
    parameters: [x1, y1, x2, y2] : bbox coordinates of the region to identify
    Example: "To identify the person in the center, I should call identify tool.\n<tool_call>Identify [300,200,600,700]</tool_call>\n"
    """

    def __init__(self):
        # Pattern for inference (no <tool_response> token)
        self.tool_call_pattern = r"<tool_call>Identify \[([0-9.,\s]+)\]</tool_call>"

        # Pattern for training detection (with <tool_response> token marker)
        self.tool_call_with_response_pattern = r"<tool_call>Identify \[([0-9.,\s]+)\]</tool_call><tool_response>([^<]+)</tool_response>"

    def extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract single tool call for INFERENCE.

        Used during real-time generation to detect tool calls.
        Pattern: <tool_call>Identify [x1,y1,x2,y2]</tool_call> (NO <tool_response>)

        Returns:
            Tool call dict or None if not found
        """
        match = re.search(self.tool_call_pattern, text)
        if match:
            coords_str = match.group(1)
            coords = [float(x.strip()) for x in coords_str.split(",")]
            if len(coords) == 4:
                return {
                    "parameters": [coords],
                    "tool_name": "identify",
                    "tool_instance": self,
                }
        return None

    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all tool calls for TRAINING DETECTION.

        Detects pattern WITH <tool_response> for training data.

        Returns:
            List of tool call dicts with position info
        """
        tool_calls = []

        # Pattern WITH </tool_call><tool_response>...</tool_response> (training format)
        for match in re.finditer(self.tool_call_with_response_pattern, text):
            coords_str = match.group(1)
            coords = [float(x.strip()) for x in coords_str.split(",")]
            if len(coords) == 4:
                tool_calls.append(
                    {
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "full_match": match.group(0),
                        "parameters": [coords],
                    }
                )

        return tool_calls

    def execute(self, image: Optional[Image.Image], parameters: Any) -> Optional[Dict]:
        """
        Execute identify tool - used for BOTH inference and training.

        Returns structured result based on whether image is available.
        The return type determines how inference/training pipeline handles it:
        - {"type": "text", "content": "name"} → wrapped in <tool_response>

        Args:
            image: PIL Image to identify from (None if unavailable)
            parameters: [coords] from tool call

        Returns:
            {"type": "text", "content": identification_result}
        """
        if image is None:
            return {"type": "text", "content": "Unable to identify - no image provided"}

        return self.identify(image, parameters[0])

    def identify(self, image: Image.Image, bbox: List[float]) -> Dict:
        """Execute identification operation"""
        coordinates = bbox

        if len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            width, height = image.size

            # Convert to pixel coordinates
            x1_px = int(max(0, min(x1, width)))
            y1_px = int(max(0, min(y1, height)))
            x2_px = int(max(0, min(x2, width)))
            y2_px = int(max(0, min(y2, height)))

            # Ensure valid crop region
            if x2_px <= x1_px or y2_px <= y1_px:
                x2_px = min(x1_px + 10, width)
                y2_px = min(y1_px + 10, height)

            # Crop the image region for identification
            cropped_region = image.crop((x1_px, y1_px, x2_px, y2_px))

            # Mock identification - in real scenario, this would call a recognition model
            return {"type": "text", "content": "Theo"}

        return {"type": "text", "content": "Unable to identify - invalid coordinates"}
