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
        self.tool_call_pattern = r"<tool_call>Identify \[([0-9.,\s]+)\]</tool_call>"

    def extract_tool_call(self, text: str) -> Dict[str, Any]:
        """Optimized extraction for single tool call.
        Returns parameters only since that's all we need now.
        """
        match = re.search(self.tool_call_pattern, text)
        if match:
            coords_str = match.group(1)
            coords = [float(x.strip()) for x in coords_str.split(",")]
            if len(coords) == 4:
                return {
                    "parameters": [coords],
                    "tool_name": "identify",  # Each tool manages its own name
                    "tool_instance": self,  # Each tool manages its own instance
                }
        return {}

    def execute(self, image: Optional[Image.Image], parameters: Any) -> Optional[dict]:
        self.identify(image, parameters[0])

    def identify(self, image: Image.Image, bbox: List[float]):
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

            # Crop the image region
            cropped_region = image.crop((x1_px, y1_px, x2_px, y2_px))

            print(f"Identifying region: [{x1}, {y1}, {x2}, {y2}]")
            print(f"Cropped region size: {cropped_region.size}")

            return {"type": "text", "content": "This person's name is Theo"}

        return {"type": "text", "content": "Unable to identify - invalid coordinates"}
