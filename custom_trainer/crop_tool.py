"""
Crop tool implementation for Qwen 2.5 VL
Handles image cropping based on tool call coordinates
"""

import re
from typing import List, Dict, Any
from PIL import Image
import numpy as np


class CropTool:
    """
    Tool for cropping image regions based on coordinates
    """
    
    def __init__(self):
        self.tool_call_pattern = r'<tool_call>Crop \[([0-9.,\s]+)\]</tool_call>'
        
    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all tool calls from text
        Returns list of tool calls with their positions and coordinates
        """
        tool_calls = []
        
        for match in re.finditer(self.tool_call_pattern, text):
            coords_str = match.group(1)
            coords = [float(x.strip()) for x in coords_str.split(',')]
            
            if len(coords) == 4:
                tool_calls.append({
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'full_match': match.group(0),
                    'coordinates': coords,
                    'bbox': {
                        'x1': coords[0],
                        'y1': coords[1],
                        'x2': coords[2],
                        'y2': coords[3],
                    }
                })
        
        return tool_calls
    
    def crop_image(
        self,
        image: Image.Image,
        bbox: List[float],
        return_pil: bool = True
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
    
    def process_text_with_crops(
        self,
        text: str,
        image: Image.Image,
        processor=None,
        return_all_crops: bool = False
    ) -> Dict[str, Any]:
        """
        Process text containing tool calls and perform crops
        
        Args:
            text: Text containing tool calls
            image: Original image to crop from
            processor: Optional processor to prepare cropped images
            return_all_crops: Whether to return all cropped images
        
        Returns:
            Dictionary containing:
            - processed_text: Text with tool calls processed
            - cropped_images: List of cropped images
            - tool_calls: List of tool call information
            - segments: Text segments and associated images
        """
        tool_calls = self.extract_tool_calls(text)
        
        if not tool_calls:
            return {
                'processed_text': text,
                'cropped_images': [],
                'tool_calls': [],
                'segments': [{'text': text, 'image': None}]
            }
        
        cropped_images = []
        segments = []
        last_pos = 0
        
        for tool_call in tool_calls:
            # Add text before tool call
            if last_pos < tool_call['start_pos']:
                segments.append({
                    'text': text[last_pos:tool_call['start_pos']],
                    'image': None,
                    'type': 'text'
                })
            
            # Crop the image
            cropped = self.crop_image(image, tool_call['coordinates'])
            cropped_images.append(cropped)
            
            # Add tool call segment with cropped image
            segments.append({
                'text': tool_call['full_match'],
                'image': cropped,
                'type': 'tool_call',
                'bbox': tool_call['bbox']
            })
            
            last_pos = tool_call['end_pos']
        
        # Add remaining text after last tool call
        if last_pos < len(text):
            segments.append({
                'text': text[last_pos:],
                'image': None,
                'type': 'text'
            })
        
        return {
            'processed_text': text,
            'cropped_images': cropped_images if return_all_crops else [],
            'tool_calls': tool_calls,
            'segments': segments
        }
    
    def format_for_training(
        self,
        text: str,
        image: Image.Image,
        processor=None
    ) -> List[Dict[str, Any]]:
        """
        Format text with tool calls for training
        Creates a sequence of text and image inputs
        
        Returns:
            List of content items for training, where each item is either:
            - {'type': 'text', 'text': str}
            - {'type': 'image', 'image': PIL.Image}
        """
        result = self.process_text_with_crops(text, image)
        
        content = []
        for segment in result['segments']:
            if segment['type'] == 'text' and segment['text']:
                content.append({
                    'type': 'text',
                    'text': segment['text']
                })
            elif segment['type'] == 'tool_call':
                # Add the tool call text
                content.append({
                    'type': 'text',
                    'text': segment['text']
                })
                # Add the cropped image right after
                if segment['image'] is not None:
                    content.append({
                        'type': 'image',
                        'image': segment['image']
                    })
        
        return content
    
    def execute_and_continue(
        self,
        text: str,
        image: Image.Image,
        model=None,
        processor=None,
        max_iterations: int = 5
    ) -> str:
        """
        Execute tool calls and continue generation
        This simulates the interactive process during inference
        
        Args:
            text: Initial generated text with tool calls
            image: Original image
            model: Model for continued generation
            processor: Processor for preparing inputs
            max_iterations: Maximum number of tool call iterations
        
        Returns:
            Final generated text with all tool calls executed
        """
        if model is None or processor is None:
            # If no model provided, just return the text
            return text
        
        current_text = text
        iteration = 0
        
        while iteration < max_iterations:
            tool_calls = self.extract_tool_calls(current_text)
            
            if not tool_calls:
                # No more tool calls, we're done
                break
            
            # Process the first tool call
            first_call = tool_calls[0]
            
            # Get text up to and including the tool call
            prefix = current_text[:first_call['end_pos']]
            
            # Crop the image
            cropped = self.crop_image(image, first_call['coordinates'])
            
            # Prepare input for continued generation
            # The model should continue from where it left off, with the cropped image
            messages = [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": prefix},
                        {"type": "image", "image": cropped}
                    ]
                }
            ]
            
            # Continue generation
            # This is a simplified version - actual implementation would need proper model integration
            # For now, we'll just append a placeholder continuation
            continuation = f" [Processed crop at {first_call['bbox']}]"
            current_text = prefix + continuation
            
            iteration += 1
        
        return current_text


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
    pattern = r'\{Crop[^}]*\[([\d.,\s]+)\]\}'
    
    def replace_func(match):
        coords = match.group(1)
        return f"<tool_call>Crop [{coords}]</tool_call>"
    
    return re.sub(pattern, replace_func, text)