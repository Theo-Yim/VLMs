"""
Tool system base classes for Ovis2.5
"""

import importlib
import inspect
import logging
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image


class ToolBase:
    """
    Base class for all tools.

    To create a new tool, inherit from this class and implement:
    - extract_tool_call(text) - for inference detection
    - extract_tool_calls(text) - for training detection
    - execute(image, parameters) - to perform the tool action
    """

    def extract_tool_call(self, text: str) -> Optional[Dict]:
        """
        Extract a single tool call from text (for inference).

        Args:
            text: Text that may contain a tool call

        Returns:
            Dict with tool call info, or None if not found
            Format: {
                "parameters": [...],
                "tool_name": "tool_name",
                "tool_instance": self
            }
        """
        raise NotImplementedError("Tools must implement extract_tool_call()")

    def extract_tool_calls(self, text: str) -> List[Dict]:
        """
        Extract all tool calls from text (for training detection).

        Args:
            text: Text that may contain multiple tool calls

        Returns:
            List of dicts with tool call info
            Format: [{
                "start_pos": int,
                "end_pos": int,
                "full_match": str,
                "parameters": [...]
            }, ...]
        """
        return []  # Optional - default to empty list

    def execute(self, image: Optional[Image.Image], parameters) -> Optional[Dict]:
        """
        Execute the tool with given parameters.

        Args:
            image: PIL Image (can be None for text-only tools)
            parameters: Tool-specific parameters

        Returns:
            Result dict or None
            Format: {"type": "image"|"text", "content": ...}
        """
        raise NotImplementedError("Tools must implement execute()")


class ToolRegistry:
    """
    Universal registry for handling multiple tool types.
    Supports both inference and training with ANY tool.
    """

    def __init__(self):
        self.tools = {}
        self._setup_available_tools()

    def _setup_available_tools(self):
        """
        Auto-detect and load all tools from tools/ directory.

        Scans for *_tool.py files and automatically imports/registers any ToolBase subclasses.
        This enables zero-configuration tool addition - just drop a new *_tool.py file!
        """

        # Get the directory where tool_base.py is located
        tools_dir = os.path.dirname(__file__)

        # Find all *_tool.py files
        for filename in os.listdir(tools_dir):
            if filename.endswith("_tool.py") and filename != "tool_base.py":
                module_name = filename[:-3]  # Remove .py extension

                try:
                    # Import the module dynamically
                    module = importlib.import_module(f"src_theo.tools.{module_name}")

                    # Find all ToolBase subclasses in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Check if it's a ToolBase subclass (but not ToolBase itself)
                        if issubclass(obj, ToolBase) and obj is not ToolBase:
                            # Instantiate and register the tool
                            tool_instance = obj()
                            # Tool name: remove '_tool' suffix from filename
                            tool_name = module_name.replace("_tool", "")
                            self.register_tool(tool_name, tool_instance)
                            logging.info(f"Registered tool: {tool_name} from {filename}")

                except Exception as e:
                    logging.warning(f"Failed to load tool from {filename}: {e}")

    def register_tool(self, tool_name: str, tool_instance):
        """Register a tool instance"""
        self.tools[tool_name] = tool_instance

    def detect_tool_calls(self, text: str) -> bool:
        """Check if text contains any supported tool calls"""
        if not text or not isinstance(text, str):
            return False

        for tool_instance in self.tools.values():
            if hasattr(tool_instance, "extract_tool_calls"):
                if tool_instance.extract_tool_calls(text):
                    return True
        return False

    def process_tools_for_training(
        self, text: str, image: Optional[Image.Image]
    ) -> Tuple[List[Image.Image], str]:
        """
        UNIVERSAL tool processing for training.

        Handles TWO types of markers:
        - <image> for image-returning tools
        - <tool_response> for text-returning tools

        Args:
            text: Assistant response with tool calls and markers
            image: Original image (for tools that need it)

        Returns:
            (result_images, cleaned_text) tuple
        """
        result_images = []
        cleaned_text = text

        # Process each registered tool
        for tool_instance in self.tools.values():
            if not hasattr(tool_instance, "extract_tool_calls"):
                continue

            # Find all calls for this tool
            tool_calls = tool_instance.extract_tool_calls(text)

            for tc in tool_calls:
                # Execute the tool
                result = tool_instance.execute(image, tc["parameters"])

                if result is None:
                    continue

                # Route by result type
                if result["type"] == "image":
                    # Image tool: extract image, remove <image> marker
                    # Note: full_match includes <image> marker from the pattern
                    result_images.append(result["content"])

                    # Replace the full match (which includes <image>) with just the tool call
                    # Extract just the tool call part (without <image>)
                    tool_call_only = tc["full_match"].replace("<image>", "")
                    cleaned_text = cleaned_text.replace(tc["full_match"], tool_call_only, 1)

                elif result["type"] == "text":
                    # Text tools: keep <tool_response> in text (already in training data)
                    pass

        return result_images, cleaned_text

    # ========== Inference Methods ==========

    def get_system_prompt_tools(self) -> str:
        """
        Generate system prompt describing available tools.
        """
        if not self.tools:
            return ""

        # Extract tool descriptions from docstrings
        tool_descriptions = {}
        for tool_name, tool_instance in self.tools.items():
            if tool_instance.__doc__:
                tool_descriptions[tool_name] = tool_instance.__doc__.strip()

        if not tool_descriptions:
            return ""

        # Build system prompt
        tool_prompt = "You have access to the following tools for enhanced analysis:\n\n"

        for i, (name, desc) in enumerate(tool_descriptions.items(), 1):
            tool_prompt += f"**Tool {i}: {name.capitalize()}**\n{desc}\n\n"

        tool_prompt += (
            "Important: When calling a tool, strictly follow this format:\n"
            "<tool_call>ToolName [parameters]</tool_call>\n\n"
            "The tool will execute and provide results to enhance your response."
        )

        return tool_prompt

    def detect_tool_call(self, text: str) -> Optional[Dict]:
        """
        Detect single tool call during generation (for inference).

        Args:
            text: Generated text that may contain a tool call

        Returns:
            Tool call dict or None
        """
        if not text or not isinstance(text, str):
            return None

        # Check all tools - return first match
        for tool_instance in self.tools.values():
            if hasattr(tool_instance, "extract_tool_call"):
                tool_call = tool_instance.extract_tool_call(text)
                if tool_call:
                    return tool_call

        return None

    def execute_tool_call(
        self, tool_call: Dict, original_image: Optional[Image.Image]
    ) -> Optional[Dict]:
        """
        Execute a tool call and return result.

        Args:
            tool_call: Tool call dict from detect_tool_call()
            original_image: Original image for the tool

        Returns:
            {"type": "image"|"text", "content": ...} or None
        """
        tool_instance = tool_call["tool_instance"]
        return tool_instance.execute(original_image, tool_call["parameters"])
