import importlib
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

# Set up logger for this module
logger = logging.getLogger(__name__)


class ToolBase:
    def __init__(self):
        self.tool_call_pattern = ""

    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    def extract_tool_call(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError()

    def execute(self, input: Any) -> Any:
        raise NotImplementedError()


class ToolRegistry:
    """Registry for handling multiple tool types with auto-detection"""

    def __init__(self):
        self.tools = {}
        self.tool_descriptions = {}
        self._setup_available_tools()

    def _setup_available_tools(self):
        """Auto-detect and initialize available tools"""
        # Get the directory of this module
        tools_dir = Path(os.path.dirname(__file__))

        # Scan for *_tool.py files
        tool_files = list(tools_dir.glob("*_tool.py"))

        for file in tool_files:
            if file.name in ["tool_base.py"]:
                continue

            # Import with absolute module path
            module_name = f"src_theo.tools.{file.stem}"

            try:
                module = importlib.import_module(module_name)

                # Find ToolBase subclasses in the module
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and hasattr(obj, "__bases__")
                        and any(base.__name__ == "ToolBase" for base in obj.__bases__)
                    ):
                        # Generate tool name from class name
                        tool_name = name.lower().replace("tool", "")

                        # Register the tool
                        self.register_tool(tool_name, obj())
                        logger.info(f"Auto-registered tool: {tool_name} ({name})")

            except ImportError as e:
                logger.warning(f"Tool {file.stem} not available: {e}")
            except Exception as e:
                logger.error(f"Error loading tool {file.stem}: {e}")

    def register_tool(self, tool_name: str, tool_instance):
        """Register a tool instance"""
        self.tools[tool_name] = tool_instance
        self.tool_descriptions[tool_name] = tool_instance.__doc__

    def get_system_prompt_tools(self) -> str:
        """Generate tool descriptions for system prompt"""
        if not self.tool_descriptions:
            return ""

        tool_prompt = "You have tools. Look at the tool descriptions below and use the tool if it is relevant to the task.\n"
        for i, (name, desc) in enumerate(self.tool_descriptions.items()):
            tool_prompt += f"### Tool {i + 1}: {name}\n{desc}\n"
        tool_prompt += "When you call any tool, strictly follow the tool calling format: <tool_call>ToolName parameters</tool_call>, as shown in Usages."

        return tool_prompt

    def detect_tool_call(self, text: str) -> Dict:
        """
        Detection for single tool call (used during generation).
        Assumes text contains exactly one complete tool call.
        Returns the tool call dict or None if not found.
        """
        if not text or not isinstance(text, str):
            return None

        # Check all tools generically - return first match
        for tool_name, tool_instance in self.tools.items():
            # Use optimized single extraction if available
            if hasattr(tool_instance, "extract_tool_call"):
                tool_call = tool_instance.extract_tool_call(text)
                if tool_call:
                    return tool_call

        return None

    def execute_tool_call(
        self, tool_call: Dict, original_image: Optional[Image.Image]
    ) -> Optional[dict]:
        """Execute a single tool call and return structured result or None"""
        tool_instance = tool_call["tool_instance"]

        # Tools return structured format: {"type": "image|text|multimodal", "content": ...} or None
        return tool_instance.execute(original_image, tool_call["parameters"])
