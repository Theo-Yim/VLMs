"""
Inference for Ovis2.5-9B
Based on official guide: https://huggingface.co/AIDC-AI/Ovis2.5-9B
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from config import InferenceConfig
from data_utils import GroundingParser
from PIL import Image
from transformers import AutoModelForCausalLM

sys.path.append("..")
from crop_tool import CropTool, parse_and_replace_tool_calls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OvisInference:
    """
    Inference class for Ovis2.5-9B
    Handles image-text generation with thinking mode and grounding
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = config.device
        self.grounding_parser = GroundingParser()
        self.crop_tool = CropTool()  # Use existing CropTool from QwenVL2.5

        # Load model
        self.load_model()

    def load_model(self):
        """Load Ovis2.5-9B model"""
        logger.info(f"Loading Ovis2.5-9B model from: {self.config.model_path}")

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto" if self.device == "cuda" else "cpu",
        }

        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)

        logger.info("Model loaded successfully")

    def prepare_messages(
        self,
        image: Union[Image.Image, str, Path],
        question: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        """Prepare messages in Ovis format"""

        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Prepare content
        content = [{"type": "image", "image": image}, {"type": "text", "text": question}]

        messages = [{"role": "user", "content": content}]

        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return messages

    def generate(
        self,
        image: Union[Image.Image, str, Path],
        question: str,
        system_prompt: Optional[str] = None,
        **generation_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate response for image-question pair
        Returns dict with generated text and parsed elements
        """

        # Prepare messages
        messages = self.prepare_messages(image, question, system_prompt)

        # Merge generation kwargs with config defaults
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample,
            "enable_thinking": self.config.enable_thinking,
            "enable_thinking_budget": self.config.enable_thinking_budget,
            "thinking_budget": self.config.thinking_budget,
        }
        gen_kwargs.update(generation_kwargs)

        try:
            # Preprocess inputs using Ovis method
            input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=gen_kwargs["enable_thinking"],
                max_pixels=self.config.max_pixels,
            )

            # Move to device
            if self.device == "cuda":
                input_ids = input_ids.cuda()
                pixel_values = pixel_values.cuda() if pixel_values is not None else None
                grid_thws = grid_thws.cuda() if grid_thws is not None else None

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    **{k: v for k, v in gen_kwargs.items() if k not in ["enable_thinking"]},
                )

            # Decode response using model's text tokenizer
            response = self.model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse response
            parsed_response = self.parse_response(response)

            return {
                "response": response,
                "question": question,
                "parsed": parsed_response,
                "generation_kwargs": gen_kwargs,
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {"response": "", "question": question, "parsed": {}, "error": str(e)}

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Ovis response for thinking content, answer, grounding, and tool calls
        """
        parsed = {}

        # Extract thinking content
        if self.config.extract_think_answer:
            think_pattern = r"<think>(.*?)</think>"
            answer_pattern = r"<answer>(.*?)</answer>"

            think_match = re.search(think_pattern, response, re.DOTALL)
            answer_match = re.search(answer_pattern, response, re.DOTALL)

            parsed["think_content"] = think_match.group(1).strip() if think_match else ""
            parsed["answer_content"] = answer_match.group(1).strip() if answer_match else response
        else:
            parsed["answer_content"] = response

        # Parse tool calls (if present)
        if self.config.parse_tool_calls:
            tool_calls = self.crop_tool.extract_tool_calls(response)
            parsed["tool_calls"] = tool_calls
            parsed["has_tool_calls"] = bool(tool_calls)

        # Parse grounding elements (Ovis format)
        if self.config.parse_grounding:
            grounding = self.grounding_parser.parse_grounding(response)
            parsed["grounding"] = grounding
            parsed["has_grounding"] = bool(
                grounding["refs"] or grounding["boxes"] or grounding["points"]
            )

        return parsed

    def batch_generate(
        self,
        images: List[Union[Image.Image, str, Path]],
        questions: List[str],
        system_prompt: Optional[str] = None,
        **generation_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple image-question pairs
        """
        results = []

        for image, question in zip(images, questions):
            result = self.generate(
                image=image, question=question, system_prompt=system_prompt, **generation_kwargs
            )
            results.append(result)

        return results

    def generate_with_grounding(
        self,
        image: Union[Image.Image, str, Path],
        object_description: str,
        grounding_type: str = "box",
        **generation_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate response with explicit grounding request
        """

        if grounding_type == "box":
            question = f"Find the <ref>{object_description}</ref> in the image. Please provide the bounding box coordinates."
        elif grounding_type == "point":
            question = f"Find the <ref>{object_description}</ref> in the image. Please provide the point coordinates."
        else:
            raise ValueError("grounding_type must be 'box' or 'point'")

        return self.generate(image=image, question=question, **generation_kwargs)


def main():
    """Example usage"""

    # Initialize inference
    config = InferenceConfig(
        model_path="AIDC-AI/Ovis2.5-9B",  # Use base model for demo
        max_new_tokens=1024,
        enable_thinking=True,
        thinking_budget=512,
    )

    inference = OvisInference(config)

    # Example 1: Basic inference
    try:
        # Create a simple test image
        test_image = Image.new("RGB", (224, 224), color="blue")

        result = inference.generate(image=test_image, question="What color is this image?")

        print("Generated response:")
        print(result["response"])

        if result["parsed"]["think_content"]:
            print("\nThinking process:")
            print(result["parsed"]["think_content"])

        print("\nFinal answer:")
        print(result["parsed"]["answer_content"])

    except Exception as e:
        print(f"Example failed: {e}")


if __name__ == "__main__":
    main()
