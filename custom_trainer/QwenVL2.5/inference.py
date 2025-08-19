"""
Inference script for Qwen 2.5 VL with tool call parsing
"""

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from config import InferenceConfig, ModelConfig
from data_utils import resize_crop_smart, resize_image_shortest_side
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

sys.path.append("..")
from crop_tool import CropTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for inference results"""

    raw_output: str
    think_content: Optional[str] = None
    answer_content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = None
    regions: List[List[float]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            "raw_output": self.raw_output,
            "think_content": self.think_content,
            "answer_content": self.answer_content,
            "tool_calls": self.tool_calls,
            "regions": self.regions,
        }


class QwenVLInference:
    """Inference class for Qwen 2.5 VL with crop tool support"""

    def __init__(self, config: InferenceConfig, model_config: ModelConfig = None):
        self.config = config
        self.model_config = model_config or ModelConfig()
        self.device = torch.device(config.device)
        self.crop_tool = CropTool()

        # Load model and processor
        self.load_model_and_processor()

    def load_model_and_processor(self):
        """Load model and processor for inference"""
        logger.info(f"Loading model from: {self.config.model_path}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )

        # Model loading configuration
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }

        # Check if this is a LoRA model
        lora_path = Path(self.config.model_path) / "adapter_model.bin"
        if lora_path.exists():
            # Load base model
            base_model = AutoModelForVision2Seq.from_pretrained(
                self.model_config.model_name, **model_kwargs
            )
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(
                base_model, self.config.model_path, is_trainable=False
            )
            self.model = self.model.merge_and_unload()
        else:
            # Load full model
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_path, **model_kwargs
            )

        self.model.eval()
        logger.info("Model loaded successfully")

    def parse_output(self, text: str) -> InferenceResult:
        """
        Parse model output to extract think content, answer, and tool calls
        """
        result = InferenceResult(raw_output=text)

        # Extract think content
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, text, re.DOTALL)
        if think_match:
            result.think_content = think_match.group(1).strip()

        # Extract answer content
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_match = re.search(answer_pattern, text, re.DOTALL)
        if answer_match:
            result.answer_content = answer_match.group(1).strip()

        # Extract and parse tool calls using CropTool
        if self.config.parse_tool_calls:
            tool_calls_data = self.crop_tool.extract_tool_calls(text)

            tool_calls = []
            regions = []

            for tc in tool_calls_data:
                tool_calls.append(
                    {"type": "crop", "coordinates": tc["coordinates"], "bbox": tc["bbox"]}
                )
                regions.append(tc["coordinates"])

            result.tool_calls = tool_calls
            result.regions = regions

        return result

    def generate_with_tool_execution(
        self, image: Image.Image, question: str, max_iterations: int = 3, **generation_kwargs
    ) -> InferenceResult:
        """
        Generate response with automatic crop tool execution
        This simulates the interactive process where crops are executed and fed back

        Args:
            image: PIL Image
            question: Text question
            max_iterations: Maximum number of tool call iterations
            **generation_kwargs: Additional generation parameters

        Returns:
            InferenceResult with complete response including tool executions
        """
        # Initial generation
        result = self.generate(image, question, **generation_kwargs)

        if not result.tool_calls or not self.config.parse_tool_calls:
            return result

        # Process tool calls iteratively
        current_text = result.raw_output
        all_cropped_regions = []

        for iteration in range(max_iterations):
            tool_calls_data = self.crop_tool.extract_tool_calls(current_text)

            if not tool_calls_data:
                break

            # For simplicity, we'll just track the crops but not do multi-turn generation
            # In a real implementation, you'd feed cropped images back to the model
            for tc in tool_calls_data:
                cropped = self.crop_tool.crop_image(image, tc["coordinates"])
                # Apply smart resize to match training behavior
                cropped_resized = resize_crop_smart(cropped, self.config.image_size)
                all_cropped_regions.append(
                    {"iteration": iteration, "bbox": tc["bbox"], "cropped_image": cropped_resized}
                )

            # In production, you would continue generation here with cropped images
            # For now, we break after first iteration
            break

        # Update result with all processed crops
        result.cropped_regions = all_cropped_regions
        return result

    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> InferenceResult:
        """
        Generate response for a given image and question

        Args:
            image: PIL Image
            question: Text question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling

        Returns:
            InferenceResult with parsed output
        """
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        do_sample = do_sample if do_sample is not None else self.config.do_sample

        # Resize image
        image = resize_image_shortest_side(image, self.config.image_size)

        # Prepare messages
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=self.config.num_beams,
            )

        # Decode output
        generated_text = self.processor.decode(
            generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Remove input prompt from output
        if text in generated_text:
            generated_text = generated_text.replace(text, "").strip()

        # Parse output
        result = self.parse_output(generated_text)

        return result

    def batch_generate(
        self, images: List[Image.Image], questions: List[str], **kwargs
    ) -> List[InferenceResult]:
        """
        Generate responses for multiple image-question pairs

        Args:
            images: List of PIL Images
            questions: List of text questions
            **kwargs: Additional generation parameters

        Returns:
            List of InferenceResults
        """
        results = []

        # Process each image-question pair
        for image, question in zip(images, questions):
            result = self.generate(image, question, **kwargs)
            results.append(result)

        return results

    def interactive_generate(self, image_path: str, question: str, **kwargs) -> Dict:
        """
        Interactive generation with file path input

        Args:
            image_path: Path to image file
            question: Text question
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with formatted results
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Generate response
        result = self.generate(image, question, **kwargs)

        # Format output
        output = {
            "image_path": image_path,
            "question": question,
            "response": result.to_dict(),
        }

        # If tool calls were made, execute them
        if result.tool_calls:
            cropped_regions = []
            for tool_call in result.tool_calls:
                if tool_call["type"] == "crop":
                    cropped = self.crop_tool.crop_image(image, tool_call["coordinates"])
                    # Apply smart resize to match training behavior
                    cropped_resized = resize_crop_smart(cropped, self.config.image_size)
                    cropped_regions.append(
                        {
                            "bbox": tool_call["bbox"],
                            "size": cropped_resized.size,
                        }
                    )
            output["cropped_regions"] = cropped_regions

        return output


def main():
    """Main inference entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen 2.5 VL Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum tokens to generate"
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--output_json", type=str, help="Path to save output JSON")

    args = parser.parse_args()

    # Initialize inference config
    config = InferenceConfig(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Initialize inference engine
    inference = QwenVLInference(config)

    # Run inference
    result = inference.interactive_generate(
        image_path=args.image_path,
        question=args.question,
    )

    # Print results
    print("\n" + "=" * 50)
    print("QUESTION:", result["question"])
    print("=" * 50)

    if result["response"]["think_content"]:
        print("\nTHINKING:")
        print(result["response"]["think_content"])

    if result["response"]["answer_content"]:
        print("\nANSWER:")
        print(result["response"]["answer_content"])

    if result["response"]["tool_calls"]:
        print("\nTOOL CALLS:")
        for i, tool_call in enumerate(result["response"]["tool_calls"]):
            print(f"  {i + 1}. Crop region: {tool_call['bbox']}")

    print("\nRAW OUTPUT:")
    print(result["response"]["raw_output"])
    print("=" * 50)

    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
