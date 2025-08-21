"""
Ovis2.5 VL Model Inference Code
This implementation uses the custom modeling file instead of transformers AutoModelForCausalLM
"""

from typing import List, Union

import requests
import torch
from moviepy import VideoFileClip
from PIL import Image
from transformers import TextIteratorStreamer

# from Ovis.HF_Repo.configuration_ovis2_5 import Ovis2_5_Config

# Instead of using: from transformers import AutoModelForCausalLM
# We'll use the custom modeling file
from Ovis.HF_Repo.modeling_ovis2_5 import Ovis2_5


class BudgetAwareTextStreamer(TextIteratorStreamer):
    """
    A streamer compatible with Ovis two-phase generation.
    Call .manual_end() after generation to flush any remaining text.
    """

    def manual_end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""
        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    # Disable base class's end hook; we'll finalize via manual_end()
    def end(self):
        pass


class Ovis25Inference:
    """
    Comprehensive inference wrapper for Ovis2.5 VL model
    """

    def __init__(
        self,
        model_path: str = "AIDC-AI/Ovis2.5-9B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the Ovis2.5 model

        Args:
            model_path: Path to the model (9B or 2B version)
            device: Device to run the model on
            torch_dtype: Torch dtype for inference
        """
        self.device = device
        self.torch_dtype = torch_dtype

        # Load model using custom implementation
        print("Loading Ovis2.5 model...")
        self.model = Ovis2_5.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device)

        self.text_tokenizer = self.model.text_tokenizer
        print(f"Model loaded successfully on {device}")

    def single_image_inference(
        self,
        image_input: Union[str, Image.Image],
        text_prompt: str,
        enable_thinking: bool = False,
        enable_thinking_budget: bool = False,
        thinking_budget: int = 2048,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        do_sample: bool = True,
        min_pixels: int = 448 * 448,
        max_pixels: int = 1792 * 1792,
    ) -> str:
        """
        Perform single image inference

        Args:
            image_input: Path to image or PIL Image object
            text_prompt: Text prompt for the image
            enable_thinking: Whether to enable thinking mode
            enable_thinking_budget: Whether to limit thinking tokens
            thinking_budget: Maximum tokens for thinking phase (default: 2048)
            max_new_tokens: Maximum new tokens to generate (default: 1024)
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            min_pixels: Minimum image pixels
            max_pixels: Maximum image pixels

        Returns:
            Generated response string
        """
        # Validate thinking budget constraint
        if enable_thinking and enable_thinking_budget:
            if max_new_tokens <= thinking_budget:
                max_new_tokens = thinking_budget + min(1024, max_new_tokens)
            # if max_new_tokens <= thinking_budget + 25:
            #     raise ValueError(f"max_new_tokens ({max_new_tokens}) must be > thinking_budget + 25 ({thinking_budget + 25})")

        # Load image
        if isinstance(image_input, str):
            if image_input.startswith("http"):
                image = Image.open(requests.get(image_input, stream=True).raw)
            else:
                image = Image.open(image_input)
        else:
            image = image_input

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        # Preprocess inputs
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # Move to device
        input_ids = input_ids.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if grid_thws is not None:
            grid_thws = grid_thws.to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                enable_thinking=enable_thinking,
                enable_thinking_budget=enable_thinking_budget,
                thinking_budget=thinking_budget,
                eos_token_id=self.text_tokenizer.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
            )

        # Decode response
        response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def single_image_inference_streaming(
        self,
        image_input: Union[str, Image.Image],
        text_prompt: str,
        enable_thinking: bool = False,
        enable_thinking_budget: bool = False,
        thinking_budget: int = 2048,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        min_pixels: int = 448 * 448,
        max_pixels: int = 1792 * 1792,
    ):
        """
        Perform single image inference with streaming output

        Args:
            Same as single_image_inference

        Yields:
            Streaming text tokens
        """
        # Validate thinking budget constraint
        if enable_thinking and enable_thinking_budget:
            if max_new_tokens <= thinking_budget:
                max_new_tokens = thinking_budget + min(1024, max_new_tokens)
            # if max_new_tokens <= thinking_budget + 25:
            #     raise ValueError(f"max_new_tokens ({max_new_tokens}) must be > thinking_budget + 25 ({thinking_budget + 25})")

        # Load image
        if isinstance(image_input, str):
            if image_input.startswith("http"):
                image = Image.open(requests.get(image_input, stream=True).raw)
            else:
                image = Image.open(image_input)
        else:
            image = image_input

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        # Preprocess inputs
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # Move to device
        input_ids = input_ids.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if grid_thws is not None:
            grid_thws = grid_thws.to(self.device)

        # Setup streamer
        streamer = BudgetAwareTextStreamer(
            self.text_tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Generate response with streaming
        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                enable_thinking=enable_thinking,
                enable_thinking_budget=enable_thinking_budget,
                thinking_budget=thinking_budget,
                eos_token_id=self.text_tokenizer.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                streamer=streamer,
            )

        # Manual end for streamer
        streamer.manual_end()
        return outputs

    def multi_image_inference(
        self,
        image_paths: List[Union[str, Image.Image]],
        text_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        do_sample: bool = True,
        max_pixels: int = 896 * 896,
    ) -> str:
        """
        Perform multi-image inference

        Args:
            image_paths: List of image paths or PIL Image objects
            text_prompt: Text prompt for the images
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            max_pixels: Maximum image pixels (lower for multi-image)

        Returns:
            Generated response string
        """
        # Load images
        images = []
        for img_input in image_paths:
            if isinstance(img_input, str):
                if img_input.startswith("http"):
                    image = Image.open(requests.get(img_input, stream=True).raw)
                else:
                    image = Image.open(img_input)
            else:
                image = img_input
            images.append(image.convert("RGB"))

        # Prepare content
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": text_prompt})

        messages = [{"role": "user", "content": content}]

        # Preprocess and generate
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages, add_generation_prompt=True, max_pixels=max_pixels
        )

        input_ids = input_ids.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if grid_thws is not None:
            grid_thws = grid_thws.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                eos_token_id=self.text_tokenizer.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
            )

        response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def video_inference(
        self,
        video_path: str,
        text_prompt: str,
        num_frames: int = 8,
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        do_sample: bool = True,
        max_pixels: int = 896 * 896,
    ) -> str:
        """
        Perform video inference by sampling frames

        Args:
            video_path: Path to video file
            text_prompt: Text prompt for the video
            num_frames: Number of frames to sample
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            max_pixels: Maximum image pixels per frame

        Returns:
            Generated response string
        """
        # Sample frames from video
        with VideoFileClip(video_path) as clip:
            total_frames = int(clip.fps * clip.duration)
            indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            frames = [
                Image.fromarray(clip.get_frame(t)) for t in (idx / clip.fps for idx in indices)
            ]

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        # Preprocess and generate
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages, add_generation_prompt=True, max_pixels=max_pixels
        )

        input_ids = input_ids.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if grid_thws is not None:
            grid_thws = grid_thws.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                eos_token_id=self.text_tokenizer.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
            )

        response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def text_only_inference(
        self,
        text_prompt: str,
        max_new_tokens: int = 1024,  # Note: Text-only can use smaller max_new_tokens
        temperature: float = 0.6,
        do_sample: bool = True,
        enable_thinking: bool = False,
        enable_thinking_budget: bool = False,
        thinking_budget: int = 2048,
    ) -> str:  # Fixed: was 1024
        """
        Perform text-only inference

        Args:
            text_prompt: Text prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            enable_thinking: Whether to enable thinking mode
            enable_thinking_budget: Whether to limit thinking tokens
            thinking_budget: Maximum tokens for thinking phase

        Returns:
            Generated response string
        """
        # Validate thinking budget constraint
        if enable_thinking and enable_thinking_budget:
            if max_new_tokens <= thinking_budget:
                max_new_tokens = thinking_budget + min(1024, max_new_tokens)
            # if max_new_tokens <= thinking_budget + 25:
            #     raise ValueError(f"max_new_tokens ({max_new_tokens}) must be > thinking_budget + 25 ({thinking_budget + 25})")

        messages = [{"role": "user", "content": text_prompt}]

        input_ids, _, _ = self.model.preprocess_inputs(
            messages=messages, add_generation_prompt=True, enable_thinking=enable_thinking
        )

        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                enable_thinking=enable_thinking,
                enable_thinking_budget=enable_thinking_budget,
                thinking_budget=thinking_budget,
                eos_token_id=self.text_tokenizer.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
            )

        response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def grounding_inference(
        self, image_input: Union[str, Image.Image], text_prompt: str, request_type: str = "box"
    ) -> str:
        """
        Perform visual grounding to get bounding boxes or points

        Args:
            image_input: Path to image or PIL Image object
            text_prompt: Text prompt with grounding request
            request_type: "box" for bounding boxes, "point" for points

        Returns:
            Generated response with coordinates
        """
        # Add grounding suffix to prompt
        if text_prompt[-1] != ".":
            text_prompt += "."
        if request_type == "box":
            prompt_with_grounding = f"{text_prompt} Please provide the bounding box coordinates."
        else:
            prompt_with_grounding = f"{text_prompt} Please provide the point coordinates."

        return self.single_image_inference(
            image_input=image_input,
            text_prompt=prompt_with_grounding,
            temperature=0.0,  # Use deterministic generation for grounding
            do_sample=False,
        )


# Example usage and demonstrations
def main():
    """
    Demonstration of Ovis2.5 capabilities
    """
    # Initialize model
    print("Initializing Ovis2.5 model...")
    ovis = Ovis25Inference(model_path="AIDC-AI/Ovis2.5-9B")

    # Example 1: Single image analysis with standard parameters
    print("\n=== Single Image Analysis ===")
    image_url = "https://cdn-uploads.huggingface.co/production/uploads/658a8a837959448ef5500ce5/TIlymOb86R6_Mez3bpmcB.png"
    response = ovis.single_image_inference(
        image_input=image_url,
        text_prompt="What do you see in this image? Describe it in detail.",
        enable_thinking=False,
        max_new_tokens=1024,
    )
    print(f"Response: {response}")

    # Example 2: Mathematical reasoning with thinking mode and budget
    print("\n=== Mathematical Reasoning with Thinking Mode ===")
    math_response = ovis.single_image_inference(
        image_input=image_url,
        text_prompt="Calculate the sum of the numbers in the middle box in figure (c).",
        enable_thinking=True,
        enable_thinking_budget=True,
        thinking_budget=2048,  # Fixed: Using correct default
        max_new_tokens=3072,  # Fixed: Must be > thinking_budget + 25
    )
    print(f"Math Response: {math_response}")

    # Example 3: Thinking mode without budget (for complex reasoning)
    print("\n=== Complex Reasoning without Budget Limit ===")
    complex_response = ovis.single_image_inference(
        image_input=image_url,
        text_prompt="Analyze this image step by step and provide detailed insights about its composition, mathematical elements, and educational purpose.",
        enable_thinking=True,
        enable_thinking_budget=False,  # No budget limit
        max_new_tokens=4096,  # Higher for complex reasoning
    )
    print(f"Complex Response: {complex_response}")

    # Example 4: Multi-image analysis
    print("\n=== Multi-Image Analysis ===")
    # Note: Replace with actual image paths
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]
    # multi_response = ovis.multi_image_inference(
    #     image_paths=image_paths,
    #     text_prompt="Compare these images and describe the differences.",
    #     max_new_tokens=3072  # Using correct default
    # )
    # print(f"Multi-image Response: {multi_response}")

    # Example 5: Text-only conversation with thinking
    print("\n=== Text-Only Conversation with Thinking ===")
    text_response = ovis.text_only_inference(
        text_prompt="Explain the concept of artificial intelligence in simple terms.",
        enable_thinking=True,
        enable_thinking_budget=True,
        thinking_budget=2048,
        max_new_tokens=3072,
    )
    print(f"Text Response: {text_response}")

    # Example 6: Visual grounding
    print("\n=== Visual Grounding ===")
    grounding_response = ovis.grounding_inference(
        image_input=image_url,
        text_prompt="Find the <ref>mathematical equations</ref> in the image.",
        request_type="box",
    )
    print(f"Grounding Response: {grounding_response}")

    # Example 7: OCR and document analysis
    print("\n=== OCR and Document Analysis ===")
    ocr_response = ovis.single_image_inference(
        image_input=image_url,
        text_prompt="Extract all text from this image and organize it in a structured format.",
        max_new_tokens=1024,
    )
    print(f"OCR Response: {ocr_response}")


if __name__ == "__main__":
    # Installation requirements
    requirements = """
    Required installations:
    pip install torch==2.4.0 transformers==4.51.3 numpy==1.25.0 pillow==10.3.0 moviepy==1.0.3
    pip install flash-attn==2.7.0.post2 --no-build-isolation
    
    Also download the modeling files:
    - modeling_ovis2_5.py
    - configuration_ovis2_5.py
    """
    print(requirements)

    # Run examples
    main()
