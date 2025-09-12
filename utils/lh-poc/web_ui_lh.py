import argparse
import json
from typing import Optional, Tuple

import gradio as gr
import PIL.Image
import torch
from ovis.model.modeling_ovis2_5 import Ovis2_5

model: Ovis2_5 = None
optimal_max_pixels: int = None


def get_optimal_max_pixels() -> int:
    """
    Automatically determine optimal max_pixels based on available GPU memory.
    Returns max_pixels value suitable for the current GPU.
    """
    try:
        if torch.cuda.is_available():
            # Get GPU memory in GB
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")
            # Set max_pixels based on GPU memory
            if gpu_memory_gb >= 21:  # >24GB GPUs
                max_pixels = 1792 * 1792  # 3211264
                print(f"Using max_pixels = 1792 * 1792 for high-memory GPU ({gpu_memory_gb:.1f}GB)")
            elif gpu_memory_gb >= 20:  # ~24GB GPUs
                max_pixels = 1168 * 1168  # 1364624
                print(f"Using max_pixels = 1168 * 1168 for 24GB GPU ({gpu_memory_gb:.1f}GB)")
            else:  # <20GB GPUs
                max_pixels = 1024 * 1024  # 1048576
                print(f"Using max_pixels = 1024 * 1024 for ~20GB GPU ({gpu_memory_gb:.1f}GB)")
            return max_pixels
        else:
            print("CUDA not available, using default max_pixels")
            return 1792 * 1792  # 3211264
    except Exception as e:
        print(f"Error detecting GPU memory: {e}, using default max_pixels")
        return 1792 * 1792  # 3211264


def run_single_model_internal(
    image_input: Optional[PIL.Image.Image],
    prompt: str,
    do_sample: bool,
    enable_thinking: bool,
) -> str:
    """Run single model inference using the chat method."""
    max_new_tokens = 2048

    if not image_input:
        gr.Warning("Please upload an image.")
        return ""

    # Prepare vision inputs
    images = [image_input] if image_input else None

    if not enable_thinking:
        prompt = prompt + "\nEnd your response with 'Final answer: ', followed by the json object."
    try:
        # Check if model has custom chat method (for custom checkpoints)
        if hasattr(model, "chat"):
            # Path 1: Use the custom chat method
            response, thinking, _ = model.chat(
                prompt=prompt,
                history=None,  # Always start a new conversation
                images=images,
                videos=None,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                temperature=0.6 if do_sample else 0.0,
                enable_thinking=enable_thinking,
                enable_thinking_budget=enable_thinking,
                max_pixels=optimal_max_pixels,
            )

            # Format output
            if enable_thinking and thinking:
                # return f"**Thinking:**\n```text\n{thinking}\n```\n\n**Response:**\n{response}"
                response = json.dumps(json.loads(response), indent=2, ensure_ascii=False)
                response = (
                    response.replace("space", "공간")
                    .replace("defect_type", "하자유형")
                    .replace("defect_description", "하자내용")
                    .replace("material_part", "부위자재")
                    .replace("location_in_image", "이미지 내 위치")
                    .replace("defect_present", "하자 존재 여부")
                )
                return f"**Response:**\n```json\n{response}\n```"
            else:
                thinking = response[: response.find("Final answer: ") + 13].strip()
                response2 = response[response.find("Final answer: ") + 13 :].strip()
                response2 = json.dumps(json.loads(response2), indent=2, ensure_ascii=False)
                response2 = (
                    response2.replace("space", "공간")
                    .replace("defect_type", "하자유형")
                    .replace("defect_description", "하자내용")
                    .replace("material_part", "부위자재")
                    .replace("location_in_image", "이미지 내 위치")
                    .replace("defect_present", "하자 존재 여부")
                )
                # return f"**Pseudo Thinking (CoT):**\n```text\n{thinking}\n```\n\n**Response:**\n```json\n{response2}\n```"
                return f"**Response:**\n```json\n{response2}\n```"
            # return response

        else:
            # Path 2: Fallback for original pretrained checkpoints
            # Prepare messages based on input type
            if images:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": images[0]},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            # Preprocess inputs
            input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                min_pixels=448 * 448,
                max_pixels=optimal_max_pixels,
            )

            # Move to device
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            if grid_thws is not None:
                grid_thws = grid_thws.to(device)

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    max_new_tokens=max_new_tokens,
                    temperature=0.6 if do_sample else 0.0,
                    do_sample=do_sample,
                    enable_thinking=enable_thinking,
                    enable_thinking_budget=enable_thinking,
                    eos_token_id=model.text_tokenizer.eos_token_id,
                    pad_token_id=model.text_tokenizer.pad_token_id,
                )

            # Decode response
            response_text = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract thinking and response if thinking is enabled
            if enable_thinking and "<think>" in response_text and "</think>" in response_text:
                thinking_start = response_text.find("<think>") + 7
                thinking_end = response_text.find("</think>")
                thinking = response_text[thinking_start:thinking_end].strip()
                response = response_text[thinking_end + 8 :].strip()

                # return f"**Thinking:**\n```text\n{thinking}\n```\n\n**Response:**\n{response}"
                response = (
                    response.replace("space", "공간")
                    .replace("defect_type", "하자유형")
                    .replace("defect_description", "하자내용")
                    .replace("material_part", "부위자재")
                    .replace("location_in_image", "이미지 내 위치")
                    .replace("defect_present", "하자 존재 여부")
                )
                response = json.dumps(json.loads(response), indent=2, ensure_ascii=False)
                return f"**Response:**\n```json\n{response}\n```"
            else:
                # Clean up the response by removing any system tokens
                response = (
                    response_text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
                )
                thinking = response[: response.find("Final answer: ") + 13].strip()
                response2 = response[response.find("Final answer: ") + 13 :].strip()
                response2 = json.dumps(json.loads(response2), indent=2, ensure_ascii=False)
                response2 = (
                    response2.replace("space", "공간")
                    .replace("defect_type", "하자유형")
                    .replace("defect_description", "하자내용")
                    .replace("material_part", "부위자재")
                    .replace("location_in_image", "이미지 내 위치")
                    .replace("defect_present", "하자 존재 여부")
                )
                # return f"**Pseudo Thinking (CoT):**\n```text\n{thinking}\n```\n\n**Response:**\n```json\n{response2}\n```"
                return f"**Response:**\n```json\n{response2}\n```"

    except Exception as e:
        gr.Warning(f"Error during inference: {str(e)}")
        return f"Error: {str(e)}"


def clear_interface() -> Tuple[str, None]:
    """Reset all inputs and outputs."""
    return "", None


def start_generation() -> Tuple[gr.update, gr.update, gr.update]:
    """Update UI status when generation starts."""
    return (
        gr.update(value="⏳ Generating...", interactive=False),
        gr.update(interactive=False),
        gr.update(value="⏳ Model is generating..."),
    )


def finish_generation() -> Tuple[gr.update, gr.update]:
    """Restore UI status after generation ends."""
    return gr.update(value="Generate", interactive=True), gr.update(interactive=True)


def build_demo(model_path: str, gpu: int):
    """Build single-model Gradio demo interface."""
    global model, optimal_max_pixels

    # Define the prompt as a constant
    prompt_input = "<image>\n## You are a professional house construction inspector. Your job is to examine the provided image and determine if there is any defect. You need to guess the space, defect type, defect description, material part, and location in the image of the image.\n\nYou must output the answer in the json format with the following fields:\n- space: [Space name from the list of Spaces]\n- defect_present: Yes / No\n- If Yes, also include:\n  - defect_type: [type from the list of Defect Types]\n  - defect_description: [brief description of the defect]\n  - material_part: [material part from the list of Material Parts]\n  - location_in_image: [describe location within the image, if applicable]\n\n### Instructions\n- Carefully examine each part of the image.\n- Identify the space of the image.\n- Identify the material part of the image.\n- Identify the defect type of the image.\n- Identify the defect description of the image.\n- Identify the location in the image of the image.\n"

    # Create a wrapper function that uses the constant prompt
    def run_model_with_prompt(image_input: Optional[PIL.Image.Image], do_sample: bool, enable_thinking: bool) -> str:
        return run_single_model_internal(image_input, prompt_input, do_sample, enable_thinking)

    device = f"cuda:{gpu}"
    print(f"Loading model {model_path} to device {device}...")

    model = Ovis2_5.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    # Set optimal max_pixels once at initialization
    optimal_max_pixels = get_optimal_max_pixels()
    # Check which inference path will be used
    if hasattr(model, "chat"):
        print("✓ Model loaded successfully with custom chat method support!")
    else:
        print("✓ Model loaded successfully - using fallback inference method.")

    custom_css = """
    #output_md .prose { font-size: 18px !important; }
    #output_md pre { 
        max-height: none !important; 
        height: auto !important; 
        overflow: visible !important; 
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
    #output_md code { 
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
    }
    """
    with gr.Blocks(theme=gr.themes.Default(), css=custom_css) as demo:
        gr.Markdown("# LH - 유지보수 유형 분석 by Superb AI")

        with gr.Row():
            # Left column - inputs
            with gr.Column(scale=1):
                gr.Markdown("### Inputs")
                image_input = gr.Image(label="Image Input", type="pil", height=400)

                with gr.Accordion("Generation Settings", open=True):
                    do_sample = gr.Checkbox(label="Enable Sampling (Do Sample)", value=False)
                    enable_thinking = gr.Checkbox(label="Deep Thinking", value=True)

                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                    generate_btn = gr.Button("Generate", variant="primary", scale=2)

            # Right column - output
            with gr.Column(scale=2):
                gr.Markdown("### Model Output\n")
                output_display = gr.Markdown(elem_id="output_md")

        # Event handlers
        run_inputs = [
            image_input,
            do_sample,
            enable_thinking,
        ]

        generate_btn.click(
            fn=start_generation, outputs=[generate_btn, clear_btn, output_display]
        ).then(fn=run_model_with_prompt, inputs=run_inputs, outputs=[output_display]).then(
            fn=finish_generation, outputs=[generate_btn, clear_btn]
        )

        clear_btn.click(
            fn=clear_interface,
            outputs=[output_display, image_input],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio interface for Ovis.")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to run the model.")
    parser.add_argument("--port", type=int, default=9901, help="Port to run the Gradio service.")
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Server name for Gradio app."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    demo = build_demo(model_path=args.model_path, gpu=args.gpu)

    print(f"Launching Gradio app at http://{args.server_name}:{args.port}")
    demo.queue().launch(
        server_name=args.server_name, server_port=args.port, share=False, ssl_verify=False
    )
