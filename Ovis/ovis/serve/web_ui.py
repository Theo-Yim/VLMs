import argparse
from typing import List, Optional, Tuple

import gradio as gr
import moviepy as mp
import numpy as np
import PIL.Image
import torch
from ovis.model.modeling_ovis2_5 import Ovis2_5

model: Ovis2_5 = None


def load_video_frames(
    video_path: Optional[str], n_frames: int = 8
) -> Optional[List[PIL.Image.Image]]:
    """Extract a fixed number of frames from the video file."""
    if not video_path:
        return None
    try:
        with mp.VideoFileClip(video_path) as clip:
            duration = clip.duration
            if duration is None or clip.fps is None or duration <= 0 or clip.fps <= 0:
                print(f"Warning: Unable to process video {video_path}. Invalid duration or fps.")
                return None

            total_possible_frames = int(duration * clip.fps)
            num_to_extract = min(n_frames, total_possible_frames)

            if num_to_extract <= 0:
                print(
                    f"Warning: Cannot extract frames from {video_path}. Computed extractable frames is zero."
                )
                return None

            frames = []
            timestamps = np.linspace(0, duration, num_to_extract, endpoint=True)
            for t in timestamps:
                frame_np = clip.get_frame(t)
                frames.append(PIL.Image.fromarray(frame_np))
        print(f"Successfully extracted {len(frames)} frames from {video_path}.")
        return frames
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None


def run_single_model(
    image_input: Optional[PIL.Image.Image],
    video_input: Optional[str],
    prompt: str,
    do_sample: bool,
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    """Run single model inference using the chat method."""
    if not prompt and not image_input and not video_input:
        gr.Warning("Please enter a prompt, upload an image, or upload a video.")
        return ""

    # Prepare vision inputs
    images = [image_input] if image_input else None
    video_frames = load_video_frames(video_input)
    videos = [video_frames] if video_frames else None

    # Check for conflicting inputs
    if images and videos:
        gr.Warning("Please provide either an image or a video, not both.")
        return ""

    try:
        # Check if model has custom chat method (for custom checkpoints)
        if hasattr(model, "chat"):
            # Path 1: Use the custom chat method
            response, thinking, _ = model.chat(
                prompt=prompt,
                history=None,  # Always start a new conversation
                images=images,
                videos=videos,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                temperature=0.6 if do_sample else 0.0,
                enable_thinking=enable_thinking,
                enable_thinking_budget=enable_thinking,
            )

            # Format output
            if enable_thinking and thinking:
                return f"**Thinking:**\n```text\n{thinking}\n```\n\n**Response:**\n{response}"
            return response

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
            elif videos:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": videos[0]},
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
                max_pixels=1792 * 1792,
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

                return f"**Thinking:**\n```text\n{thinking}\n```\n\n**Response:**\n{response}"
            else:
                # Clean up the response by removing any system tokens
                response = (
                    response_text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
                )
                return response

    except Exception as e:
        gr.Warning(f"Error during inference: {str(e)}")
        return f"Error: {str(e)}"


def toggle_media_input(choice: str) -> Tuple[gr.update, gr.update]:
    """Toggle visibility of image and video input components."""
    if choice == "Image":
        return gr.update(visible=True, value=None), gr.update(visible=False, value=None)
    else:
        return gr.update(visible=False, value=None), gr.update(visible=True, value=None)


def clear_interface() -> Tuple[str, None, None, str, str]:
    """Reset all inputs and outputs."""
    return "", None, None, "", "Image"


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
    global model
    device = f"cuda:{gpu}"
    print(f"Loading model {model_path} to device {device}...")

    model = Ovis2_5.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    # Check which inference path will be used
    if hasattr(model, "chat"):
        print("✓ Model loaded successfully with custom chat method support!")
    else:
        print("✓ Model loaded successfully - using fallback inference method.")

    custom_css = "#output_md .prose { font-size: 18px !important; }"
    with gr.Blocks(theme=gr.themes.Default(), css=custom_css) as demo:
        gr.Markdown("# Ovis2.5 VLM Web UI")
        gr.Markdown(f"Running on **GPU {gpu}**. Each submission starts a new conversation.")

        with gr.Row():
            # Left column - inputs
            with gr.Column(scale=1):
                gr.Markdown("### Inputs")
                input_type_radio = gr.Radio(
                    choices=["Image", "Video"], value="Image", label="Select Input Type"
                )
                image_input = gr.Image(label="Image Input", type="pil", visible=True, height=400)
                video_input = gr.Video(label="Video Input", visible=False)
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here... (Press Enter to submit)",
                    value="<image>\n## You are a professional house construction inspector. Your job is to examine the provided image and determine if there is any defect. You need to guess the space, defect type, defect description, material part, and location in the image of the image.\n\nYou must output the answer in the json format with the following fields:\n- space: [Space name from the list of Spaces]\n- defect_present: Yes / No\n- If Yes, also include:\n  - defect_type: [type from the list of Defect Types]\n  - defect_description: [brief description of the defect]\n  - material_part: [material part from the list of Material Parts]\n  - location_in_image: [describe location within the image, if applicable]\n\n### Instructions\n- Carefully examine each part of the image.\n- Identify the space of the image.\n- Identify the material part of the image.\n- Identify the defect type of the image.\n- Identify the defect description of the image.\n- Identify the location in the image of the image.\n",
                    lines=3,
                )
                with gr.Accordion("Generation Settings", open=True):
                    do_sample = gr.Checkbox(label="Enable Sampling (Do Sample)", value=False)
                    max_new_tokens = gr.Slider(
                        minimum=32, maximum=2048, value=1024, step=32, label="Max New Tokens"
                    )
                    enable_thinking = gr.Checkbox(label="Deep Thinking", value=False)

                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                    generate_btn = gr.Button("Generate", variant="primary", scale=2)

            # Right column - output
            with gr.Column(scale=2):
                model_name = model_path.strip().strip("/").split("/")[-1]
                gr.Markdown(f"### Model Output\n`{model_name}`")
                output_display = gr.Markdown(elem_id="output_md")

        # Event handlers
        input_type_radio.change(
            fn=toggle_media_input, inputs=input_type_radio, outputs=[image_input, video_input]
        )

        run_inputs = [
            image_input,
            video_input,
            prompt_input,
            do_sample,
            max_new_tokens,
            enable_thinking,
        ]

        generate_btn.click(
            fn=start_generation, outputs=[generate_btn, clear_btn, output_display]
        ).then(fn=run_single_model, inputs=run_inputs, outputs=[output_display]).then(
            fn=finish_generation, outputs=[generate_btn, clear_btn]
        )

        prompt_input.submit(
            fn=start_generation, outputs=[generate_btn, clear_btn, output_display]
        ).then(fn=run_single_model, inputs=run_inputs, outputs=[output_display]).then(
            fn=finish_generation, outputs=[generate_btn, clear_btn]
        )

        clear_btn.click(
            fn=clear_interface,
            outputs=[output_display, image_input, video_input, prompt_input, input_type_radio],
        ).then(fn=toggle_media_input, inputs=input_type_radio, outputs=[image_input, video_input])

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
