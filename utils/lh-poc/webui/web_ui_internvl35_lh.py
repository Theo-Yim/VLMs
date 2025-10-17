import argparse
import json
import re
from typing import Optional, Tuple

import gradio as gr
import PIL.Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

model: AutoModel = None
tokenizer: AutoTokenizer = None

R1_SYSTEM_PROMPT = """You are a professional home inspector AI that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section."""
SIMPLE_SYSTEM_PROMPT = "You are a professional home inspector AI."


def fix_json_list_format(text: str) -> str:
    """
    Extract and fix JSON format from text that may contain:
    1. [...] array with proper formatting
    2. {...}, {...} objects with commas but no array brackets
    3. {...} {...} objects without commas (spaces only)
    4. Extra text outside of JSON structure

    Returns properly formatted JSON string.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[text.find("```json") + 7 : text.rfind("```")].strip()

    # Case 1: If already has array brackets [...], extract it
    if "[" in text and "]" in text:
        text = text[text.find("[") : text.rfind("]") + 1].strip()
    # Case 2 & 3: Has curly braces {...}
    elif "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1].strip()

        # Fix pattern: } { -> }, { (add commas between objects)
        text = re.sub(r"}\s+{", "}, {", text)

        # If we have multiple objects (indicated by "},"), wrap in array brackets
        if "}," in text and text.startswith("{") and text.endswith("}"):
            text = "[" + text + "]"

    return text


def parse_response(response: str) -> str:
    # Format output - Extract thinking and response
    if "<think>" in response and "</think>" in response:
        thinking_start = response.find("<think>") + 7
        thinking_end = response.find("</think>")
        thinking = response[thinking_start:thinking_end].strip()
        answer = response[thinking_end + 8 :].strip()
    elif "final answer" in response.lower():
        # Extract thinking (CoT) and final answer
        thinking = response[: response.lower().rfind("final answer")].strip()
        answer = response[response.lower().rfind("final answer") + 13 :].strip("\n :")
    else:
        # No thinking tags nor Final answer found, just return response
        # Fix: sometimes response has {...} {...}, list without commas
        answer = response

    answer = fix_json_list_format(answer)
    try:
        answer_json = json.dumps(json.loads(answer), indent=2, ensure_ascii=False)
        answer_json = (
            answer_json.replace("space", "공간")
            .replace("defect_type", "하자유형")
            .replace("defect_description", "하자내용")
            .replace("material_part", "부위자재")
            .replace("location_in_image", "이미지 내 위치")
            .replace("defect_present", "하자 존재 여부")
        )
        return f"**Response:**\n```json\n{answer_json}\n```"
    except json.JSONDecodeError:
        print("Parse error")
        return f"**Response:**\n{answer}"


def build_transform(input_size):
    """Build image transformation pipeline."""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamically preprocess image into multiple patches."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    """Load and preprocess image for InternVL."""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_optimal_max_num() -> int:
    """
    Automatically determine optimal max_num (image patches) based on available GPU memory.
    Returns max_num value suitable for the current GPU.
    """
    try:
        if torch.cuda.is_available():
            # Get GPU memory in GB
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")
            # Set max_num based on GPU memory
            if gpu_memory_gb >= 20:
                max_num = 12
                print(f"Using max_num = {max_num} for high-memory GPU ({gpu_memory_gb:.1f}GB)")
            elif gpu_memory_gb >= 15:
                max_num = 8
                print(f"Using max_num = {max_num} for 16GB GPU ({gpu_memory_gb:.1f}GB)")
            else:
                max_num = 6
                print(f"Using max_num = {max_num} for ~16GB GPU ({gpu_memory_gb:.1f}GB)")
            return max_num
        else:
            print("CUDA not available, using default max_num")
            return 12
    except Exception as e:
        print(f"Error detecting GPU memory: {e}, using default max_num")
        return 12


def run_single_model_internal(
    image_input: Optional[PIL.Image.Image],
    prompt: str,
    do_sample: bool,
    enable_thinking: bool,
    max_num: int = 12,
) -> str:
    """Run single model inference using InternVL3.5."""
    max_new_tokens = 2048

    if not image_input:
        gr.Warning("Please upload an image.")
        return ""

    try:
        # Get model device
        model_device = next(model.parameters()).device

        # Preprocess image with dynamic max_num
        pixel_values = load_image(image_input, input_size=448, max_num=max_num)
        pixel_values = pixel_values.to(torch.bfloat16).to(model_device)

        # Prepare generation config
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            generation_config["temperature"] = 0.6

        # Set system message for thinking mode
        if enable_thinking:
            model.system_message = R1_SYSTEM_PROMPT
            question = f"<image>\n{prompt}"
        else:
            # Reset to default system message
            model.system_message = SIMPLE_SYSTEM_PROMPT
            question = f"<image>\n{prompt}"

        # Run inference
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        # print(response + "\n")
        answer = parse_response(response)
        print(answer)
        return answer

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


def build_demo(model_path: str, gpu: int, prompt_sb=False):
    """Build single-model Gradio demo interface."""
    global model, tokenizer

    # Determine optimal max_num based on GPU memory
    optimal_max_num = get_optimal_max_num()

    # Define the prompt as a constant
    prompt_input_th = '## Your job is to analyze a home image for structural issues and defects. Carefully examine all visual cues in the original image. Use logical reasoning to identify the areas and materials that might be associated with a potential defect, as well as any physical clues.\nYou need to guess the space, defect type, defect description, material part, and location in the image of the image.\n\nYou must output the final answer in the list of json objects, where each json is possible defect information with the following fields:\n- space: [Space name from the list of Spaces]\n- defect_present: "Yes" / "Unknown"\n- If "Yes", also include:\n  - defect_type: [type from the list of Defect Types]\n  - defect_description: [brief description of the defect]\n  - material_part: [material part from the list of Material Parts]\n  - location_in_image: [describe location within the image, if applicable]\nList up the json objects in the order of possible combinations, and try to generate 2-3 possible json objects.\n\n### List of Defect Types and Descriptions\nPoor Caulking : The sealant (caulking/silicone) applied around window frames, bathtubs, or sinks to block moisture and air is peeling, cracked, or missing.\nWater Leak : Water seeps into the building\'s interior (walls, ceiling, floor) due to a damaged waterproof layer, broken pipes, etc.\nDelamination / Flaking : A paint or coating layer flakes or peels off like a film. It often implies a more severe condition than \'Lifting\' (들뜸).\nCorrosion / Rust : Metal materials deteriorate and rust due to chemical reactions with oxygen and moisture in the air.\nScratch / Nick / Dent : A general term for marks or damage on a surface, including nicks, dents, and scratches.\nDamage / Breakage : A material is broken or shattered due to external impact, losing its original function.\nPoor Finishing : The final touches, such as wallpapering, painting, or caulking, are messy or incomplete.\nLifting / Peeling : Wallpaper, vinyl sheets, tiles, or flooring material is detaching from the surface due to poor adhesion.\nCrack : A line-shaped break that appears on the surface of concrete or finishing materials like walls, floors, or ceilings. It can be caused by structural issues or material shrinkage/expansion.\nStain / Contamination : A surface has a mark, spot, or dirt that cannot be easily removed.\nHole / Puncture : An unintentional hole in a wall, floor, or door.\nGrout Loss : The grout material between tiles is cracking, crumbling, or falling out.\nDiscoloration : The original color of a material has changed due to factors like sunlight, chemicals, or moisture.\nGap / Separation : An unintended space has opened up between two components that should be joined, such as a door frame and a wall.\nPoor paintwork : A defect in the quality or application of paint on a building\'s surface\nPeeling : A defect where a layer or coating comes off, typically referring to paint or plaster.\nPoor Joint / Seam : The connection point between two materials (e.g., wallpaper sheets, floorboards) is not smooth or has a gap.\nSinking / Indentation : A part of a surface, such as a floor or wall, is dented or has sunk inward.\nCondensation : Water droplets form on surfaces like walls and windows due to temperature differences, as moisture from the air condenses. It is a primary cause of mold.\nPoor wallpapering : A defect or issue related to improper or poor quality wallpaper installation\nTear : A type of defect characterized by a long, narrow opening or breakage in a material or structure'

    prompt_input_sb = "## Your job is to analyze a home image for structural issues and defects. Carefully examine all visual cues in the original image. Use logical reasoning to identify the areas and materials that might be associated with a potential defect, as well as any physical clues.\nYou need to guess the space, defect type, defect description, material part, and location in the image of the image.\n\n* End your response with 'Final answer: '. In the Final answer section, there should be the list of json objects, where each json is possible defect information with the following fields:\n- space: [Space name from the list of Spaces]\n- defect_present: \"Yes\" / \"Unknown\"\n- If \"Yes\", also include:\n  - defect_type: [type from the list of Defect Types]\n  - defect_description: [brief description of the defect]\n  - material_part: [material part from the list of Material Parts]\n  - location_in_image: [describe location within the image, if applicable]\n* List up the json objects in the order of possible combinations, and try to generate 2-3 possible json objects.\n\n### List of Defect Types and Descriptions\nPoor Caulking : The sealant (caulking/silicone) applied around window frames, bathtubs, or sinks to block moisture and air is peeling, cracked, or missing.\nWater Leak : Water seeps into the building's interior (walls, ceiling, floor) due to a damaged waterproof layer, broken pipes, etc.\nDelamination / Flaking : A paint or coating layer flakes or peels off like a film. It often implies a more severe condition than 'Lifting' (들뜸).\nCorrosion / Rust : Metal materials deteriorate and rust due to chemical reactions with oxygen and moisture in the air.\nScratch / Nick / Dent : A general term for marks or damage on a surface, including nicks, dents, and scratches.\nDamage / Breakage : A material is broken or shattered due to external impact, losing its original function.\nPoor Finishing : The final touches, such as wallpapering, painting, or caulking, are messy or incomplete.\nLifting / Peeling : Wallpaper, vinyl sheets, tiles, or flooring material is detaching from the surface due to poor adhesion.\nCrack : A line-shaped break that appears on the surface of concrete or finishing materials like walls, floors, or ceilings. It can be caused by structural issues or material shrinkage/expansion.\nStain / Contamination : A surface has a mark, spot, or dirt that cannot be easily removed.\nHole / Puncture : An unintentional hole in a wall, floor, or door.\nGrout Loss : The grout material between tiles is cracking, crumbling, or falling out.\nDiscoloration : The original color of a material has changed due to factors like sunlight, chemicals, or moisture.\nGap / Separation : An unintended space has opened up between two components that should be joined, such as a door frame and a wall.\nPoor paintwork : A defect in the quality or application of paint on a building's surface\nPeeling : A defect where a layer or coating comes off, typically referring to paint or plaster.\nPoor Joint / Seam : The connection point between two materials (e.g., wallpaper sheets, floorboards) is not smooth or has a gap.\nSinking / Indentation : A part of a surface, such as a floor or wall, is dented or has sunk inward.\nCondensation : Water droplets form on surfaces like walls and windows due to temperature differences, as moisture from the air condenses. It is a primary cause of mold.\nPoor wallpapering : A defect or issue related to improper or poor quality wallpaper installation\nTear : A type of defect characterized by a long, narrow opening or breakage in a material or structure"

    if not prompt_sb:
        prompt_input = prompt_input_th
    else:
        prompt_input = prompt_input_sb

    # Create a wrapper function that uses the constant prompt with optimal max_num
    def run_model_with_prompt(image_input, do_sample: bool, enable_thinking: bool) -> str:
        return run_single_model_internal(
            image_input, prompt_input, do_sample, enable_thinking, max_num=optimal_max_num
        )

    device = f"cuda:{gpu}"
    print(f"Loading model {model_path} to device {device}...")

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    print("✓ Model loaded successfully!")

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
                    enable_thinking = gr.Checkbox(label="Deep Thinking", value=False)

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
    parser = argparse.ArgumentParser(description="Gradio interface for InternVL3.5.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to InternVL3.5 model")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to run the model.")
    parser.add_argument("--port", type=int, default=9902, help="Port to run the Gradio service.")
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Server name for Gradio app."
    )
    parser.add_argument("--prompt-sb", action="store_true", help="Use the SB style prompt.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    demo = build_demo(model_path=args.model_path, gpu=args.gpu, prompt_sb=args.prompt_sb)

    print(f"Launching Gradio app at http://{args.server_name}:{args.port}")
    demo.queue().launch(
        server_name=args.server_name, server_port=args.port, share=False, ssl_verify=False
    )
