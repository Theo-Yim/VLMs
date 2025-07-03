import gc
import math
import os
import time

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height), Image.LANCZOS)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size), Image.LANCZOS)
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0

    return device_map


# 더 효율적인 generation config
generation_config = dict(
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
)
# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
model_path = "OpenGVLab/InternVL3-78B"
device_map = split_model(model_path)

model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map,
).eval()
try:
    model = torch.compile(model, mode="reduce-overhead")
except Exception:
    print("Failed to compile model. Continuing without compilation.")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    return frame_indices


def load_video(
    video_path, bound=None, input_size=448, max_num=2, num_segments=None, max_segments=50
):
    # max_num : max number of split per frame
    max_num = max(1, max_num)

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    if num_segments is None or num_segments <= 0:
        num_segments = math.ceil(max_frame * 2 / fps)

    num_segments = min(max_segments // max_num, num_segments)

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=False, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


@torch.inference_mode()
def process_multiple_videos_optimized(video_paths, model, tokenizer, **video_kwargs):
    results = []

    for i, video_path in enumerate(video_paths):
        print(f"Processing video {i + 1}/{len(video_paths)}: {os.path.basename(video_path)}")
        start_time = time.time()

        pixel_values, num_patches_list = load_video(video_path, **video_kwargs)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        video_prefix = "".join([f"Frame{j + 1}: <image>\n" for j in range(len(num_patches_list))])
        question = video_prefix + user_question

        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )

        processing_time = time.time() - start_time
        results.append(
            {
                "video": os.path.basename(video_path),
                "response": response,
                "processing_time": processing_time,
            }
        )

        print(f"A: {response}")
        print(f"Processing time: {processing_time:.2f}s\n")

        # 메모리 정리
        del pixel_values, num_patches_list, response
        torch.cuda.empty_cache()
        gc.collect()

    return results


user_question = """Given the media, output a detailed analysis of how you analyzed the scene in JSON format. 
Conduct an analysis of what you see and how each component interacts with each other.

Follow this JSON structure exactly:

{
 "planning": {
  "ROI": [
   {"category": "noun_category (e.g., person, car, dog)", "description": "noun phrase with short participial phrase and appearance"},
  ]
 },
 "reasoning": {
  "regional_analysis": [
   {"region": "region name from planning stage, sequentially",
    "observations or analysis": ["detailed analysis point 1", "detailed analysis point 2"]}
  ],
  "overall_scene": {
   "observations or analysis": ["overall scene analysis points"]
  }
 },
 "conclusion": {
 "comprehensive_analysis": "comprehensive and detailed summary assembling all reasoning"
 }
}

IMPORTANT: 
- Include only few major regions of interest (ROI).
- During reasoning, follow natural reasoning flow - if you discover something new or need to reconsider, express it naturally (e.g., "Aha,", "Wait,", "Actually...", "Looking more carefully..."). Only use such expressions when there's a genuine reasoning transition, not artificially.
- Each observation or analysis should combine specific visual details with interpretive reasoning
- For video content, note any significant temporal changes or developments naturally within your observations"""

video_path = "/workspace/vss-engine/samples/untitled/"
video_paths = [
    os.path.join(video_path, "37.mp4"),
    os.path.join(video_path, "36.mp4"),
    os.path.join(video_path, "35.mp4"),
    os.path.join(video_path, "34.mp4"),
]

print("Starting optimized video processing...")
start_total = time.time()

results = process_multiple_videos_optimized(
    video_paths, model, tokenizer, num_segments=None, max_num=2
)

total_time = time.time() - start_total

print(f"\n=== Processing Summary ===")
print(f"Total videos: {len(results)}")
print(f"Total time: {total_time:.2f}s")
print(f"Average per video: {total_time / len(results):.2f}s")


for result in results:
    video_name = result["video"]
    response = result["response"]
    proc_time = result["processing_time"]
    print(f"\nVideo: {video_name} ({proc_time:.2f}s)")
    # print(f'Response: {response[:100]}...')

print("=== Experiment done! ===")
