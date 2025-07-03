"""
Compared to inference_mp.py, this script optimizes the video processing by using two sets of 4 GPUs
for continuous processing of multiple videos.
"""
import gc
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.multiprocessing as mp
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


def split_model_for_group(model_path, gpu_group: list):
    """Create device map for a specific group of 4 GPUs"""
    device_map = {}
    world_size = len(gpu_group)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        gpu_id = gpu_group[i]
        for _ in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = gpu_id
            layer_cnt += 1
    first_gpu = gpu_group[0]
    device_map["vision_model"] = first_gpu
    device_map["mlp1"] = first_gpu
    device_map["language_model.model.tok_embeddings"] = first_gpu
    device_map["language_model.model.embed_tokens"] = first_gpu
    device_map["language_model.output"] = first_gpu
    device_map["language_model.model.norm"] = first_gpu
    device_map["language_model.model.rotary_emb"] = first_gpu
    device_map["language_model.lm_head"] = first_gpu

    return device_map


class ModelWorker:
    def __init__(self, gpu_group, model_path, worker_id):
        self.gpu_group = gpu_group
        self.model_path = model_path
        self.worker_id = worker_id
        self.model = None
        self.tokenizer = None
        self.generation_config = dict(
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
        self.initialize_model()

    def initialize_model(self):
        """Initialize model on specific GPU group"""
        device_map = split_model_for_group(self.model_path, self.gpu_group)

        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        except Exception:
            print("Failed to compile model. Continuing without compilation.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )

        print(f"Worker {self.worker_id} initialized on GPUs {self.gpu_group}")

    def process_video(self, video_path, user_question, **video_kwargs):
        """Process a single video"""

        print(f"Worker {self.worker_id} processing: {os.path.basename(video_path)}")
        start_time = time.time()

        # Load video and move to first GPU of the group
        pixel_values, num_patches_list = load_video(video_path, **video_kwargs)
        pixel_values = pixel_values.to(torch.bfloat16).cuda(self.gpu_group[0])

        # Create question
        video_prefix = "".join([f"Frame{j + 1}: <image>\n" for j in range(len(num_patches_list))])
        question = video_prefix + user_question

        # Process through model
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            self.generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
        processing_time = time.time() - start_time

        print(f"A: {response}")
        print(f"Processing time: {processing_time:.2f}s\n")

        # 메모리 정리
        del pixel_values, num_patches_list
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "video": os.path.basename(video_path),
            "response": response,
            "worker_id": self.worker_id,
        }


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
def process_batch_videos(video_paths, model_path, user_question, **video_kwargs):
    """
    Process videos using two sets of 4 GPUs continuously
    """
    # Define GPU groups (modify these based on your GPU setup)
    gpu_group_1 = [0, 1, 2, 3]  # First set of 4 GPUs
    gpu_group_2 = [4, 5, 6, 7]  # Second set of 4 GPUs

    print("Initializing workers (this may take a few minutes)...")

    # Create and initialize workers in main thread (CRUCIAL for CUDA context)
    workers = [
        ModelWorker(gpu_group_1, model_path, worker_id=1),
        ModelWorker(gpu_group_2, model_path, worker_id=2),
    ]

    print("Both workers initialized! Starting continuous processing...\n")

    # Submit ALL videos to ThreadPool at once for optimal load balancing
    all_results = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all videos immediately - let the scheduler handle distribution
        future_to_video = {}

        for video_path in video_paths:
            # Alternate between workers for initial distribution
            worker_idx = len(future_to_video) % 2
            worker = workers[worker_idx]

            future = executor.submit(
                worker.process_video, video_path, user_question, **video_kwargs
            )
            future_to_video[future] = video_path

        print(f"Submitted {len(video_paths)} videos to workers. Processing...")

        # Collect results as they complete (no batching)
        from concurrent.futures import as_completed

        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                result = future.result()
                all_results.append(result)

                completed_count = len(all_results)
                total_count = len(video_paths)

                print(
                    f"✓ Worker {result['worker_id']} completed: {result['video']} "
                    f"({completed_count}/{total_count})"
                )

            except Exception as exc:
                print(f"✗ Video {os.path.basename(video_path)} generated an exception: {exc}")

    print(f"\n=== All {len(all_results)} videos processed! ===")
    return all_results


# Usage
if __name__ == "__main__":
    model_path = "OpenGVLab/InternVL3-78B"

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

    # Process with batch size 2 using two sets of 4 GPUs
    results = process_batch_videos(
        video_paths, model_path, user_question, num_segments=None, max_num=2
    )

    total_time = time.time() - start_total

    print(f"\n=== Processing Summary ===")
    print(f"Total videos: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per video: {total_time / len(results):.2f}s")

    for result in results:
        print(f"Video: {result['video']} (Worker {result['worker_id']})")
        print(f"Response: {result['response']}\n")

    print("=== Batch processing completed! ===")
