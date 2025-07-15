"""
InternVL3 Video Inference Script.
It processes videos one by one using model parallelization across full GPUs.
"""

import gc
import logging
import os
import time

import torch

from InternVL3.utils.constants import generation_config, user_question
from InternVL3.utils.preprocess import load_video
from InternVL3.utils.processor import load_models, split_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("inference_img_batch.log", mode="w"),  # Log to file
    ],
)

model_path = "OpenGVLab/InternVL3-78B"
device_map = split_model(model_path)
model, tokenizer = load_models(model_path, device_map)


@torch.inference_mode()
def process_multiple_videos_optimized(video_paths, model, tokenizer, **video_kwargs):
    results = []

    for i, video_path in enumerate(video_paths):
        logging.info(f"Processing video {i + 1}/{len(video_paths)}: {os.path.basename(video_path)}")
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

        logging.info(f"A: {response}")
        logging.info(f"Processing time: {processing_time:.2f}s\n")

        # 메모리 정리
        del pixel_values, num_patches_list, response
        torch.cuda.empty_cache()
        gc.collect()

    return results


if __name__ == "__main__":
    video_path = "/workspace/vss-engine/samples/untitled/"
    video_paths = [
        os.path.join(video_path, "37.mp4"),
        os.path.join(video_path, "36.mp4"),
        os.path.join(video_path, "35.mp4"),
        os.path.join(video_path, "34.mp4"),
    ]

    logging.info("Starting optimized video processing...")
    start_total = time.time()

    results = process_multiple_videos_optimized(
        video_paths, model, tokenizer, num_segments=None, max_num=2
    )

    total_time = time.time() - start_total

    logging.info("\n=== Processing Summary ===")
    logging.info(f"Total videos: {len(results)}")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average per video: {total_time / len(results):.2f}s")

    for result in results:
        video_name = result["video"]
        response = result["response"]
        proc_time = result["processing_time"]
        logging.info(f"\nVideo: {video_name} ({proc_time:.2f}s)")
        # logging.info(f'Response: {response[:100]}...')

    logging.info("=== Experiment done! ===")
