"""
InternVL3 Video Inference Script.
This script processes videos one by one using model parallelization across full GPUs.
"""

import gc
import json
import logging
import os
import sys
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
        logging.FileHandler("inference_video.log", mode="w"),  # Log to file
    ],
)

model_path = "OpenGVLab/InternVL3-78B"
device_map = split_model(model_path)
model, tokenizer = load_models(model_path, device_map)


def process_videos(video_paths, model, tokenizer, json_output_path, **video_kwargs):
    """Process multiple videos sequentially with comprehensive tracking"""
    num_success = 0
    if not video_paths:
        logging.warning("No videos to process")
        return

    for i, video_path in enumerate(video_paths):
        logging.info(f"Processing {i + 1}/{len(video_paths)}...")
        ret = process_single_video(
            video_path, model, tokenizer, json_output_path, **video_kwargs
        )
        if ret is not None and ret:
            # logging.info(
            #     f"Processed video {i + 1}/{len(video_paths)}: {os.path.basename(video_path)} in {proc_time:.2f}s"
            # )
            num_success += 1

    return num_success


@torch.inference_mode()
def process_single_video(video_path, model, tokenizer, json_output_path, **video_kwargs):
    """Process a single video with comprehensive error handling"""

    # Validate video file exists
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return None

    # Check file size (optional warning for very large files)
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if file_size_mb > 1000:  # > 1GB
        logging.warning(f"Large video file ({file_size_mb:.1f}MB): {os.path.basename(video_path)}")

    start_time = time.time()
    # Load video with error handling
    try:
        pixel_values, num_patches_list = load_video(video_path, **video_kwargs)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
    except Exception as e:
        logging.error(f"Failed to load video {video_path}: {str(e)}")
        return None

    # Validate loaded data
    if len(num_patches_list) == 0:
        logging.warning(f"No frames extracted from video: {video_path}")
        return None

    # Create video prefix and question
    video_prefix = "".join([f"Frame{j + 1}: <image>\n" for j in range(len(num_patches_list))])
    question = video_prefix + user_question

    try:
        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error(
            f"GPU out of memory for video {video_path}. Try reducing num_segments or max_num."
        )
        torch.cuda.empty_cache()
        gc.collect()
        return None
    # except Exception as e:
    #     logging.error(f"Model inference failed for {video_path}: {str(e)}")
    #     return None

    processing_time = time.time() - start_time
    logging.info(f"Processed {len(num_patches_list[0])} tiles of {len(num_patches_list)} frames of video {os.path.basename(video_path)} in {processing_time:.2f}s")

    # Save response
    save_response(video_path, response, json_output_path)

    # Memory cleanup
    del pixel_values, response
    torch.cuda.empty_cache()
    gc.collect()
    return True


def save_response(video_path, response, json_output_path):
    """Save video analysis response"""
    try:
        output_file = os.path.basename(video_path)
        # Replace video extensions with .json
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]:
            output_file = output_file.replace(ext, ".json")

        output_path = os.path.join(json_output_path, output_file)

        # Extract JSON from response
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1

        if start_idx == -1 or end_idx == 0:
            logging.error(f"No JSON found in response for {video_path}")
            # Save raw response as fallback
            with open(output_path.replace(".json", "_raw.txt"), "w", encoding="utf-8") as f:
                f.write(response)
            return

        json_str = response[start_idx:end_idx]

        try:
            json_data = json.loads(json_str)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error for {video_path}: {str(e)}")
            # Save raw JSON string for debugging
            with open(output_path.replace(".json", "_invalid.json"), "w", encoding="utf-8") as f:
                f.write(json_str)

    except Exception as e:
        logging.error(f"Failed to save response for {video_path}: {str(e)}")


if __name__ == "__main__":
    # Dataset configuration - similar to inference_img_batch.py
    dataset_list = [
        "untitled",
        # Add your video dataset names here
    ]
    video_path_root = "/workspace/vss-engine/samples/"
    # # You can also specify individual video paths for testing
    # manual_video_paths = [
    #     # "/workspace/vss-engine/samples/untitled/37.mp4",
    #     # "/workspace/vss-engine/samples/untitled/36.mp4",
    #     # "/workspace/vss-engine/samples/untitled/35.mp4",
    #     # "/workspace/vss-engine/samples/untitled/34.mp4",
    # ]

    if not os.path.exists(video_path_root):
        logging.error(f"Base video path does not exist: {video_path_root}")
        sys.exit(1)

    json_output_path_root = video_path_root

    # Video processing parameters
    video_kwargs = {
        "num_segments": None,  # Auto-calculate based on video
        "max_num": 2,  # Max split per frame
        "max_segments": 50,  # Maximum segments to process
    }

    # # Process manual paths if specified
    # if manual_video_paths:
    #     logging.info("Processing manually specified video paths...")

    #     # Validate manual paths
    #     valid_paths = [p for p in manual_video_paths if os.path.exists(p)]
    #     if not valid_paths:
    #         logging.error("None of the manual video paths exist")
    #     else:
    #         # Create output directory for manual processing
    #         manual_output_path = "./manual_video_output"
    #         if not os.path.exists(manual_output_path):
    #             os.makedirs(manual_output_path)

    #         start_total = time.time()
    #         results = process_multiple_videos_optimized(
    #             valid_paths, model, tokenizer, manual_output_path, **video_kwargs
    #         )
    #         total_time = time.time() - start_total
    #         logging.info(f"\nManual processing completed in {total_time:.2f}s")

    for dataset in dataset_list:
        video_path = os.path.join(video_path_root, dataset)
        json_output_path = os.path.join(json_output_path_root, f"{dataset}_json")

        # Check if dataset path exists
        if not os.path.exists(video_path):
            logging.warning(f"Dataset path does not exist: {video_path}. Skipping.")
            continue

        if not os.path.exists(json_output_path):
            os.makedirs(json_output_path)

        # Get video files
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"]
        video_list = os.listdir(video_path)
        video_paths = [
            os.path.join(video_path, video)
            for video in video_list
            if any(video.lower().endswith(ext) for ext in video_extensions)
        ]

        if not video_paths:
            logging.warning(f"No valid videos found in {video_path}")
            continue

        logging.info(f"Processing {len(video_paths)} videos from {dataset}")
        start_total = time.time()

        try:
            num_success = process_videos(
                video_paths, model, tokenizer, json_output_path, **video_kwargs
            )
            total_time = time.time() - start_total
            logging.info(f"\n=== {dataset} Processing Summary ===")
            logging.info(f"Total videos processed: {num_success}/{len(video_paths)}")
            logging.info(f"Total time: {total_time:.2f}s")
            logging.info(f"Average time per video: {total_time / len(video_paths):.2f}s")
        except Exception as e:
            logging.error(f"Failed to process dataset {dataset}: {str(e)}")
            continue

    logging.info("=== All datasets processed! ===")
