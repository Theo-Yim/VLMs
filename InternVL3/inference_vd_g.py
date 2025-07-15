"""
InternVL3 Video Inference Script using Two Sets of 4 GPUs
Compared to inference_vd.py, this script optimizes the video processing by using two sets of 4 GPUs.
Compared to model parallelization, this script is a combination of model parallelization and data parallelization.
"""

import gc
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from InternVL3.utils.constants import generation_config, user_question
from InternVL3.utils.preprocess import load_video
from InternVL3.utils.processor import load_models, split_model_for_group

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("inference_img_batch.log", mode="w"),  # Log to file
    ],
)


class ModelWorker:
    def __init__(self, gpu_group, model_path, worker_id):
        self.gpu_group = gpu_group
        self.model_path = model_path
        self.worker_id = worker_id
        self.model = None
        self.tokenizer = None
        self.initialize_model()

    def initialize_model(self):
        """Initialize model on specific GPU group"""
        device_map = split_model_for_group(self.model_path, self.gpu_group)
        self.model, self.tokenizer = load_models(self.model_path, device_map)
        logging.info(f"Worker {self.worker_id} initialized on GPUs {self.gpu_group}")

    def process_video(self, video_path, user_question, **video_kwargs):
        """Process a single video"""

        logging.info(f"Worker {self.worker_id} processing: {os.path.basename(video_path)}")
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
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
        processing_time = time.time() - start_time

        logging.info(f"A: {response}")
        logging.info(f"Processing time: {processing_time:.2f}s\n")

        # 메모리 정리
        del pixel_values, num_patches_list
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "video": os.path.basename(video_path),
            "response": response,
            "worker_id": self.worker_id,
        }


@torch.inference_mode()
def process_batch_videos(video_paths, model_path, user_question, **video_kwargs):
    """
    Process videos using two sets of 4 GPUs continuously
    """
    # Define GPU groups (modify these based on your GPU setup)
    gpu_group_1 = [0, 1, 2, 3]  # First set of 4 GPUs
    gpu_group_2 = [4, 5, 6, 7]  # Second set of 4 GPUs

    logging.info("Initializing workers (this may take a few minutes)...")

    # Create and initialize workers in main thread (CRUCIAL for CUDA context)
    workers = [
        ModelWorker(gpu_group_1, model_path, worker_id=1),
        ModelWorker(gpu_group_2, model_path, worker_id=2),
    ]

    logging.info("Both workers initialized! Starting continuous processing...\n")

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

        logging.info(f"Submitted {len(video_paths)} videos to workers. Processing...")

        # Collect results as they complete (no batching)
        from concurrent.futures import as_completed

        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                result = future.result()
                all_results.append(result)

                completed_count = len(all_results)
                total_count = len(video_paths)

                logging.info(
                    f"✓ Worker {result['worker_id']} completed: {result['video']} "
                    f"({completed_count}/{total_count})"
                )

            except Exception as exc:
                logging.error(
                    f"✗ Video {os.path.basename(video_path)} generated an exception: {exc}"
                )

    logging.info(f"\n=== All {len(all_results)} videos processed! ===")
    return all_results


# Usage
if __name__ == "__main__":
    model_path = "OpenGVLab/InternVL3-78B"

    video_path = "/workspace/vss-engine/samples/untitled/"
    video_paths = [
        os.path.join(video_path, "37.mp4"),
        os.path.join(video_path, "36.mp4"),
        os.path.join(video_path, "35.mp4"),
        os.path.join(video_path, "34.mp4"),
    ]

    logging.info("Starting optimized video processing...")
    start_total = time.time()

    # Process with batch size 2 using two sets of 4 GPUs
    results = process_batch_videos(
        video_paths, model_path, user_question, num_segments=None, max_num=2
    )

    total_time = time.time() - start_total

    logging.info("\n=== Processing Summary ===")
    logging.info(f"Total videos: {len(results)}")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average per video: {total_time / len(results):.2f}s")

    for result in results:
        logging.info(f"Video: {result['video']} (Worker {result['worker_id']})")
        logging.info(f"Response: {result['response']}\n")

    logging.info("=== Batch processing completed! ===")
