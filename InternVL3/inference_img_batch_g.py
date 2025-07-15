"""
InternVL3 Batch Image Processing Script using Two Sets of 4 GPUs.
Compared to inference_img_batch.py, this script optimizes the image processing by using two sets of 4 GPUs
Compared to model parallelization, this script is a combination of model parallelization and batch processing.
"""

import gc
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from InternVL3.utils.constants import generation_config, user_question
from InternVL3.utils.preprocess import load_image
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

    def process_image(self, image_path, user_question, max_num=12):
        start_time = time.time()
        pixel_values = load_image(image_path, max_num=max_num).to(torch.bfloat16).cuda()
        question = f"<image>\n{user_question}"
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        processing_time = time.time() - start_time
        logging.info(f"Processed image {os.path.basename(image_path)} in {processing_time:.2f}s")

        # 메모리 정리
        del pixel_values
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "name": os.path.basename(image_path),
            "response": response,
            "worker_id": self.worker_id,
        }


@torch.inference_mode()
def process_batch_images(image_paths, model_path, user_question, max_num=12):
    """
    Process images using two sets of 4 GPUs continuously
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

    # Submit ALL images to ThreadPool at once for optimal load balancing
    all_results = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all images immediately - let the scheduler handle distribution
        future_to_image = {}

        for image_path in image_paths:
            # Alternate between workers for initial distribution
            worker_idx = len(future_to_image) % 2
            worker = workers[worker_idx]
            future = executor.submit(worker.process_image, image_path, user_question, max_num)
            future_to_image[future] = image_path
        logging.info(f"Submitted {len(image_paths)} images to workers. Processing...")

        # Collect results as they complete (no batching)
        from concurrent.futures import as_completed

        for future in as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                result = future.result()
                all_results.append(result)

                completed_count = len(all_results)
                total_count = len(image_paths)

                logging.info(
                    f"✓ Worker {result['worker_id']} completed: {result['name']} "
                    f"({completed_count}/{total_count})"
                )

            except Exception as exc:
                logging.error(
                    f"✗ Image {os.path.basename(image_path)} generated an exception: {exc}"
                )

    logging.info(f"\n=== All {len(all_results)} images processed! ===")
    return all_results


# Usage
if __name__ == "__main__":
    model_path = "OpenGVLab/InternVL3-78B"

    image_path = "./data_tmp/sample_dataset/HumanRef"
    img_list = os.listdir(image_path)
    img_paths = [
        os.path.join(image_path, img) for img in img_list[:4] if img.endswith((".jpg", ".png"))
    ]

    start_total = time.time()

    responses_all = process_batch_images(img_paths, model_path, user_question, max_num=12)

    total_time = time.time() - start_total

    logging.info("\n=== Processing Summary ===")
    logging.info(f"Total videos: {len(responses_all)}")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Average per video: {total_time / len(responses_all):.2f}s")

    for result in responses_all:
        logging.info(f"Video: {result['name']} (Worker {result['worker_id']})")
        logging.info(f"Response: {result['response']}\n")

    logging.info("=== Batch processing completed! ===")
