"""
InternVL3 Inference for Multiple Images in Batch.
This script does batch-processing using model parallelization across full GPUs.
"""

import gc
import json
import logging
import os
import sys
import time

import torch

from InternVL3.utils.constants import generation_config, user_question
from InternVL3.utils.preprocess import load_image
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
def process_images_batch(img_paths, model, tokenizer, json_output_path, batch_size=8, max_num=12):
    num_success = 0
    if not img_paths:
        logging.warning("No images to process")
        return

    for i in range(0, len(img_paths), batch_size):
        start_time = time.time()
        pixel_values_batch = []
        img_path_batch = []

        for j in range(batch_size):
            if i + j >= len(img_paths):
                break

            img_path = img_paths[i + j]
            try:
                pixel_values = load_image(img_path, max_num=max_num).to(torch.bfloat16).cuda()
                pixel_values_batch.append(pixel_values)
                img_path_batch.append(img_path)
            except Exception as e:
                logging.error(f"Failed to load image {img_path}: {str(e)}")
                continue

        if not pixel_values_batch:
            logging.warning(f"No valid images in batch {i}")
            continue

        pixel_values = torch.cat(pixel_values_batch, dim=0)
        num_patches_list = [pv.size(0) for pv in pixel_values_batch]
        questions = [f"<image>\n{user_question}"] * len(num_patches_list)
        try:
            responses = model.batch_chat(
                tokenizer,
                pixel_values,
                questions=questions,
                generation_config=generation_config,
                num_patches_list=num_patches_list,
            )
        except torch.cuda.OutOfMemoryError:
            logging.error(f"GPU out of memory at batch {i}. Try reducing batch_size.")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        # except Exception as e:
        #     logging.error(f"Error processing batch {i}: {str(e)}")
        #     continue

        processing_time = time.time() - start_time
        logging.info(
            f"Processed {len(num_patches_list)} images of batch {i + 1}/{len(img_path_batch)} in {processing_time:.2f}s"
        )
        num_success += len(num_patches_list)

        # Save responses
        for j, response in enumerate(responses):
            save_response(img_path_batch[j], response, json_output_path)

        # Memory cleanup
        del pixel_values_batch, responses
        torch.cuda.empty_cache()
        gc.collect()
    return num_success  # results


def save_response(img_path, response, json_output_path):
    """Save image analysis response"""
    try:
        output_file = os.path.basename(img_path).replace(".jpg", ".json").replace(".png", ".json")
        output_path = os.path.join(json_output_path, output_file)

        # Extract JSON from response
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1

        if start_idx == -1 or end_idx == 0:
            logging.error(f"No JSON found in response for {img_path}")
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
            logging.error(f"JSON decode error for {img_path}: {str(e)}")
            # Save raw JSON string for debugging
            with open(output_path.replace(".json", "_invalid.json"), "w", encoding="utf-8") as f:
                f.write(json_str)

    except Exception as e:
        logging.error(f"Failed to save response for {img_path}: {str(e)}")


if __name__ == "__main__":
    dataset_list = [
        # "HumanRef",
        "flickr30k",
        "gqa",
        "homeobjects-3K",
    ]
    image_path_root = "./data_tmp/sample_dataset/"

    # Validate base path
    if not os.path.exists(image_path_root):
        logging.error(f"Base image path does not exist: {image_path_root}")
        sys.exit(1)

    json_output_path_root = image_path_root

    batch_size = 8  # Adjust batch size as needed
    max_num = 12  # Maximum number of splits per image

    for dataset in dataset_list:
        image_path = os.path.join(image_path_root, dataset)
        json_output_path = os.path.join(json_output_path_root, f"{dataset}_json")

        # Check if dataset path exists
        if not os.path.exists(image_path):
            logging.warning(f"Dataset path does not exist: {image_path}. Skipping.")
            continue

        if not os.path.exists(json_output_path):
            os.makedirs(json_output_path)

        img_list = os.listdir(image_path)
        img_paths = [
            os.path.join(image_path, img)
            for img in img_list
            if img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        if not img_paths:
            logging.warning(f"No valid images found in {image_path}")
            continue

        logging.info(f"Processing {len(img_paths)} images from {dataset}")
        start_total = time.time()

        try:
            num_success = process_images_batch(
                img_paths, model, tokenizer, json_output_path, batch_size, max_num
            )
            total_time = time.time() - start_total
            logging.info(f"\n=== {dataset} Processing Summary ===")
            logging.info(f"Total images processed: {num_success}/{len(img_paths)}")
            logging.info(f"Total time: {total_time:.2f}s")
            logging.info(f"Average time per image: {total_time / len(img_paths):.2f}s")
        except Exception as e:
            logging.error(f"Failed to process dataset {dataset}: {str(e)}")
            continue

    logging.info("=== All datasets processed! ===")
