"""
InternVL3 Inference for Multiple Images in Batch.
This script does batch-processing using model parallelization across full GPUs.
"""

import gc
import json
import logging
import os
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
def process_multiple_images_optimized(img_paths, model, tokenizer, user_question, batch_size=8):
    # results = []

    for i in range(0, len(img_paths), batch_size):
        start_time = time.time()
        pixel_values_list = []
        for j in range(batch_size):
            if i + j >= len(img_paths):
                break
            pixel_values = load_image(img_paths[i + j], max_num=12).to(torch.bfloat16).cuda()
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list, dim=0)
        num_patches_list = [pv.size(0) for pv in pixel_values_list]
        questions = [f"<image>\n{user_question}"] * len(num_patches_list)
        responses = model.batch_chat(
            tokenizer,
            pixel_values,
            questions=questions,
            generation_config=generation_config,
            num_patches_list=num_patches_list,
        )
        processing_time = time.time() - start_time
        logging.info(f"Processed {i} ~ {i + batch_size - 1} images in {processing_time:.2f}s")
        # results.extend(responses)

        # Save the first response to a JSON file
        for j in range(len(responses)):
            output_file = img_paths[i + j].replace(".jpg", ".json").replace(".png", ".json")
            output_file = os.path.basename(output_file)

            buf = responses[j][responses[j].find("{") : responses[j].rfind("}") + 1]
            try:
                buf_json = json.loads(buf)
            except json.JSONDecodeError:
                logging.error(f"JSON decode error for image {img_paths[i + j]}")
                with open(
                    os.path.join(json_output_path, output_file),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(buf)
                continue

            with open(os.path.join(json_output_path, output_file), "w", encoding="utf-8") as f:
                json.dump(buf_json, f, ensure_ascii=False, indent=2)

        # 메모리 정리
        del pixel_values_list, responses, buf
        torch.cuda.empty_cache()
        gc.collect()
    return  # results


if __name__ == "__main__":
    dataset_list = [
        "HumanRef",
        "flickr30k",
        "gqa",
        "homeobjects-3K",
    ]
    image_path_root = "./data_tmp/sample_dataset/"
    json_output_path_root = image_path_root

    for dataset in dataset_list:
        image_path = os.path.join(image_path_root, dataset)
        json_output_path = os.path.join(json_output_path_root, f"{dataset}_json")
        if not os.path.exists(json_output_path):
            os.makedirs(json_output_path)

        img_list = os.listdir(image_path)
        img_paths = [
            os.path.join(image_path, img) for img in img_list if img.endswith((".jpg", ".png"))
        ]

        start_total = time.time()
        process_multiple_images_optimized(img_paths, model, tokenizer, user_question=user_question)
        total_time = time.time() - start_total
        logging.info("\n=== Processing Summary ===")
        logging.info(f"Total images processed: {len(img_paths)}")
        logging.info(f"Total time: {total_time:.2f}s")
        logging.info(f"Average time per image: {total_time / len(img_paths):.2f}s")
        logging.info("=== Experiment done! ===")
