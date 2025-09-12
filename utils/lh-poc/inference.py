import os
from openai import OpenAI
import base64
import subprocess
import json
from prompt import ENGLISH_INFERENCE_PROMPT, ENGLISH_TRAIN_PROMPT, R1_SYSTEM_PROMPT
from dataloader import LHDataLoader
from tqdm import tqdm
import argparse
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import gc
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except:
    print("try to update transformers version to 4.50.0")
from utils import load_image
import torch.multiprocessing as mp
import math
import time
from PIL import Image
from qwen_vl_utils import process_vision_info
from google import genai
from google.genai import types

def evaluate_image_with_openai(image_url, api_key="EMPTY", api_base="http://localhost:8000/v1", model="Qwen/Qwen2.5-VL-7B-Instruct", text_query="What is the text in the illustrate?"):
    # Set OpenAI's API key and API base to use vLLM's API server.
    if api_base is None:
        client = OpenAI(
            api_key=api_key,
        )
    else:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    while True:
        try:
            chat_response = client.chat.completions.create(
                model=model,
                messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url
                                    },
                                },
                                {"type": "text", "text": text_query},
                            ],
                        },
                    ],
                )
            break
        except Exception as e:
            if hasattr(e, 'status_code') and e.status_code == 429:
                print("Rate limit exceeded, retrying...")
                time.sleep(1)  # Wait 1 second before retrying
            else:
                raise e

    print("Chat response:", chat_response.choices[0].message.content)
    return chat_response.choices[0].message.content

def evaluate_image_with_gemini(client, model="gemini-2.5-flash", contents=[], config=None):
    while True:
        try:
            chat_response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            break
        except Exception as e:
            if hasattr(e, 'status_code') and e.status_code == 429:
                print("Rate limit exceeded, retrying...")
                time.sleep(1)  # Wait 1 second before retrying
            else:
                raise e

    print("Chat response:", chat_response.text)
    return chat_response.text

def evaluate_image_with_local_model(image_url, model, tokenizer, text_query="What is the text in the illustrate?"):
    # set the max number of tiles in `max_num`
    pixel_values = load_image(image_url, max_num=2).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024*8, do_sample=True, temperature=0.6)

    # single-image single-round conversation
    question = f'<image>\n{text_query}'
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(f'User: {question}\nAssistant: {response}')
    return response

def evaluate_image_with_qwen(image_url, model, processor, text_query="What is the text in the illustrate?"):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                },
                {"type": "text", "text": text_query},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if isinstance(output_text, list):
        output_text = output_text[0]
    return output_text


def evaluate_image_with_ovis(image_url, model, text_query="What is the text in the illustrate?"):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Image.open(image_url),
                },
                {"type": "text", "text": text_query},
                ],
        }
    ]
    enable_thinking = True
    enable_thinking_budget = True
    thinking_budget = 1024*1
    max_new_tokens = 1024*2
    max_pixels = 1024*1024
    min_pixels = 448*448

    input_ids, pixel_values, grid_thws = model.preprocess_inputs(messages=messages, add_generation_prompt=True, enable_thinking=enable_thinking, min_pixels=min_pixels, max_pixels=max_pixels)
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda() if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    # generate output
    with torch.inference_mode():
        gen_kwargs = dict(
            enable_thinking=enable_thinking,
            enable_thinking_budget=enable_thinking_budget,
            thinking_budget=thinking_budget,
            max_new_tokens=max_new_tokens,
            grid_thws=grid_thws,
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, **gen_kwargs)[0]
        output = model.text_tokenizer.decode(output_ids, skip_special_tokens=True)
    torch.cuda.empty_cache()
    gc.collect()
    return output

def process_batch(gpu_id, data_items, args):
    # Set the GPU device for this process
    torch.cuda.set_device(gpu_id)
    print(f"Process {gpu_id} using GPU: {torch.cuda.current_device()}")

    result_dir = args.result_dir
    purpose = args.purpose
    mode = args.mode
    model = None
    
    # Load model for this GPU if using local mode
    for item in tqdm(data_items, desc=f"GPU {gpu_id}"):
        image_file = item['image_file']
        label_id = item['label_id']
        properties = item['annotation_data']['metadata']
        unpredictable = "NO(이미지 판단 불가)" in set(item['annotation_data']['tags'])
        print(unpredictable, item['annotation_data']['tags'])
        if unpredictable:
            defect_present = "Unpredictable"
        else:
            defect_present = "Yes"
        space = properties['공간']
        material_part = properties['부위자재']
        defect_type = properties['하자유형']
        defect_content = properties['하자내용']
        label = f"space: {space}\ndefect_present: {defect_present}\nmaterial_part: {material_part}\ndefect_type: {defect_type}\ndefect_content: {defect_content}"

        if purpose == "test":
            prompt = ENGLISH_INFERENCE_PROMPT
        else:
            prompt = f"### Existing Label:\n{label}" + ENGLISH_TRAIN_PROMPT

        result_path = os.path.join(result_dir, f"{label_id}.txt")
        
        if os.path.exists(result_path) and not args.rerun:
            continue

        if mode == "vllm":
            if args.name == "hyperclova/HCX-005":
                with open(image_file, "rb") as file:
                    image_url = base64.b64encode(file.read()).decode('utf-8')
                base64_image_url = f"data:image/jpeg;base64,{image_url}"
                model = args.name.split("/")[1]
                result = evaluate_image_with_openai(base64_image_url, api_key=os.getenv("HYPERCLOVA_API_KEY"), api_base="https://clovastudio.stream.ntruss.com/v1/openai", model=model, text_query=prompt)
            elif args.name == "openai/gpt-5":
                with open(image_file, "rb") as file:
                    image_url = base64.b64encode(file.read()).decode('utf-8')
                base64_image_url = f"data:image/jpeg;base64,{image_url}"
                model = args.name.split("/")[1]
                result = evaluate_image_with_openai(base64_image_url, api_key=os.getenv("OPENAI_API_KEY"), api_base=None, model=model, text_query=prompt)
            elif args.name == "google/gemini-2.5-flash":
                with open(image_file, "rb") as file:
                    image_bytes = file.read()
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                contents = [
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    ),
                    prompt
                ]
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=1024*4)
                )
                model = args.name.split("/")[1]
                result = evaluate_image_with_gemini(client=client, model=model, contents=contents, config=config)
            elif args.name == "Qwen/Qwen2.5-VL-7B-Instruct":
                with open(image_file, "rb") as file:
                    image_url = base64.b64encode(file.read()).decode('utf-8')
                base64_image_url = f"data:image;base64,{image_url}"
                model = args.name
                result = evaluate_image_with_openai(base64_image_url, api_key="EMPTY", api_base="http://localhost:8000/v1", model=model, text_query=prompt)
        elif mode == "local":
            name = args.name
            if "OpenGVLab/InternVL" in name:
                if model is None:
                    model = AutoModel.from_pretrained(
                        name,
                        torch_dtype=torch.bfloat16,
                        load_in_8bit=False,
                        low_cpu_mem_usage=True,
                        use_flash_attn=True,
                        trust_remote_code=True,
                        device_map=f"cuda:{gpu_id}").eval()
                    model.system_message = R1_SYSTEM_PROMPT
                    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=False)
                result = evaluate_image_with_local_model(image_file, model, tokenizer, text_query=prompt)
            elif "Qwen/Qwen2.5-VL" in name:
                if model is None:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        name,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        device_map=f"cuda:{gpu_id}").eval()
                    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True, use_fast=False)
                result = evaluate_image_with_qwen(image_file, model, processor, text_query=prompt)
            elif "AIDC-AI/Ovis2.5" in name:
                assert transformers.__version__ == "4.51.3", "transformers version must be 4.51.3"
                if model is None:
                    model = AutoModelForCausalLM.from_pretrained(
                        name,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        device_map=f"cuda:{gpu_id}").eval()
                result = evaluate_image_with_ovis(image_file, model, text_query=prompt)
            elif "Superb-AI/Ovis2.5" in name:
                assert transformers.__version__ == "4.51.3", "transformers version must be 4.51.3"
                if model is None:
                    model = AutoModelForCausalLM.from_pretrained(
                        name,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        device_map=f"cuda:{gpu_id}").eval()
                result = evaluate_image_with_ovis(image_file, model, text_query=prompt)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if result:
            print(f"GPU {gpu_id} - Response from server:")
            print(result)
            with open(result_path, 'w') as f:
                f.write(result)
        else:
            print(f"GPU {gpu_id} - Failed to get response from server")

def save_ovis_format(result_dir, data_items, data_root):
    ovis_result_dir = os.path.join(result_dir, "ovis")
    os.makedirs(ovis_result_dir, exist_ok=True)

    json_list = []
    for i, item in tqdm(enumerate(data_items), total=len(data_items)):
        image_file = item['image_file']
        label_id = item['label_id']
        result_path = os.path.join(result_dir, f"{label_id}.txt")
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result = f.read()
            json_list.append({
                "id": i,
                "image_url": image_file,
                "conversations": [
                    {
                        "role": "human",
                        "content": f"<image>\n{ENGLISH_INFERENCE_PROMPT}"
                    },
                    {
                        "role": "gpt",
                        "content": result
                    }
                ]
            })

    local_lh_path = os.path.join(ovis_result_dir, "lh.json")
    with open(local_lh_path, 'w') as f:
        json.dump(json_list, f)

    print("OVIS format saved")
    datainfo = {
        "lh_local": {
            "meta_file": local_lh_path,
            "storage_type": "hybrid",
            "data_format": "conversation",
            "image_directory": data_root,
        }
    }
    with open(os.path.join(ovis_result_dir, "datainfo.json"), 'w') as f:
        json.dump(datainfo, f)
    print("Datainfo saved")

"""
python inference.py --mode local --name OpenGVLab/InternVL3_5-8B --purpose train --rerun --debug
python inference.py --mode local--name OpenGVLab/InternVL3_5-1B --purpose test --rerun
python inference.py --mode local --name Qwen/Qwen2.5-VL-3B-Instruct --purpose test --rerun
python inference.py --mode local --name AIDC-AI/Ovis2.5-1B --purpose test --rerun
python inference.py --mode local --name AIDC-AI/Ovis2.5-9B --purpose test --rerun
python inference.py --mode vllm --name hyperclova/HCX-005 --purpose test --rerun --debug
python inference.py --mode vllm --name Qwen/Qwen2.5-VL-7B-Instruct --purpose train --rerun --debug
python inference.py --mode vllm --name openai/gpt-5 --purpose test --rerun
python inference.py --mode vllm --name google/gemini-2.5-flash --purpose test --rerun --debug
"""
if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/nas2/users/sbchoi/kh-practices/lh-poc/")
    parser.add_argument("--result_dir", type=str, default="/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data-result")
    parser.add_argument("--mode", type=str, choices=["vllm", "local"], default="local")
    parser.add_argument("--name", type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--purpose", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    args.result_dir = os.path.join(f"{args.result_dir}-{args.purpose}", args.name.replace("/", "-"))
    os.makedirs(args.result_dir, exist_ok=True)

    # Get available GPU IDs from CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    num_gpus = torch.cuda.device_count()
    gpu_ids = list(range(num_gpus))
    print(f"Number of available GPUs: {num_gpus}")
    print(f"Using GPU IDs: {gpu_ids}")

    # Load all data
    loader = LHDataLoader(args.data_root, type=args.purpose)
    all_items = list(loader)

    new_all_items = []
    for all_item in all_items:
        label_id = all_item['label_id']
        result_path = os.path.join(args.result_dir, f"{label_id}.txt")
        
        if os.path.exists(result_path) and not args.rerun:
            continue
        new_all_items.append(all_item)
    
    if args.debug:
        new_all_items = new_all_items[:num_gpus * 10]  # Limit items in debug mode

    # Split data into batches for each GPU
    items_per_gpu = math.ceil(len(new_all_items) / num_gpus)
    batches = [new_all_items[i:i + items_per_gpu] for i in range(0, len(new_all_items), items_per_gpu)]

    if num_gpus == 1:
        process_batch(gpu_ids[0], batches[0], args)
    else:
        # Start multiprocessing with spawn method
        mp.set_start_method('spawn', force=True)
        processes = []
        for i, gpu_id in enumerate(gpu_ids):
            if i < len(batches):
                p = mp.Process(target=process_batch, args=(gpu_id, batches[i], args))
                p.start()
                processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()

    save_ovis_format(args.result_dir, new_all_items, args.data_root)

    print("Inference completed")