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
from transformers import AutoModel, AutoTokenizer
from utils import load_image
import torch.multiprocessing as mp
import math

def evaluate_image_with_openai(image_url, text_query="What is the text in the illustrate?"):
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
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
    print("Chat response:", chat_response.choices[0].message.content)
    return chat_response.choices[0].message.content

def evaluate_image_with_curl(image_url, text_query="What is the text in the illustrate?"):
    # Prepare the JSON payload
    payload = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": text_query}
            ]}
        ]
    }
    
    # Convert payload to JSON string
    json_payload = json.dumps(payload)
    
    # Prepare curl command
    curl_command = [
        "curl",
        "-X", "POST",
        "http://localhost:8000/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-d", json_payload
    ]
    
    try:
        # Execute curl command
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing curl command: {e}")
        print(f"Error output: {e.stderr}")
        return None

def evaluate_image_with_local_model(image_url, model, tokenizer, text_query="What is the text in the illustrate?"):
    # set the max number of tiles in `max_num`
    pixel_values = load_image(image_url, max_num=2).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024*8, do_sample=True, temperature=0.6)

    # single-image single-round conversation
    question = f'<image>\n{text_query}'
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(f'User: {question}\nAssistant: {response}')
    return response

def process_batch(gpu_id, data_items, args):
    # Set the GPU device for this process
    torch.cuda.set_device(gpu_id)
    print(f"Process {gpu_id} using GPU: {torch.cuda.current_device()}")
    
    result_dir = args.result_dir
    purpose = args.purpose
    mode = args.mode
    
    # Load model for this GPU if using local mode
    if mode == "local":
        name = args.name
        if name == "OpenGVLab/InternVL3_5-8B":
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
    else:
        model = None
        tokenizer = None

    for item in tqdm(data_items, desc=f"GPU {gpu_id}"):
        image_file = item['image_file']
        label_id = item['label_id']
        properties = item['annotation_data']['metadata']

        if purpose == "inference":
            prompt = ENGLISH_INFERENCE_PROMPT
        else:
            prompt = f"### Existing Label:\n{properties}" + ENGLISH_TRAIN_PROMPT

        result_path = os.path.join(result_dir, f"{label_id}.txt")
        
        if os.path.exists(result_path) and not args.rerun:
            continue

        if mode == "vllm":
            with open(image_file, "rb") as image_file:
                image_url = base64.b64encode(image_file.read()).decode('utf-8')
            base64_image_url = f"data:image;base64,{image_url}"
            result = evaluate_image_with_openai(base64_image_url, text_query=prompt)
        elif mode == "local":
            result = evaluate_image_with_local_model(image_file, model, tokenizer, text_query=prompt)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if result:
            print(f"GPU {gpu_id} - Response from server:")
            print(result)
            with open(result_path, 'w') as f:
                f.write(result)
        else:
            print(f"GPU {gpu_id} - Failed to get response from server")

def save_ovis_format(result_dir, data_items, image_root):
    ovis_result_dir = os.path.join(result_dir, "ovis")
    os.makedirs(ovis_result_dir, exist_ok=True)

    json_list = []
    for i, item in tqdm(enumerate(data_items), total=len(data_items)):
        image_file = item['image_file']
        label_id = item['label_id']
        result_path = os.path.join(result_dir, f"{label_id}.txt")
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
            "image_directory": image_root,
        }
    }
    with open(os.path.join(ovis_result_dir, "datainfo.json"), 'w') as f:
        json.dump(datainfo, f)
    print("Datainfo saved")

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/nas2/users/sbchoi/kh-practices/lh-poc/")
    parser.add_argument("--result_dir", type=str, default="/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data-result")
    parser.add_argument("--mode", type=str, choices=["vllm", "local"], default="local")
    parser.add_argument("--name", type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--purpose", type=str, default="train", choices=["train", "inference"])
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    args.result_dir = f"{args.result_dir}-{args.purpose}"
    os.makedirs(args.result_dir, exist_ok=True)

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Load all data
    loader = LHDataLoader(args.data_root)
    all_items = list(loader)
    
    if args.debug:
        all_items = all_items[:num_gpus * 10]  # Limit items in debug mode

    # Split data into batches for each GPU
    items_per_gpu = math.ceil(len(all_items) / num_gpus)
    batches = [all_items[i:i + items_per_gpu] for i in range(0, len(all_items), items_per_gpu)]

    # Start multiprocessing
    processes = []
    for gpu_id in range(num_gpus):
        if gpu_id < len(batches):
            p = mp.Process(target=process_batch, args=(gpu_id, batches[gpu_id], args))
            p.start()
            processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()

    save_ovis_format(args.result_dir, all_items, args.image_root)

    print("Inference completed")