import os
from openai import OpenAI
import base64
import subprocess
import json
from prompt import prompt
from dataloader import LHDataLoader
from tqdm import tqdm
import argparse

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

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data/K-LH-302 2025-08-22 155843_export")
    parser.add_argument("--image_root", type=str, default="/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data-image/image/20250722")
    parser.add_argument("--result_dir", type=str, default="/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data-result")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    
    data_root = args.data_root
    image_root = args.image_root
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    loader = LHDataLoader(data_root, image_root)
    index = 0
    for item in tqdm(loader):
        data_key = item['meta_data']['data_key']
        label_id = item['label_id']
        result_path = os.path.join(result_dir, f"{label_id}.txt")
        if os.path.exists(result_path) and not args.rerun:
            index += 1
            continue
        image_path = loader.image_root / data_key

        with open(image_path, "rb") as image_file:
            image_url = base64.b64encode(image_file.read()).decode('utf-8')
        base64_image_url = f"data:image;base64,{image_url}"
        result = evaluate_image_with_openai(base64_image_url, text_query=prompt)
        
        if result:
            print("Response from vLLM server:")
            print(result)
            # Save result to file
            with open(result_path, 'w') as f:
                f.write(result)
        else:
            print("Failed to get response from server")