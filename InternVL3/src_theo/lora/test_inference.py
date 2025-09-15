#!/usr/bin/env python3
"""
InternVL 3.5 LoRA Model Inference Script
Test the fine-tuned model with thinking mode support
"""

import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# InternVL 3.5 Thinking Mode System Prompt
THINKING_SYSTEM_PROMPT = """You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.""".strip()

def load_model_with_lora(base_model_path: str, lora_path: str):
    """Load InternVL 3.5 model with LoRA weights."""
    print(f"Loading base model from {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Load base model
    model = AutoModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=True,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Load LoRA weights
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # Merge LoRA weights into base model
    else:
        print(f"LoRA path not found: {lora_path}, using base model only")
    
    return model, tokenizer

def load_and_preprocess_image(image_path: str, size: int = 448):
    """Load and preprocess image for InternVL."""
    try:
        # Simple image preprocessing - you might want to use the original utilities
        import torchvision.transforms as T
        
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16)
        return pixel_values
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def format_conversation(question: str, use_thinking: bool = True):
    """Format conversation for InternVL."""
    formatted = ""
    
    if use_thinking:
        formatted += f"<|im_start|>system\n{THINKING_SYSTEM_PROMPT}<|im_end|>\n"
    
    formatted += f"<|im_start|>user\n{question}<|im_end|>\n"
    formatted += f"<|im_start|>assistant\n"
    
    return formatted

def generate_response(model, tokenizer, question: str, pixel_values=None, use_thinking: bool = True):
    """Generate response using the model."""
    # Format conversation
    conversation = format_conversation(question, use_thinking)
    
    # Tokenize
    inputs = tokenizer(
        conversation,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Add pixel values if provided
    if pixel_values is not None:
        inputs['pixel_values'] = pixel_values.to(model.device)
    
    # Generate
    with torch.no_grad():
        generation_config = {
            'max_new_tokens': 1024,
            'do_sample': True,
            'temperature': 0.6,
            'top_p': 0.9,
            'pad_token_id': tokenizer.eos_token_id,
        }
        
        outputs = model.generate(
            **inputs,
            **generation_config
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    """Main inference function."""
    # Paths - adjust these according to your setup
    BASE_MODEL_PATH = "OpenGVLab/InternVL3_5-8B"
    LORA_PATH = "/workspace/VLMs/InternVL3/checkpoints/internvl3_5_8b_lora_thinking_mode"
    
    # Test data
    IMAGE_PATH = "/workspace/VLMs/Ovis/src_theo/sample_data/sample_small.png"
    
    # Load model
    model, tokenizer = load_model_with_lora(BASE_MODEL_PATH, LORA_PATH)
    model.eval()
    
    print("\n" + "="*60)
    print("🧠 InternVL 3.5 LoRA Model - Thinking Mode Test")
    print("="*60)
    
    # Test 1: Text-only with thinking mode
    print("\n📝 Test 1: Text-only question with thinking mode")
    print("-" * 50)
    question1 = "Explain the importance of multimodal AI models."
    response1 = generate_response(model, tokenizer, question1, use_thinking=True)
    print(f"Question: {question1}")
    print(f"Response: {response1}")
    
    # Test 2: Image + text with thinking mode
    if os.path.exists(IMAGE_PATH):
        print("\n🖼️  Test 2: Image analysis with thinking mode")
        print("-" * 50)
        question2 = "<image>\nAnalyze this mathematical content step by step."
        pixel_values = load_and_preprocess_image(IMAGE_PATH)
        
        if pixel_values is not None:
            response2 = generate_response(model, tokenizer, question2, pixel_values, use_thinking=True)
            print(f"Question: {question2}")
            print(f"Response: {response2}")
        else:
            print("Failed to load image for test 2")
    else:
        print(f"\n⚠️  Image not found: {IMAGE_PATH}")
    
    # Test 3: Same question without thinking mode
    print("\n💭 Test 3: Same question without thinking mode")
    print("-" * 50)
    question3 = "Explain the importance of multimodal AI models."
    response3 = generate_response(model, tokenizer, question3, use_thinking=False)
    print(f"Question: {question3}")
    print(f"Response: {response3}")
    
    print("\n" + "="*60)
    print("✅ Testing completed!")
    print("="*60)

if __name__ == "__main__":
    main()
