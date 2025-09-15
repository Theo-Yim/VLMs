"""
InternVL 3.5 Dataset and Data Collator for Training
Handles multimodal data with thinking mode support
"""

import os
import json
from typing import Dict, Any
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from InternVL3.utils.preprocess import load_image

# InternVL 3.5 Thinking Mode System Prompt
THINKING_SYSTEM_PROMPT = """You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.""".strip()


class InternVLDataset(Dataset):
    """Dataset class for InternVL 3.5 training data."""
    
    def __init__(
        self, 
        data_path: str, 
        image_folder: str, 
        tokenizer,
        image_size: int = 448,
        max_dynamic_patches: int = 12
    ):
        super().__init__()
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_dynamic_patches = max_dynamic_patches
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} training samples")
        
    def __len__(self):
        return len(self.data)
    
    def has_thinking_mode(self, response: str) -> bool:
        """Check if response contains thinking tags."""
        return "<think>" in response and "</think>" in response
    
    def format_conversation(self, sample: Dict[str, Any]) -> str:
        """Format conversation with appropriate system prompt."""
        conversations = sample['conversations']
        
        # Determine if we need thinking mode system prompt
        assistant_responses = [conv['value'] for conv in conversations if conv['from'] == 'gpt']
        use_thinking = any(self.has_thinking_mode(resp) for resp in assistant_responses)
        
        # Build conversation
        formatted = ""
        
        # Add system prompt if using thinking mode
        if use_thinking:
            formatted += f"<|im_start|>system\n{THINKING_SYSTEM_PROMPT}<|im_end|>\n"
        
        # Add conversation turns
        for conv in conversations:
            role = "user" if conv['from'] == 'human' else "assistant"
            message = conv['value']
            formatted += f"<|im_start|>{role}\n{message}<|im_end|>\n"
        
        return formatted
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample."""
        sample = self.data[idx]
        
        # Load image if present
        pixel_values = None
        if 'image' in sample and sample['image']:
            image_path = os.path.join(self.image_folder, sample['image'])
            pixel_values = load_image(image_path, self.image_size, self.max_dynamic_patches)
        
        # Format conversation
        text = self.format_conversation(sample)
        
        return {
            'text': text,
            'pixel_values': pixel_values,
            'sample_id': sample.get('id', str(idx))
        }


class MultimodalDataCollator:
    """Custom data collator for handling both text and images."""
    
    def __init__(self, tokenizer, response_template: str = "<|im_start|>assistant\n"):
        self.tokenizer = tokenizer
        self.response_template = response_template
        
    def __call__(self, features):
        # Extract texts and images
        texts = [f['text'] for f in features]
        pixel_values_list = [f.get('pixel_values') for f in features]
        
        # Tokenize texts
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        # Add labels (copy of input_ids for causal LM)
        batch['labels'] = batch['input_ids'].clone()
        
        # Handle images - pad to same number of patches
        if any(pv is not None for pv in pixel_values_list):
            # Find max patches
            max_patches = max(pv.shape[0] if pv is not None else 0 for pv in pixel_values_list)
            
            # Pad all to max_patches
            padded_pixel_values = []
            for pv in pixel_values_list:
                if pv is not None:
                    if pv.shape[0] < max_patches:
                        # Pad with zeros
                        pad_size = max_patches - pv.shape[0]
                        padding = torch.zeros(pad_size, *pv.shape[1:], dtype=pv.dtype)
                        pv = torch.cat([pv, padding], dim=0)
                    padded_pixel_values.append(pv)
                else:
                    # Create dummy image for text-only samples
                    dummy = torch.zeros(max_patches, 3, 448, 448)
                    padded_pixel_values.append(dummy)
            
            batch['pixel_values'] = torch.stack(padded_pixel_values)
        
        return batch


def create_internvl_dataset(data_path: str, image_folder: str, tokenizer, image_size: int = 448, max_dynamic_patches: int = 12):
    """Create InternVL dataset and convert to HuggingFace format."""
    dataset = InternVLDataset(
        data_path=data_path,
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_size=image_size,
        max_dynamic_patches=max_dynamic_patches
    )
    
    # Convert to HuggingFace dataset format
    def data_generator():
        for item in dataset:
            yield {
                "text": item["text"],
                "pixel_values": item["pixel_values"],
                "sample_id": item["sample_id"]
            }
    
    hf_dataset = HFDataset.from_generator(data_generator)
    return hf_dataset
