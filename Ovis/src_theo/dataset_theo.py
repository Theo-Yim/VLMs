"""
Simple dataset class for Ovis training
Follows original Ovis logic exactly - clean and simple
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
import torch
from PIL import Image
from torch.utils.data import Dataset

# Import from original Ovis if available
try:
    from ovis.util.constants import IGNORE_ID, IMAGE_TOKEN, VIDEO_TOKEN
    from ovis.util.utils import rank0_print
except ImportError:
    # Fallback constants
    IGNORE_ID = -100
    IMAGE_TOKEN = "<image>"
    VIDEO_TOKEN = "<video>"
    
    def rank0_print(*args):
        print(*args)


class SimpleConversationDataset(Dataset):
    """
    Simple conversation dataset for custom JSON + images
    Follows original Ovis conversation_dataset.py structure exactly
    """
    
    def __init__(self, data_path: str, image_folder: str, model, training_args):
        self.data_path = data_path
        self.image_folder = image_folder
        self.model = model
        self.training_args = training_args
        
        # Load samples (same as original)
        self.samples = self.load()
        rank0_print(f"[{datetime.now()}] Loaded {len(self.samples)} samples from {data_path}")
    
    def load(self):
        """Load samples from JSON file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict) and "conversations" in data:
            return data["conversations"]
        elif isinstance(data, list):
            return data
        else:
            return [data]
    
    def __len__(self):
        return len(self.samples)
    
    def read_image(self, image_path: str):
        """Read image from path (same as original)"""
        try:
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.image_folder, image_path)
            image = Image.open(image_path).convert('RGB')
            return image, None
        except Exception as e:
            return None, e
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[i]
        conversations = sample.get("conversations", [])
        
        # Handle images (same logic as original)
        images = None
        videos = None
        n_image_or_frame = 0
        
        if 'image' in sample and sample['image']:
            images = []
            image_paths = sample['image']
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            
            for image_path in image_paths:
                image, error = self.read_image(image_path)
                if image is None:
                    logging.warning(f"Failed to load image {image_path}: {error}")
                    # Skip this sample rather than creating dummy image
                    continue
                images.append(image)
            n_image_or_frame = len(images)
        
        # Handle videos (treat as image sequence)
        elif 'video' in sample and sample['video']:
            images = []
            frame_paths = sample['video']
            if isinstance(frame_paths, str):
                frame_paths = [frame_paths]
            
            for frame_path in frame_paths:
                image, error = self.read_image(frame_path)
                if image is None:
                    logging.warning(f"Failed to load video frame {frame_path}: {error}")
                    continue
                images.append(image)
            n_image_or_frame = len(images)
        
        # Determine pixel constraints (original Ovis logic)
        if images is None:
            min_pixels = 0
            max_pixels = 0
        elif len(images) == 1:
            min_pixels = self.training_args.single_image_min_pixels
            max_pixels = self.training_args.single_image_max_pixels
        else:
            min_pixels = self.training_args.multiple_image_min_pixels
            max_pixels = self.training_args.multiple_image_max_pixels
        
        if min_pixels < 0:
            min_pixels = self.training_args.single_image_min_pixels
        if max_pixels < 0 and n_image_or_frame > 0:
            max_pixels = max(min_pixels, self.training_args.single_image_max_pixels // n_image_or_frame)
        
        # Use original Ovis preprocessing - this handles ALL special tokens automatically
        try:
            prompt, input_ids, pixel_values, grid_thws, labels = self.model.preprocess_inputs(
                conversations,
                images=images,
                videos=None,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                generation_preface=None,
                return_labels=True,
            )
        except Exception as e:
            logging.error(f"Error preprocessing sample {i}: {e}")
            # Return dummy data for failed samples
            input_ids = torch.tensor([0], dtype=torch.long)
            labels = torch.tensor([IGNORE_ID], dtype=torch.long)
            pixel_values, grid_thws = None, None
        
        # Simple truncation (same as original)
        if pixel_values is None:
            max_length = self.training_args.text_max_length or self.training_args.multimodal_max_length
        else:
            max_length = self.training_args.multimodal_max_length
        
        # Ensure proper tensor shape
        if len(input_ids.shape) > 1:
            input_ids = input_ids[0]
        if len(labels.shape) > 1:
            labels = labels[0]
        
        # Truncate if needed
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        
        return dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
            labels=labels
        )


class DataCollatorForMultimodalDataset:
    """
    Data collator following original Ovis structure exactly
    Simple and clean - no unnecessary validation
    """
    
    def __init__(self, text_tokenizer):
        self.text_tokenizer = text_tokenizer
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract components (same as original)
        input_ids = [instance["input_ids"] for instance in instances]
        pixel_values = [instance["pixel_values"] for instance in instances if instance["pixel_values"] is not None]
        grid_thws = [instance["grid_thws"] for instance in instances if instance["grid_thws"] is not None]
        attention_mask = [instance["attention_mask"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        # Pad sequences (same as original)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.text_tokenizer.pad_token_id
        )
        
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=False
        )
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_ID
        )
        
        # Handle visual data (same as original)
        pixel_values = torch.cat(pixel_values, dim=0) if pixel_values else None
        grid_thws = torch.cat(grid_thws, dim=0) if grid_thws else None
        
        # Add padding token if needed (original logic)
        if input_ids.size(1) > 0 and not torch.any(input_ids == self.text_tokenizer.pad_token_id):
            input_ids = torch.nn.functional.pad(input_ids, (0, 1), value=self.text_tokenizer.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, 1), value=False)
            labels = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_ID)
        
        # Warn if all labels ignored (original logic)
        if torch.all(labels == IGNORE_ID):
            logging.warning("All samples in current batch have ignored labels")
        
        return dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            attention_mask=attention_mask,
            labels=labels
        )