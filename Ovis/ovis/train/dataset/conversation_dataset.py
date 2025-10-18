import json
import logging
from datetime import datetime
from typing import Dict

import torch

from ovis.train.dataset.multimodal_dataset import MultimodalDataset
from ovis.util.constants import IGNORE_ID
from ovis.util.utils import rank0_print

# Use the universal ToolRegistry from tool_base
try:
    from src_theo.tools.tool_base import ToolRegistry
except ImportError:
    logging.warning("ToolRegistry not available - tool calls will not be processed during training")

    # Fallback dummy registry
    class ToolRegistry:
        """Dummy registry when real one is not available"""

        def __init__(self):
            self.tools = {}

        def detect_tool_calls(self, text: str) -> bool:
            return False

        def process_tools_for_training(self, text: str, image):
            return [], text


class ConversationDataset(MultimodalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize tool processors for handling multimodal content
        self.tool_processors = ToolRegistry()

    def _has_tool_calls(self, text):
        """Check if text contains any supported tool calls"""
        if not text or not isinstance(text, str):
            return False
        return self.tool_processors.detect_tool_calls(text)

    def load(self):
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} from {self.meta_file} begin")
        with open(self.meta_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
        rank0_print(f"#samples: {len(samples)}")
        rank0_print(f"sample: {samples[0]}")
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} end")
        return samples

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[i]
        conversations = sample["conversations"]

        # Load images/videos
        images = None
        videos = None
        n_image_or_frame = 0
        if "image" in sample:
            images = []
            image_paths = sample["image"]
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            for image_path in image_paths:
                image, last_e = self.read_image(image_path)
                assert image is not None, f"Failed to read image from {image_path}"
                images.append(image)
            n_image_or_frame = len(images)
        elif "video" in sample or "video_frames" in sample:
            video, last_e = self.read_video(
                sample, min_frames=self.min_frames, max_frames=self.max_frames
            )
            video_path = sample.get("video") or sample.get("video_frames")
            assert video is not None, f"Failed to read video from {video_path}"
            videos = [video]
            n_image_or_frame = len(video)

        # ========== PRE-PROCESS TOOL CALLS ==========
        tool_result_images = []
        processed_conversations = []

        for conv in conversations:
            if conv["from"] == "gpt" and self._has_tool_calls(conv["value"]):
                # Assistant message with tool calls - pre-process them
                original_image = images[0] if images else None

                # Universal tool processing
                result_images, cleaned_text = self.tool_processors.process_tools_for_training(
                    conv["value"], original_image
                )

                tool_result_images.extend(result_images)
                processed_conversations.append({"from": conv["from"], "value": cleaned_text})
            else:
                processed_conversations.append(conv)

        # Combine all images
        all_images = []
        if images:
            all_images.extend(images)
        all_images.extend(tool_result_images)

        if all_images:
            n_image_or_frame = len(all_images)

        # Set pixel values
        if not all_images and videos is None:
            min_pixels = 0
            max_pixels = 0
        elif videos is not None:
            min_pixels = self.training_args.video_min_pixels
            max_pixels = self.training_args.video_max_pixels
        elif len(all_images) == 1:
            min_pixels = self.training_args.single_image_min_pixels
            max_pixels = self.training_args.single_image_max_pixels
        else:
            min_pixels = self.training_args.multiple_image_min_pixels
            max_pixels = self.training_args.multiple_image_max_pixels

        if min_pixels < 0:
            min_pixels = self.training_args.single_image_min_pixels
        if max_pixels < 0:
            max_pixels = max(
                min_pixels, self.training_args.single_image_max_pixels // n_image_or_frame
            )

        # Convert to messages (tools already processed, so skip_tool_processing=True)
        messages, has_think_tag = self._convert_conversations_to_messages(
            processed_conversations, all_images, videos
        )

        # Generate input_ids
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]

        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=user_messages,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            add_generation_prompt=True,
            enable_thinking=has_think_tag,
        )

        # Add assistant response
        if assistant_messages:
            assistant_content = assistant_messages[0]["content"]
            assistant_tokens = []

            if isinstance(assistant_content, list):
                for content_item in assistant_content:
                    if content_item["type"] == "text":
                        text_tokens = self.text_tokenizer.encode(
                            content_item["text"], add_special_tokens=False
                        )
                        assistant_tokens.extend(text_tokens)
            else:
                assistant_tokens = self.text_tokenizer.encode(
                    assistant_content, add_special_tokens=False
                )

            im_end_tokens = self.text_tokenizer.encode("<|im_end|>", add_special_tokens=False)

            input_ids_list = input_ids[0].tolist()
            input_ids_list.extend(assistant_tokens)
            input_ids_list.extend(im_end_tokens)
            input_ids = torch.tensor(input_ids_list, dtype=torch.long).unsqueeze(0)

        # Generate labels
        labels = self._generate_labels(input_ids, messages)

        # Truncate
        if pixel_values is None:
            input_ids, pixel_values, grid_thws, labels = self.truncate_inputs(
                input_ids,
                pixel_values,
                grid_thws,
                labels,
                max_length=self.training_args.text_max_length,
            )
        else:
            input_ids, pixel_values, grid_thws, labels = self.truncate_inputs(
                input_ids,
                pixel_values,
                grid_thws,
                labels,
                max_length=self.training_args.multimodal_max_length,
            )

        assert self.text_tokenizer.pad_token_id not in input_ids

        del sample
        return dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            attention_mask=torch.full_like(input_ids, fill_value=True, dtype=torch.bool),
            labels=labels,
        )

    def _convert_conversations_to_messages(self, conversations, images, videos):
        """
        Convert conversations to messages format.

        """
        messages = []
        has_think_tag = False

        assert len(conversations) < 3 or (
            len(conversations) == 3 and conversations[0]["from"] == "system"
        ), "Multi-turn conversations are not supported yet."

        for conv in conversations:
            # Handle system, user, and assistant roles separately
            if conv["from"] == "system":
                # System message - add as system role
                messages.append({"role": "system", "content": conv["value"]})
                continue

            role = "user" if conv["from"] == "human" else "assistant"

            if role == "user":
                # User message - add images/videos
                content = []

                if images is not None:
                    for image in images:
                        content.append({"type": "image", "image": image})

                if videos is not None:
                    content.append({"type": "video", "video": videos[0]})

                content.append({"type": "text", "text": conv["value"]})

                messages.append({"role": role, "content": content})
            else:
                # Assistant message
                text = conv["value"]

                # Check for <think> tag
                if text.startswith("<think>"):
                    has_think_tag = True

                # Tool processing
                messages.append({"role": role, "content": [{"type": "text", "text": text}]})

        return messages, has_think_tag

    def _find_and_mask_sequence(self, labels, input_ids_list, start_marker, end_marker, search_start=0):
        """
        Find and mask token sequences between start_marker and end_marker.

        Args:
            labels: Label list to modify
            input_ids_list: Input token IDs
            start_marker: Start marker string (e.g., "<tool_response>")
            end_marker: End marker string (e.g., "</tool_response>")
            search_start: Position to start searching from

        Returns:
            Number of sequences masked
        """
        try:
            start_ids = self.text_tokenizer.encode(start_marker, add_special_tokens=False)
            end_ids = self.text_tokenizer.encode(end_marker, add_special_tokens=False)

            masked_count = 0
            search_pos = search_start

            while search_pos < len(input_ids_list):
                # Find start marker
                start_found = False
                for i in range(search_pos, len(input_ids_list) - len(start_ids) + 1):
                    if input_ids_list[i : i + len(start_ids)] == start_ids:
                        start_pos = i
                        start_found = True
                        break

                if not start_found:
                    break

                # Find end marker after start
                end_found = False
                for i in range(start_pos + len(start_ids), len(input_ids_list) - len(end_ids) + 1):
                    if input_ids_list[i : i + len(end_ids)] == end_ids:
                        end_pos = i + len(end_ids)
                        end_found = True
                        break

                if end_found:
                    # Mask the entire sequence including markers
                    for i in range(start_pos, end_pos):
                        labels[i] = IGNORE_ID
                    masked_count += 1
                    search_pos = end_pos
                else:
                    # End marker not found, stop searching
                    break

            return masked_count

        except Exception as e:
            logging.warning(f"Failed to mask sequence {start_marker}...{end_marker}: {e}")
            return 0

    def _generate_labels(self, input_ids, messages):
        """
        Generate labels from input_ids - only assistant parts are training targets.

        IMPORTANT: Masks <tool_response>...</tool_response> to prevent model from
        learning to generate tool responses (they should come from external system).
        """
        input_ids_list = input_ids[0].tolist()  # [1, seq_len] → [seq_len]
        labels = [IGNORE_ID] * len(input_ids_list)

        try:
            # Find <|im_start|>assistant tokens
            im_start_token = "<|im_start|>"
            assistant_token = "assistant"

            # Tokenize and find sequence
            im_start_ids = self.text_tokenizer.encode(im_start_token, add_special_tokens=False)
            assistant_ids = self.text_tokenizer.encode(assistant_token, add_special_tokens=False)

            # Find <|im_start|>assistant sequence
            target_sequence = im_start_ids + assistant_ids

            # Find this sequence in input_ids
            assistant_start_pos = None
            for i in range(len(input_ids_list) - len(target_sequence) + 1):
                if input_ids_list[i : i + len(target_sequence)] == target_sequence:
                    assistant_start_pos = i + len(target_sequence)
                    break

            if assistant_start_pos is not None:
                # Skip newline tokens (optional)
                while (
                    assistant_start_pos < len(input_ids_list)
                    and self.text_tokenizer.decode([input_ids_list[assistant_start_pos]]).strip()
                    == ""
                ):
                    assistant_start_pos += 1

                # Set training target from <|im_start|>assistant onwards
                # Exclude <|im_end|> token
                im_end_ids = self.text_tokenizer.encode("<|im_end|>", add_special_tokens=False)

                # From assistant_start_pos to end or until <|im_end|>
                end_pos = len(input_ids_list)
                for i in range(assistant_start_pos, len(input_ids_list) - len(im_end_ids) + 1):
                    if input_ids_list[i : i + len(im_end_ids)] == im_end_ids:
                        end_pos = i
                        break

                # Set assistant response as training target
                if end_pos > assistant_start_pos:
                    labels[assistant_start_pos:end_pos] = input_ids_list[
                        assistant_start_pos:end_pos
                    ]

                    # ========== MASK TOOL RESPONSES ==========
                    # Mask <tool_response>...</tool_response> regions
                    # These are provided by external system, model should NOT learn to generate them
                    masked_count = self._find_and_mask_sequence(
                        labels,
                        input_ids_list,
                        "<tool_response>",
                        "</tool_response>",
                        search_start=assistant_start_pos
                    )

                    if masked_count > 0:
                        logging.debug(f"Masked {masked_count} <tool_response> sequences")

        except Exception as e:
            # On error, keep entire sequence as IGNORE_ID (safe)
            logging.warning(f"Failed to generate labels: {e}. Using all IGNORE_ID.")

        return torch.tensor(labels, dtype=torch.long).unsqueeze(0)  # [seq_len] → [1, seq_len]
