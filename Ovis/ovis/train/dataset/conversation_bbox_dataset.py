import json
import logging
import random
from datetime import datetime
from typing import Dict

import torch

from ovis.train.dataset.multimodal_dataset import MultimodalDataset
from ovis.util.constants import IGNORE_ID
from ovis.util.utils import rank0_print


sample_data = {
  "id": "sample_00082",
  "image": "250420-61-00029/보수전/2504206100029 전.jpg",
  "bbox(xyxy)": [
    14.480478102199868,
    478.5903444743242,
    1274.384691189663,
    2048.0
  ],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n## You are a professional house construction inspector. Your job is to examine the provided image and determine if there is any defect. You need to guess the space, defect type, defect description, material part, and location in the image of the image.\n\nYou must output the answer in the json format with the following fields:\n- space: [Space name from the list of Spaces]\n- defect_present: Yes / No\n- If Yes, also include:\n  - defect_type: [type from the list of Defect Types]\n  - defect_description: [brief description of the defect]\n  - material_part: [material part from the list of Material Parts]\n  - location_in_image: [describe location within the image, if applicable]\n\n### Instructions\n- Carefully examine each part of the image.\n- Identify the space of the image.\n- Identify the material part of the image.\n- Identify the defect type of the image.\n- Identify the defect description of the image.\n- Identify the location in the image of the image.\n"
    },
    {
      "from": "gpt",
      "value": "<think>\nLooking at the image, the wall shows signs of uneven and incomplete plastering. The plaster appears patchy and uneven, with visible areas where the material has not been applied smoothly or fully. This indicates a finishing defect in the wall's plastering work, specifically on the surface which is intended to be smooth and uniform. The defect is visible in the central part of the wall within the image, where the plastering is incomplete and uneven.\n</think>\n[\n {\n  \"space\": \"Dressing Room\",\n  \"defect_present\": \"Yes\",\n  \"defect_type\": \"Finishing defect\",\n  \"defect_description\": \"Incomplete and uneven plastering\",\n  \"material_part\": \"Wall\",\n  \"location_in_image\": \"Central part of the wall\"\n }\n]"
    }
  ]
},


class ConversationBboxDataset(MultimodalDataset):
    def load(self):
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} from {self.meta_file} begin")
        with open(self.meta_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        rank0_print(f'#samples: {len(samples)}')
        rank0_print(f'sample: {samples[0]}')
        rank0_print(f"[{datetime.now()}] Loading dataset {self.name} end")
        return samples

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[i]
        conversations = sample["conversations"]
        # 이미지/비디오 처리 (기존과 동일)
        images = None
        videos = None
        n_image_or_frame = 0
        if 'image' in sample:
            images = []
            image_paths = sample['image']
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            for image_path in image_paths:
                image, last_e = self.read_image(image_path)
                
                # Random crop using bbox coordinates
                if 'bbox(xyxy)' in sample and image is not None:
                    img_width, img_height = image.size
                    bbox_xyxy = eval(sample['bbox(xyxy)'])
                    
                    # Generate random crop coordinates
                    # x0: random between 0 and bbox[0]
                    x0 = bbox_xyxy[0] # random.randint(0, max(0, int(bbox_xyxy[0])))
                    # y0: random between 0 and bbox[1] 
                    y0 = bbox_xyxy[1] # random.randint(0, max(0, int(bbox_xyxy[1])))
                    # x1: random between bbox[2] and img_width
                    x1 = bbox_xyxy[2] # random.randint(min(img_width, int(bbox_xyxy[2])), img_width)
                    # y1: random between bbox[3] and img_height
                    y1 = bbox_xyxy[3] # random.randint(min(img_height, int(bbox_xyxy[3])), img_height)
                    
                    # Ensure valid crop coordinates
                    x0 = max(0, min(x0, img_width - 1))
                    y0 = max(0, min(y0, img_height - 1))
                    x1 = max(x0 + 1, min(x1, img_width))
                    y1 = max(y0 + 1, min(y1, img_height))
                    
                    # Crop the image
                    image = image.crop((x0, y0, x1, y1))
                
                assert image is not None, f"Failed to read image from {image_path}"
                images.append(image)
            n_image_or_frame = len(images)
        elif 'video' in sample or 'video_frames' in sample:
            video, last_e = self.read_video(sample, min_frames=self.min_frames, max_frames=self.max_frames)
            video_path = sample.get('video') or sample.get('video_frames')
            assert video is not None, f"Failed to read video from {video_path}"
            videos = [video]
            n_image_or_frame = len(video)

        # 픽셀 설정 (기존과 동일)
        if images is None and videos is None:
            min_pixels = 0
            max_pixels = 0
        elif videos is not None:
            min_pixels = self.training_args.video_min_pixels
            max_pixels = self.training_args.video_max_pixels
        elif len(images) == 1:
            min_pixels = self.training_args.single_image_min_pixels
            max_pixels = self.training_args.single_image_max_pixels
        else:
            min_pixels = self.training_args.multiple_image_min_pixels
            max_pixels = self.training_args.multiple_image_max_pixels

        if min_pixels < 0:
            min_pixels = self.training_args.single_image_min_pixels
        if max_pixels < 0:
            max_pixels = max(min_pixels, self.training_args.single_image_max_pixels // n_image_or_frame)

        # ======= 핵심 수정 부분 시작 =======
        
        # 1. conversations → messages 변환
        messages, has_think_tag = self._convert_conversations_to_messages(conversations, images, videos)
        
        # 2. Ovis2.5 preprocess_inputs 호출 - user message만으로 prompt 생성
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        
        # User prompt 생성 (add_generation_prompt=True로 <|im_start|>assistant 추가)
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=user_messages,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            add_generation_prompt=True,
            enable_thinking=has_think_tag
        )
        
        # Assistant 응답 수동으로 추가
        if assistant_messages:
            assistant_text = assistant_messages[0]["content"][0]["text"]
            assistant_tokens = self.text_tokenizer.encode(assistant_text, add_special_tokens=False)
            im_end_tokens = self.text_tokenizer.encode("<|im_end|>", add_special_tokens=False)
            
            # input_ids에 assistant 내용과 im_end 추가
            input_ids_list = input_ids[0].tolist()
            input_ids_list.extend(assistant_tokens)
            input_ids_list.extend(im_end_tokens)
            input_ids = torch.tensor(input_ids_list, dtype=torch.long).unsqueeze(0)
        
        # 3. Labels 생성
        labels = self._generate_labels(input_ids, messages)
        
        # ======= 핵심 수정 부분 끝 =======

        # 길이 제한 (기존과 동일)
        if pixel_values is None:
            input_ids, pixel_values, grid_thws, labels = self.truncate_inputs(
                input_ids, pixel_values, grid_thws, labels, max_length=self.training_args.text_max_length
            )
        else:
            input_ids, pixel_values, grid_thws, labels = self.truncate_inputs(
                input_ids, pixel_values, grid_thws, labels, max_length=self.training_args.multimodal_max_length
            )
        
        assert self.text_tokenizer.pad_token_id not in input_ids, \
            f"The sample's text contains a padding token: `{self.text_tokenizer.pad_token}`"

        del sample
        return dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            attention_mask=torch.full_like(input_ids, fill_value=True, dtype=torch.bool),
            labels=labels
        )
    
    def _convert_conversations_to_messages(self, conversations, images, videos):
        """conversations 형태를 messages 형태로 변환"""
        messages = []
        has_think_tag = False

        assert len(conversations) < 3, "Multi-turn conversations are not supported yet."
        
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            
            if role == "user":
                # User 메시지에 이미지/비디오 추가
                content = []
                
                # 이미지 추가
                if images is not None:
                    for image in images:
                        content.append({"type": "image", "image": image})
                
                # 비디오 추가
                if videos is not None:
                    content.append({"type": "video", "video": videos[0]})
                
                # 텍스트 추가
                content.append({"type": "text", "text": conv["value"]})
                
                messages.append({
                    "role": role,
                    "content": content
                })
            else:
                # Assistant 메시지
                messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": conv["value"]}]
                })
                has_think_tag = conv["value"].startswith("<think>")
        
        return messages, has_think_tag
    
    def _generate_labels(self, input_ids, messages):
        """input_ids에서 labels 생성 - assistant 부분만 학습 대상으로 설정"""
        input_ids_list = input_ids[0].tolist()  # [1, seq_len] → [seq_len]
        labels = [IGNORE_ID] * len(input_ids_list)
        
        try:
            # <|im_start|>assistant 토큰 찾기
            im_start_token = "<|im_start|>"
            assistant_token = "assistant"
            
            # 토큰화해서 시퀀스 찾기
            im_start_ids = self.text_tokenizer.encode(im_start_token, add_special_tokens=False)
            assistant_ids = self.text_tokenizer.encode(assistant_token, add_special_tokens=False)
            
            # <|im_start|>assistant 시퀀스 찾기
            target_sequence = im_start_ids + assistant_ids
            
            # input_ids에서 이 시퀀스 찾기
            assistant_start_pos = None
            for i in range(len(input_ids_list) - len(target_sequence) + 1):
                if input_ids_list[i:i+len(target_sequence)] == target_sequence:
                    assistant_start_pos = i + len(target_sequence)
                    break
            
            if assistant_start_pos is not None:
                # 줄바꿈 토큰 건너뛰기 (선택적)
                while (assistant_start_pos < len(input_ids_list) and 
                       self.text_tokenizer.decode([input_ids_list[assistant_start_pos]]).strip() == ""):
                    assistant_start_pos += 1
                
                # <|im_start|>assistant 이후부터 학습 대상
                # 단, <|im_end|> 토큰은 제외
                im_end_ids = self.text_tokenizer.encode("<|im_end|>", add_special_tokens=False)
                
                # assistant_start_pos부터 끝까지 또는 <|im_end|>까지
                end_pos = len(input_ids_list)
                for i in range(assistant_start_pos, len(input_ids_list) - len(im_end_ids) + 1):
                    if input_ids_list[i:i+len(im_end_ids)] == im_end_ids:
                        end_pos = i
                        break
                
                # Assistant 응답 부분만 학습 대상으로 설정
                if end_pos > assistant_start_pos:
                    labels[assistant_start_pos:end_pos] = input_ids_list[assistant_start_pos:end_pos]
                    
        except Exception as e:
            # 에러 발생 시 전체를 IGNORE_ID로 유지 (안전)
            logging.warning(f"Failed to generate labels: {e}. Using all IGNORE_ID.")
        
        return torch.tensor(labels, dtype=torch.long).unsqueeze(0)  # [seq_len] → [1, seq_len]