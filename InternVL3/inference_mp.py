import os
import torch
import math
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in reversed(list(enumerate(num_layers_per_gpu))):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
model_path = 'OpenGVLab/InternVL3-78B'
device_map = split_model(model_path)  # 'InternVL3-78B')
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=2, num_segments=32, max_segments=30):
    # max_num : max number of split per frame
    max_num = max(1, max_num)

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    if num_segments is None or num_segments <= 0:
        num_segments = math.ceil(max_frame * 2 / fps)

    num_segments = min(max_segments // max_num, num_segments)

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=False, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def process_videos_pipeline_parallel(video_paths, model, tokenizer, generation_config, **video_kwargs):
    """
    Process videos one by one (TRUE pipeline parallelism)
    Memory stays constant regardless of video count
    """
    results = []
    
    for i, video_path in enumerate(video_paths):
        print(f"Processing video {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
        
        # Load ONE video at a time (not all together!)
        pixel_values, num_patches_list = load_video(video_path, **video_kwargs)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        
        # Create question for this single video
        video_prefix = ''.join([f'Frame{j+1}: <image>\n' for j in range(len(num_patches_list))])
        
        question = video_prefix + user_question_1
        
        # Process through your pipeline-parallel model
        response, history = model.chat(
            tokenizer, 
            pixel_values, 
            question, 
            generation_config,
            num_patches_list=num_patches_list, 
            history=None, 
            return_history=True
        )

        torch.cuda.empty_cache()

        question = user_question_2
        response, history = model.chat(tokenizer, pixel_values, question, generation_config, 
                                       num_patches_list=num_patches_list, history=history, return_history=True)
        
        results.append({
            'video': os.path.basename(video_path),
            'response': response
        })
        
        print(f'Q: {user_question_2}\nA: {response}\n')
        
        # Important: Clear memory after each video
        del pixel_values, num_patches_list, response, history
        torch.cuda.empty_cache()
    
    return results

generation_config = dict(max_new_tokens=1024, do_sample=True)

user_question_1 = 'Analyze any event in the given media.'

user_question_2 = """Given the media and description below, output a detailed analysis of how you analyzed the scene. 
You should conduct an analysis of what you see and how each component interacts with each other to the provided description.

Follow step-by-step reasoning format;

First step is the planning stage where you list up only few major regions of interest with noun category name (like person) followed by colon and noun phrase with short participial phrase (like person with red shirt) to piece together the activities within the scene. Desired format: ##[Noun Category] : [noun phrase]##;

Second step is reasoning stage where you analyze each region of interest you mentioned at the first step, plus the entire shot. Categorize with each region of interest and the entire shot. If situation is changed as the time goes by, say it with Oh!. Desired format: ##[Noun phrase] : \n- [Reasoning]##;

The last step is conclusion where you assemble all reasoning and give comprehensive answer.
"""


#####
# if True:
# Load multiple videos
video_path = "/workspace/vss-engine/samples/untitled/"
video_paths = [
    os.path.join(video_path, "37.mp4"),
    os.path.join(video_path, "36.mp4"), 
    os.path.join(video_path, "35.mp4"), 
    os.path.join(video_path, "34.mp4")
]
results = process_videos_pipeline_parallel(
    video_paths, 
    model, 
    tokenizer, 
    generation_config,
    num_segments=30,
    max_num=2
)

for result in results:
    video_name = result['video']
    response = result['response']

    print(f'Video: {video_name}\nA:\n {response}')

print("=== Experiment done! ===")