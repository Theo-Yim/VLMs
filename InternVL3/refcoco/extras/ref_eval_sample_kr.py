import math
import numpy as np
import torch
import torch.multiprocessing as mp
import transformers
from copy import deepcopy
from tqdm import tqdm
from torchvision.ops import generalized_box_iou
from datasets import load_dataset
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    GroundingDino2Config,
    GroundingDino2ForObjectDetection,
    GroundingDino2ImageProcessor,
    GroundingDino2Processor,
)


transformers.utils.logging.set_verbosity(40)

Ks = (1, 5, 10)
THRESHOLD = 0.5
NUM_GPUS = 8


def xywh_to_xyxy(box):
    bbox = deepcopy(box)
    bbox[2] = box[0] + box[2]
    bbox[3] = box[1] + box[3]
    return bbox


def prepare_batch(texts, semantics, categories, processor, device):
    length = 1
    batch = {}

    if texts is not None:
        text_encoding_siglip2 = processor.tokenizer_siglip2(
            text=texts, padding="max_length", max_length=64, return_tensors="pt",
        )
        input_ids_siglip2 = text_encoding_siglip2.pop('input_ids').unsqueeze(dim=0).cpu()

        batch["input_ids_siglip2"] = torch.cat([input_ids_siglip2]*length).to(device)
        batch["input_ids_cats"] = torch.cat([torch.tensor([categories])]*length).to(device)
        batch["text_attention_mask_siglip2"] = torch.ones(
            (length, input_ids_siglip2.shape[1]),
            dtype=torch.long,
            device=device,
        )
        batch["input_semantics"] = None
        batch["input_semantic_cats"] = None
        batch["semantic_attention_mask"] = None
    elif semantics is not None:
        semantic_encoding = processor.semantic_processor(
            images=semantics,
            input_data_format="channels_last",
            return_tensors="pt",
        )
        input_semantics = semantic_encoding.pop("pixel_values").unsqueeze(dim=0).cpu()

        batch["input_ids_siglip2"] = None
        batch["input_ids_cats"] = None
        batch["text_attention_mask_siglip2"] = None
        batch["input_semantics"] = torch.cat([input_semantics]*length).to(device)
        batch["input_semantic_cats"] = torch.cat([torch.tensor([categories])]*length).to(device)
        batch["semantic_attention_mask"] = torch.ones(
            (length, len(categories)),
            dtype=torch.long,
            device=device,
        )
    else:
        raise NotImplementedError

    batch["input_ids_bert"] = None
    batch["text_attention_mask_bert"] = None
    batch["input_semantic_masks"] = None

    return batch


def inference(
    pil_image,
    batch,
    processor,
    model,
    id2label,
    label2id,
    device,
    dtype,
):
    _image = pil_image
    s_width, s_height = _image.size

    pixel_values = processor.image_processor(_image, return_tensors="pt").pop("pixel_values")
    batch["pixel_values"] = pixel_values.to(device, dtype=dtype)
    if batch["input_semantics"] is not None:
        batch["input_semantics"] = batch['input_semantics'].to(dtype=dtype)

    with torch.no_grad():
        outputs = model(**batch)

    if batch["input_ids_cats"] is not None:
        input_contrastive_cats = batch["input_ids_cats"]
    elif batch["input_semantic_cats"] is not None:
        input_contrastive_cats = batch["input_semantic_cats"]
    else:
        raise NotImplementedError

    predictions = processor.post_process_grounded_object_detection(
        outputs,
        None,
        input_contrastive_cats,
        box_threshold=0.03,
        multimodal_threshold=0.03,
        target_sizes=torch.tensor([[s_height, s_width]]),
        num_select=300,
        id2label=id2label,
        label2id=label2id,
    )[0]

    tot_boxes = predictions['boxes'].cpu()
    tot_scores = predictions['scores'].cpu()
    tot_labels = predictions['labels'].cpu()
    return tot_boxes, tot_scores, tot_labels


def load_zero(model_name_or_path, ckpt_path):
    image_processor = GroundingDino2ImageProcessor()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_siglip2 = AutoTokenizer.from_pretrained(
        "google/siglip2-base-patch16-256",
        add_eos_token=True,
    )
    semantic_processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-256")
    processor = GroundingDino2Processor(image_processor, tokenizer, tokenizer_siglip2, semantic_processor)
    if "grounding-dino-base" in model_name_or_path:
        backbone_config = {
            "depths": [2, 2, 18, 2],
            "embed_dim": 128,
            "hidden_size": 1024,
            "image_size": 384,
            "model_type": "swin",
            "num_heads": [4, 8, 16, 32],
            "out_features": ["stage2", "stage3", "stage4"],
            "out_indices": [2, 3, 4],
            "window_size": 12,
        }
    elif "grounding-dino-tiny" in model_name_or_path:
        backbone_config = {
            "depths": [2, 2, 6, 2],
            "model_type": "swin",
            "num_heads": [3, 6, 12, 24],
            "out_features": ["stage2", "stage3", "stage4"],
            "out_indices": [2, 3, 4],
        }

    config = GroundingDino2Config(
        label2id={},
        id2label={},
        num_labels=0,
        backbone_config=backbone_config,
        use_crop_resize_planner=False,
    )
    model = GroundingDino2ForObjectDetection.from_pretrained(
        model_name_or_path,
        config=config,
    )

    contrastive_lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    )
    model.model.text_backbone_siglip2.encoder = get_peft_model(
        model.model.text_backbone_siglip2.encoder,
        contrastive_lora_config,
    )
    model.model.semantic_backbone.encoder = get_peft_model(
        model.model.semantic_backbone.encoder,
        contrastive_lora_config,
    )

    model.load_state_dict(
        torch.load(f"{ckpt_path}/pytorch_model.bin", map_location="cpu"), strict=False,
    )

    model.eval()
    return processor, model


def predict_zero(
    image,
    texts,
    processor,
    model,
    device,
    dtype,
):
    id2label = {i: lab for i, lab in enumerate(texts)}
    label2id = {lab: i for i, lab in enumerate(texts)}

    # text prompt
    txt_categories = list(range(len(texts)))
    txt_batch = prepare_batch(
        texts,
        None,
        txt_categories,
        processor,
        device,
    )

    txt_boxes, txt_scores, txt_labels = inference(
        image,
        txt_batch,
        processor,
        model,
        id2label,
        label2id,
        device,
        dtype,
    )

    return txt_boxes, txt_scores, txt_labels


def evaluate(job_idx, ckpt_path, data_path, gpu_id, dtype=torch.float32):
    torch.cuda.set_device(gpu_id)
    print(f"[GPU {gpu_id}] Running {ckpt_path} + {data_path}")

    if "base" in ckpt_path:
        model_name_or_path = "IDEA-Research/grounding-dino-base"
    elif "tiny" in ckpt_path:
        model_name_or_path = "IDEA-Research/grounding-dino-tiny"
    else:
        raise NotImplementedError

    processor_zero, model_zero = load_zero(model_name_or_path, ckpt_path)
    model_zero.to("cuda", dtype=dtype)

    dataset = load_dataset(data_path)
    split_lst = dataset.keys()
    RefScore = {split: {k: 0.0 for k in Ks} for split in split_lst}
    RefCount = {split: 0.0 for split in split_lst}

    for split in split_lst:
        print(f"[GPU {gpu_id}] Processing split: {split}")
        for sample in tqdm(dataset[split], desc=f"[{data_path}/{split}]"):
            image = sample['image'].convert("RGB")
            gt_bbox = xywh_to_xyxy(sample['bbox'])
            t_gt_bbox = torch.tensor(gt_bbox).view(-1, 4).to(dtype=dtype)
            assert len(t_gt_bbox) == 1, "sample['bbox'] should contain only a single box."

            for text in sample['answer']:
                boxes, scores, labels = predict_zero(
                    image,
                    [text],
                    processor_zero,
                    model_zero,
                    device="cuda",
                    dtype=dtype,
                )
                t_boxes = torch.tensor(boxes, dtype=dtype)
                if len(t_boxes) > 0:
                    ious = generalized_box_iou(t_gt_bbox, t_boxes)[0]
                    for k in Ks:
                        if max(ious[:k]) >= THRESHOLD:
                            RefScore[split][k] += 1

                RefCount[split] += 1

    RefValue = {
        split: {k: score / RefCount[split] for k, score in RefScore[split].items()}
        for split in split_lst
    }
    print(f"[GPU {gpu_id}] Done: {ckpt_path.split('/')[0]} + {data_path.split('/')[-1]} => {RefValue}")

    torch.save(
        RefValue,
        "_".join(["RefValue", ckpt_path.split('/')[0], data_path.split('/')[-1]]) + ".pt"
    )


def run_multi_gpu(jobs):
    mp.set_start_method("spawn", force=True)

    for cycle_idx in range(math.ceil(len(jobs) / NUM_GPUS)):
        srt = cycle_idx * NUM_GPUS
        end = (cycle_idx + 1) * NUM_GPUS

        processes = []
        for job_idx in range(srt, end):
            if job_idx == len(jobs):
                break

            ckpt_path, data_path = jobs[job_idx]
            gpu_id = job_idx % NUM_GPUS
            p = mp.Process(target=evaluate, args=(job_idx, ckpt_path, data_path, gpu_id))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == "__main__":
    ckpt_path_lst = [
        "superb_base/epoch_19/",
    ]
    data_path_lst = [
        "lmms-lab/RefCOCO",
        "lmms-lab/RefCOCOplus",
        "lmms-lab/RefCOCOg",
    ]

    jobs = [(ckpt_path, data_path) for ckpt_path in ckpt_path_lst for data_path in data_path_lst]
    run_multi_gpu(jobs)