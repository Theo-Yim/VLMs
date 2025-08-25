---
license: apache-2.0
datasets:
- AIDC-AI/Ovis-dataset
library_name: transformers
tags:
- MLLM
pipeline_tag: image-text-to-text
language:
- en
- zh
---
# Ovis2.5-9B
<div align="center">
  <img src=https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/3IK823BZ8w-mz_QfeYkDn.png width="30%"/>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2508.11737"><img src="https://img.shields.io/badge/📖_Technical_Report-Ovis2.5-b31b1b.svg" alt="technical report"></a>
  <a href="https://github.com/AIDC-AI/Ovis"><img src="https://img.shields.io/badge/GitHub-AIDC--AI/Ovis-blue?style=flat&logo=github" alt="code"></a>
  <a href="https://huggingface.co/spaces/AIDC-AI/Ovis2.5-9B"><img src="https://img.shields.io/badge/🎨_HF_Spaces-AIDC--AI/Ovis2.5--9B-lightblack" alt="demo"></a>
  <a href="https://huggingface.co/collections/AIDC-AI/ovis25-689ec1474633b2aab8809335"><img src="https://img.shields.io/badge/🤗_Models-AIDC--AI/Ovis2.5-yellow" alt="models"></a>
</p>

## Introduction


We are pleased to announce the release of **Ovis2.5**, the successor to Ovis2, designed for native-resolution visual perception and enhanced multimodal reasoning. 
It integrates a native-resolution vision transformer (NaViT) that processes images at their original, variable resolutions, eliminating the need for fixed-resolution tiling and preserving both fine details and global layout—crucial for visually dense content such as charts and diagrams. 
To strengthen reasoning, Ovis2.5 is trained not only on linear chain-of-thought (CoT) but also on reflective reasoning, including self-checking and revision. 
This advanced capability is available at inference as an optional *thinking mode*, enabling users to trade latency for higher accuracy on complex inputs.

Building on these advances, **Ovis2.5-9B** achieves an average score of 78.3 on the OpenCompass multimodal evaluation suite (SOTA among open-source MLLMs under 40B parameters), while the lightweight **Ovis2.5-2B** scores 73.9, continuing the “small model, big performance” philosophy for resource-constrained scenarios.


<div align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/kh-1dhZRAduP-P4SkIhXr.png" width="100%" />
</div>

**Key Features**  
* **Native-Resolution Perception** — NaViT vision encoder preserves fine details and global structure without lossy tiling.  
* **Deep-Reasoning Capability** — Optional *thinking mode* for self-checking and revision beyond linear CoT. *Thinking budget* supported.
* **Chart & Document OCR** — State-of-the-art at its scale for complex chart analysis, document understanding (including tables and forms), and OCR.  
* **Broad Task Coverage** — Demonstrates leading performance on image reasoning, video understanding, and grounding benchmarks, showcasing strong general multimodal capability.

<div align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/4kw2RRUhXDiMZdU7wGOfP.png" width="100%" />
</div>

## Quick Inference

Below is a simple example demonstrating how to run Ovis2.5 with a single image input. For accelerated inference with **vLLM**, refer to [GitHub](https://github.com/AIDC-AI/Ovis).

First, install the required dependencies:

```bash
pip install torch==2.4.0 transformers==4.51.3 numpy==1.25.0 pillow==10.3.0 moviepy==1.0.3
pip install flash-attn==2.7.0.post2 --no-build-isolation
```

Then, run the following code.

```python
import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM

MODEL_PATH = "AIDC-AI/Ovis2.5-9B"

# Thinking mode & budget
enable_thinking = True
enable_thinking_budget = True  # Only effective if enable_thinking is True.

# Total tokens for thinking + answer. Ensure: max_new_tokens > thinking_budget + 25
max_new_tokens = 3072
thinking_budget = 2048

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": Image.open(requests.get("https://cdn-uploads.huggingface.co/production/uploads/658a8a837959448ef5500ce5/TIlymOb86R6_Mez3bpmcB.png", stream=True).raw)},
        {"type": "text", "text": "Calculate the sum of the numbers in the middle box in figure (c)."},
    ],
}]

input_ids, pixel_values, grid_thws = model.preprocess_inputs(
    messages=messages,
    add_generation_prompt=True,
    enable_thinking=enable_thinking
)
input_ids = input_ids.cuda()
pixel_values = pixel_values.cuda() if pixel_values is not None else None
grid_thws = grid_thws.cuda() if grid_thws is not None else None

outputs = model.generate(
    inputs=input_ids,
    pixel_values=pixel_values,
    grid_thws=grid_thws,
    enable_thinking=enable_thinking,
    enable_thinking_budget=enable_thinking_budget,
    max_new_tokens=max_new_tokens,
    thinking_budget=thinking_budget,
)

response = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

The thinking and thinking budget logic can be applied in the same way for multi-image, video and pure text scenarios.

**Note (answer extraction for CoT/Thinking):**
To make evaluation and usage easier, we recommend appending a fixed suffix to prompts when using chain-of-thought (CoT) or thinking mode. This ensures the model clearly outputs a final answer that can be extracted programmatically:

```
End your response with 'Final answer: '.
```

For example:

```
Calculate the sum of the numbers in the middle box in figure (c).
End your response with 'Final answer: '.
```


**Tip:** The sections below include an optional streaming helper (compatible with two-phase thinking/budget runs) and extra inference modes: multi-image, video, and text-only.

<details>
<summary>Optional: Streaming (Advanced)</summary>

To support thinking budget, we modified the implementation of the Ovis `generate` method and the default `TextIteratorStreamer` is now incompatible. If you need to stream model output, be sure to use the helper class below.

```python
# --- Budget-aware streamer helper ---
from transformers import TextIteratorStreamer

class BudgetAwareTextStreamer(TextIteratorStreamer):
    """A streamer compatible with Ovis two-phase generation.

    Call .manual_end() after generation to flush any remaining text.
    """
    def manual_end(self):
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""
        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    # Disable base class's end hook; we'll finalize via manual_end()
    def end(self):
        pass
```

Example usage:

```python
streamer = BudgetAwareTextStreamer(
    model.text_tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

outputs = model.generate(
    inputs=input_ids,
    pixel_values=pixel_values,
    grid_thws=grid_thws,
    enable_thinking=enable_thinking,
    enable_thinking_budget=enable_thinking_budget,
    max_new_tokens=max_new_tokens,
    thinking_budget=thinking_budget,
    streamer=streamer
)

```

</details>

<details>
<summary>Example: Multi-image</summary>
Demonstrates how to run inference with multiple images and a related question.

```python
# Multi-image inference
multi_image_files = [
    "/path/to/image_1.jpg",
    "/path/to/image_2.jpg",
    "/path/to/image_3.jpg",
]

content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in multi_image_files]
content.append({"type": "text", "text": "Describe the images."})
messages = [{"role": "user", "content": content}]

input_ids, pixel_values, grid_thws = model.preprocess_inputs(messages=messages, add_generation_prompt=True, max_pixels=896*896)
input_ids = input_ids.cuda()
pixel_values = pixel_values.cuda().to(model.dtype) if pixel_values is not None else None
grid_thws = grid_thws.cuda() if grid_thws is not None else None

with torch.no_grad():
    outputs = model.generate(inputs=input_ids, pixel_values=pixel_values, grid_thws=grid_thws,
                             max_new_tokens=1024, do_sample=True,
                             eos_token_id=model.text_tokenizer.eos_token_id,
                             pad_token_id=model.text_tokenizer.pad_token_id)
print(model.text_tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</details>

<details>
<summary>Example: Video</summary>
Demonstrates how to run inference on a video by sampling multiple frames and asking the model to describe the content.

```python
# Video inference
from moviepy.editor import VideoFileClip  # pip install moviepy==1.0.3

video_file = "/path/to/video_1.mp4"
num_frames = 8

with VideoFileClip(video_file) as clip:
    total_frames = int(clip.fps * clip.duration)
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = [Image.fromarray(clip.get_frame(t)) for t in (idx / clip.fps for idx in indices)]

messages = [{"role": "user", "content": [
    {"type": "video", "video": frames},
    {"type": "text", "text": "Describe this video in detail."},
]}]

input_ids, pixel_values, grid_thws = model.preprocess_inputs(messages=messages, add_generation_prompt=True, max_pixels=896*896)
input_ids = input_ids.cuda()
pixel_values = pixel_values.cuda().to(model.dtype) if pixel_values is not None else None
grid_thws = grid_thws.cuda() if grid_thws is not None else None

with torch.no_grad():
    outputs = model.generate(inputs=input_ids, pixel_values=pixel_values, grid_thws=grid_thws,
                             max_new_tokens=1024, do_sample=True,
                             eos_token_id=model.text_tokenizer.eos_token_id,
                             pad_token_id=model.text_tokenizer.pad_token_id)
print(model.text_tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</details>

<details>
<summary>Example: Text-only</summary>
Demonstrates how to run inference using only text input without any images or videos.

```python
# Text-only inference
messages = [{"role": "user", "content": "Hi, please introduce Yellow Mountain."}]

input_ids, _, _ = model.preprocess_inputs(messages=messages, add_generation_prompt=True)
input_ids = input_ids.cuda()

with torch.no_grad():
    outputs = model.generate(inputs=input_ids, max_new_tokens=1024, do_sample=True,
                             eos_token_id=model.text_tokenizer.eos_token_id,
                             pad_token_id=model.text_tokenizer.pad_token_id)
print(model.text_tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</details>


To enable grounding, end your prompt with `Please provide the bounding box coordinates.` (for boxes) or `Please provide the point coordinates.` (for points). To target a specific object, wrap its description in `<ref>` tags, e.g.:

```text
Find the <ref>red apple</ref> in the image. Please provide the bounding box coordinates.
```

Coordinates are normalized to `[0,1)` with the origin `(0,0)` at the top-left corner of the image.

* Point: `<point>(x,y)</point>`
* Bounding box: `<box>(x1,y1),(x2,y2)</box>` where `(x1,y1)` is top-left, `(x2,y2)` is bottom-right.
* Multiple results can be listed in square brackets: `[<box>(...)</box>,<box>(...)</box> ]`

Example:

```text
The image features a serene scene with <ref>three birds</ref>[
  <box>(0.401,0.526),(0.430,0.557)</box>,
  <box>(0.489,0.494),(0.516,0.526)</box>,
  <box>(0.296,0.529),(0.324,0.576)</box>
] flying in formation against a clear blue sky.
```



## Model Zoo

| Ovis MLLMs |           ViT           |          LLM          |                      Model Weights                      |                           Demo                           |
|:-----------|:-----------------------:|:---------------------:|:-------------------------------------------------------:|:--------------------------------------------------------:|
| Ovis2.5-2B   | siglip2-so400m-patch16-512 | Qwen3-1.7B | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2.5-2B)  | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2.5-2B) |
| Ovis2.5-9B   | siglip2-so400m-patch16-512  |  Qwen3-8B  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2.5-9B)  | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2.5-9B) |


## Performance
We evaluate Ovis2.5 using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), as employed in the OpenCompass multimodal and reasoning evaluation suite.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/LstPS8KqGObo03fCT5ezn.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/idTjwTPtGVO79x9I3iDPN.png)


## Citation
If you find Ovis useful, please consider citing the paper
```bibtex
@article{lu2025ovis25technicalreport,
  title={Ovis2.5 Technical Report}, 
  author={Shiyin Lu and Yang Li and Yu Xia and Yuwei Hu and Shanshan Zhao and Yanqing Ma and Zhichao Wei and Yinglun Li and Lunhao Duan and Jianshan Zhao and Yuxuan Han and Haijun Li and Wanying Chen and Junke Tang and Chengkun Hou and Zhixing Du and Tianli Zhou and Wenjie Zhang and Huping Ding and Jiahe Li and Wen Li and Gui Hu and Yiliang Gu and Siran Yang and Jiamang Wang and Hailong Sun and Yibo Wang and Hui Sun and Jinlong Huang and Yuping He and Shengze Shi and Weihong Zhang and Guodong Zheng and Junpeng Jiang and Sensen Gao and Yi-Feng Wu and Sijia Chen and Yuhui Chen and Qing-Guo Chen and Zhao Xu and Weihua Luo and Kaifu Zhang},
  year={2025},
  journal={arXiv:2508.11737}
}

@article{lu2024ovis,
  title={Ovis: Structural Embedding Alignment for Multimodal Large Language Model},
  author={Shiyin Lu and Yang Li and Qing-Guo Chen and Zhao Xu and Weihua Luo and Kaifu Zhang and Han-Jia Ye},
  year={2024},
  journal={arXiv:2405.20797}
}
```

## License
This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) (SPDX-License-Identifier: Apache-2.0).

## Disclaimer
We used compliance-checking algorithms during the training process, to ensure the compliance of the trained model to the best of our ability. Due to the complexity of the data and the diversity of language model usage scenarios, we cannot guarantee that the model is completely free of copyright issues or improper content. If you believe anything infringes on your rights or generates improper content, please contact us, and we will promptly address the matter.