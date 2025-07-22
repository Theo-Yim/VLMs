"""
This script is simple inference code for InternVL3. But it is not up-to-date.
"""

import torch

from InternVL3.utils.constants import generation_config
from InternVL3.utils.preprocess import load_image, load_video
from InternVL3.utils.processor import load_models, split_model

model_path = "OpenGVLab/InternVL3-78B"
device_map = split_model(model_path)
model, tokenizer = load_models(model_path, device_map)
# ==================================================


if False:
    # set the max number of tiles in `max_num`
    pixel_values = load_image("./examples/image1.jpg", max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    # pure-text conversation (纯文本对话)
    question = "Hello, who are you?"
    response, history = model.chat(
        tokenizer, None, question, generation_config, history=None, return_history=True
    )
    print(f"User: {question}\nAssistant: {response}")

    question = "Can you tell me a story?"
    response, history = model.chat(
        tokenizer, None, question, generation_config, history=history, return_history=True
    )
    print(f"User: {question}\nAssistant: {response}")

    # single-image single-round conversation (单图单轮对话)
    question = "<image>\nPlease describe the image shortly."
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    print(f"User: {question}\nAssistant: {response}")

    # single-image multi-round conversation (单图多轮对话)
    question = "<image>\nPlease describe the image in detail."
    response, history = model.chat(
        tokenizer, pixel_values, question, generation_config, history=None, return_history=True
    )
    print(f"User: {question}\nAssistant: {response}")

    question = "Please write a poem according to the image."
    response, history = model.chat(
        tokenizer, pixel_values, question, generation_config, history=history, return_history=True
    )
    print(f"User: {question}\nAssistant: {response}")

    # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
    pixel_values1 = load_image("./examples/image1.jpg", max_num=12).to(torch.bfloat16).cuda()
    pixel_values2 = load_image("./examples/image2.jpg", max_num=12).to(torch.bfloat16).cuda()
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

    question = "<image>\nDescribe the two images in detail."
    response, history = model.chat(
        tokenizer, pixel_values, question, generation_config, history=None, return_history=True
    )
    print(f"User: {question}\nAssistant: {response}")

    question = "What are the similarities and differences between these two images."
    response, history = model.chat(
        tokenizer, pixel_values, question, generation_config, history=history, return_history=True
    )
    print(f"User: {question}\nAssistant: {response}")

    # multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
    pixel_values1 = load_image("./examples/image1.jpg", max_num=12).to(torch.bfloat16).cuda()
    pixel_values2 = load_image("./examples/image2.jpg", max_num=12).to(torch.bfloat16).cuda()
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

    question = "Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail."
    response, history = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=True,
    )
    print(f"User: {question}\nAssistant: {response}")

    question = "What are the similarities and differences between these two images."
    response, history = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=history,
        return_history=True,
    )
    print(f"User: {question}\nAssistant: {response}")

    # batch inference, single image per sample (单图批处理)
    pixel_values1 = load_image("./examples/image1.jpg", max_num=12).to(torch.bfloat16).cuda()
    pixel_values2 = load_image("./examples/image2.jpg", max_num=12).to(torch.bfloat16).cuda()
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

    questions = ["<image>\nDescribe the image in detail."] * len(num_patches_list)
    responses = model.batch_chat(
        tokenizer,
        pixel_values,
        num_patches_list=num_patches_list,
        questions=questions,
        generation_config=generation_config,
    )
    for question, response in zip(questions, responses):
        print(f"User: {question}\nAssistant: {response}")


#####
# if True:
video_path = "/workspace/vss-engine/samples/untitled/37.mp4"  # './examples/red-panda.mp4'
pixel_values, num_patches_list = load_video(video_path, num_segments=None, max_num=2)
pixel_values = pixel_values.to(torch.bfloat16).cuda()
video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])

user_question = "Analyze any event in the given media."
question = video_prefix + user_question
# Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
# response = model.chat(tokenizer, pixel_values, question, generation_config)
response, history = model.chat(
    tokenizer,
    pixel_values,
    question,
    generation_config,
    num_patches_list=num_patches_list,
    history=None,
    return_history=True,
)
print(f"Q: {user_question}\nA: {response}")

torch.cuda.empty_cache()

user_question = """Given the media and description from previous response, output a detailed analysis of how you analyzed the scene. 
You should conduct an analysis of what you see and how each component interacts with each other to the provided description.

Follow step-by-step reasoning format;

First step is the planning stage where you list up only few major regions of interest with noun category name (like person) followed by semicolon and noun phrase with short participial phrase (like person with red shirt) to piece together the activities within the scene.;

Second step is reasoning stage where you analyze each region of interest you mentioned at the first step, plus the entire shot. Categorize with each region of interest and the entire shot. If situation is changed as the time goes by, say it with Oh!;

The last step is conclusion where you assemble all reasoning and give comprehensive answer.
"""
question = user_question  # video_prefix + user_question + "\nDescription : " + response
# Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
response, history = model.chat(
    tokenizer,
    pixel_values,
    question,
    generation_config,
    num_patches_list=num_patches_list,
    history=history,
    return_history=True,
)
print(f"Q: {user_question}\nA: {response}")

print("=== Experiment done! ===")
