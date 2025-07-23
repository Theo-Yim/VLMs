"""
This script is simple inference code for InternVL3. But it is not up-to-date.
"""

import json
import pprint

import torch

from InternVL3.utils.constants import generation_config
from InternVL3.utils.preprocess import load_image, load_video
from InternVL3.utils.processor import load_models, split_model
# from InternVL3.refcoco.questions import questions

model_path = "OpenGVLab/InternVL3-78B"
device_map = split_model(model_path)
model, tokenizer = load_models(model_path, device_map)
# ==================================================

# def process_questions():
#     """Process questions and update responses in-place"""
    
#     for i, data in enumerate(questions):
#         # Skip if response already exists
#         if "response" in data and data["response"]:
#             print(f"[{i+1}/{len(questions)}] Skipping {data['image_path']} - response already exists")
#             continue
        
#         image_path = data["image_path"]
#         question = data["question"]
        
#         pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
#         response = model.chat(tokenizer, pixel_values, question, generation_config)
            
#         # Add response to the original data structure
#         data["response"] = response
#         print(f"Processed {image_path}")

#     """Save updated questions back to Python file"""
#     with open("questions_new.py", "w") as f:
#         f.write("questions = ")
#         pprint.pprint(questions, stream=f, indent=2, width=120, depth=None)
    
#     print("Processing complete!")

# def process_questions_json():
#     # Read and parse JSON file
#     with open("questions.json", "r") as f:
#         question_data = json.load(f)  # Changed from f.read() to json.load()

#     # Process each item
#     for data in question_data:
#         # Skip if response already exists
#         if "response" in data and data["response"]:
#             print(f"Skipping {data['image_path']} - response already exists")
#             continue

#         image_path = data["image_path"]
#         question = data["question"]

#         # Process with model
#         pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
#         response = model.chat(tokenizer, pixel_values, question, generation_config)

#         # Add response to the original data structure
#         data["response"] = response
#         print(f"Processed {image_path}")

#     # Save updated JSON back to the same file (or different file)
#     with open("questions.json", "w") as f:  # Updates original file
#         json.dump(question_data, f, indent=2)

#     print("Processing complete!")

# ==================================================


# ===== This is augmented VQA question 1 which might needs localization and CoT reasoning ===== #
pixel_values = (
    load_image("data_tmp/sample_test/COCO_train2014_000000549347_7.jpg", max_num=12)
    .to(torch.bfloat16)
    .cuda()
)
question = """<image>
Image has these objects with bboxes and descriptions:
- person 1 [152, 188, 179, 208], who is "man turned around", "looking this way", "bottom left: guy turning toward us"
- chair 1 [376, 347, 219, 133], which is "chair in front right being sat in by bald guy looking away", "chair on right, man with white shirt sitting in it", "closest chair on right"
- chair 2 [134, 389, 201, 91], which is "chair of guy looking at you in front", "front L chair", "chair bottom left"
- person 2 [0, 187, 142, 261], who is "person sitting behind green pole on left", "left person not in view"
- person 3 [400, 198, 190, 194], who is "white shirt not facing us", "man on right with back to us", "man in white shirt on right"
"""

question_1 = (
    question
    + """Create a few questions from the visuals of the person 1 [152, 188, 179, 208]. Generate a detailed reasoning from the visual clue or visual relationships, focusing on the region, and based on reasoning, give me final answer. Do not explicitly write the descriptions in your response.
Output format: {Question: ...\nReasoning: ...\nAnswer: ...}"""
)
response = model.chat(tokenizer, pixel_values, question_1, generation_config)
print(f"User: {question_1}\nAssistant: {response}")
# {Question: What is the individual near the bottom left doing in the image?
# Reasoning: The person's body language shows that they are turned around, possibly engaging with someone or something outside of the frame. Their posture suggests they might be conversing or observing an event.
# Answer: They are turned around, likely interacting or observing something not visible in the image.

# Question: How is the person near the bottom left dressed?
# Reasoning: The individual appears to be wearing a striped garment, which looks casual and suitable for a relaxed setting like a social gathering.
# Answer: They are dressed in a striped shirt.

# Question: Is the person near the bottom left facing the table?
# Reasoning: The individual's orientation in the image shows they are not directly facing the table but rather turned away from it.
# Answer: No, they are not facing the table.}

question_1_2 = (
    question
    + """Create a few questions from the visuals of the person 1 [152, 188, 179, 208]. 
Fisrt step is to write your plan. Suppose that you have crop tool and you can use it by refering the object above like {Crop object}. For example, you can say "to give an answer to the given question, I need to focus on .... There seems ... and I will crop and zoom in the person 1 to investigate more. {Crop person 1}."
Then, generate a detailed reasoning from the visual clue or visual relationships, focusing on the region, and based on reasoning, give me final answer. Do not explicitly write the descriptions in your response.
Output format: {Question: ...\nPlanning: ...\nReasoning: ...\nAnswer: ...}"""
)
response = model.chat(tokenizer, pixel_values, question_1_2, generation_config)
print(f"User: {question_1_2}\nAssistant: {response}")
# {Question: What is the man turned around looking toward the camera wearing?
# Planning: To answer the question about the man's clothing, focus on the upper body of person 1. {Crop person 1}.
# Reasoning: The man is sitting at the table and facing slightly toward the camera. From the visible part of his outfit, it looks like he is wearing a casual T-shirt.
# Answer: The man is wearing a white T-shirt.}

# {Question: Approximately what age group does the man facing the camera belong to?
# Planning: To estimate the age of the man, I need to examine his facial features and general appearance. {Crop person 1}.
# Reasoning: Through visual cues such as hair color, facial lines, and general demeanor, it seems the man is middle-aged.
# Answer: The man appears to be in the middle-aged group, around 40-50 years old.}

# {Question: What is the general setting of the meal taking place?
# Planning: To determine the setting, consider the background and surrounding elements visible to person 1. {Crop person 1}.
# Reasoning: The presence of a scenic outdoor view, garden chairs, and table setting suggests a social gathering in a garden or patio setting.
# Answer: The meal is taking place in an outdoor garden or patio setting with a scenic countryside view.}
