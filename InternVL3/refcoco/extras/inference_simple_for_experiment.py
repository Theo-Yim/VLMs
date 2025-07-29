"""
This script is simple inference code for InternVL3. But it is not up-to-date.
"""

import json
import pprint

import torch

from InternVL3.utils.constants import generation_config
from InternVL3.utils.preprocess import load_image, load_video
from InternVL3.utils.processor import load_models, split_model
from InternVL3.questions import questions

model_path = "OpenGVLab/InternVL3-78B"
device_map = split_model(model_path)
model, tokenizer = load_models(model_path, device_map)
# ==================================================

pixel_values = (
    load_image("data_tmp/sample_test/inanam-taipan - closeup1.jpg", max_num=12)
    .to(torch.bfloat16)
    .cuda()
)
question = """<image>
Image has these objects with bboxes:
- Person: [72, 39, 279, 588]
- shelf: [0, 0, 95, 187]

Question: "What color is the shirt?"
Generate reasoning that focuses on relevant regions. When focusing on the region, use the format "Region: <region_name> <bbox>."""
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(
    f"User: {question}\nAssistant: {response}"
)  # Region: Person [72, 39, 279, 588]. The shirt being referred to is worn by the person within this bounding box. Observing the portion of the image that covers the person's torso reveals that the shirt is predominantly red in color.

# ==================================================

pixel_values = (
    load_image("data_tmp/sample_test/COCO_train2014_000000549347_7.jpg", max_num=12)
    .to(torch.bfloat16)
    .cuda()
)

# ===== This is OD question ===== #
# Replace "the area circled in the image" to BBox
question = """<image>
Please carefully observe BBox (x, y, ...) and come up with a caption for the area."""

# ===== This is the question introduced in VLM-R^3 ===== #
pixel_values = (
    load_image("data_tmp/sample_test/COCO_train2014_000000549366_1.jpg", max_num=12)
    .to(torch.bfloat16)
    .cuda()
)
question_r3_1 = """<image>
You are performing "Multimodal Interleaved Reasoning". During the thinking process, you need to keep an eye on the visual cues in the original image, find regions of the image that help answer the question, and use the "Crop" tool to crop and zoom in for detailed analysis.
When using the tool, you must output a JSON object in the following format:
{"bbox_2d": [x1, y1, x2, y2]}
Ensure that you "Crop" at least once.
Continue thinking after each operation until you reach the final answer. Output the thinking process within a pair of <think> </think> tags and then output the final answer within a pair of <answer> </answer> tags.
Question: What portion of the scene is visible from zebra 2's perspective? (context: zebra 2 is the zebra on the left side of the image, partially cut off, with coordinates [0, 202, 114, 218].)
Answer: The visible portion suggests this individual is positioned near the group at the table, likely part of the social setting."""
response = model.chat(tokenizer, pixel_values, question_r3_1, generation_config)
print(f"User: {question_r3_1}\nAssistant: {response}")
# <think>
# Okay, so I need to figure out what part of the scene Zebra 2 can see. Zebra 2 is the one on the left side of the image, partially cut off. Let me look at the image again. The zebras are grazing in a grassy field with some hills in the background.

# First, I should focus on Zebra 2's position. Since it's on the left side and partially cut off, it must be looking into the scene towards the other zebras. Let me crop the image to zoom in on Zebra 2's position to see what's in front of it.

# Using the coordinates [0, 202, 114, 218], I'll crop that area to get a better view. Wait, those coordinates might be for a bounding box. Maybe I should adjust the crop to include where Zebra 2's face is pointing. Let me try cropping around where Zebra 2's head is.

# After cropping, I see that Zebra 2 is facing towards the other zebras in the group. The zebras in the middle are eating grass, and the background shows the open plain with distant hills. So from Zebra 2's perspective, it can see the zebras next to it grazing and the open grassland stretching towards the hills. There's no immediate obstacles like trees or rocks in front, so the view is pretty open except for the group of zebras.

# Therefore, the portion of the scene visible from Zebra 2's perspective would include the other zebras grazing and the expansive grassy field leading up to the hills in the distance.
# <think>

# {"bbox_2d": [0, 202, 114, 218]}

# <think>
# Looking at the cropped area, Zebra 2 is indeed positioned on the left side, partially cut off. Its head is turned slightly towards the center of the image where the other zebras are grazing. From this position, Zebra 2 would have a clear view of the zebras in front of it and the open grassland extending towards the horizon. The background features a vast plain with distant hills, suggesting an unobstructed view of the surrounding environment. The key elements in Zebra 2's visible scene are the grazing zebras and the expansive landscape.
# <think>

# <answer>
# Zebra 2 can see the other zebras grazing and the expansive grassy field extending to the distant hills.
# </answer>

question_r3_1 = """<image>
##Object names with descriptions
- zebra 1, which is "zbra front", "front zeb", "front zebra"
- zebra 2, which is "zebra back side cut off", "zebra body on left of screen", "leftmost zeb"
##
You are performing "Multimodal Interleaved Reasoning". During the thinking process, you need to keep an eye on the visual cues in the original image, find regions of the image that help answer the question, and use the "Crop" tool to crop and zoom in for detailed analysis.
When using the tool, you must output a JSON object in the following format:
{Crop (object name)}
Ensure that you "Crop" at least once. If you crop the region including multiple objects, list them with commas. For example, {Crop zebra 1, zebra 2}, {Crop zebra 1}.
Continue thinking after each operation until you reach the final answer. Output the thinking process within a pair of <think> </think> tags and then output the final answer within a pair of <answer> </answer> tags.
Question: What portion of the scene is visible from zebra 2's perspective?"""
response, history = model.chat(tokenizer, pixel_values, question_r3_1, generation_config, return_history=True)
print(f"User: {question_r3_1}\nAssistant: {response}")
# <think>
# To determine what portion of the scene is visible from zebra 2's perspective, I need to focus on the position and orientation of zebra 2 in the image. Zebra 2 is on the left side of the image, and its head is facing downward as it grazes. 
# First, I will crop the image to zoom in on zebra 2 to better understand its orientation and what it might see.
# {Crop zebra 2}
# From this closer view, it's evident that zebra 2's head is lowered toward the ground. This suggests that its line of sight would be primarily towards the ground in front of it and slightly to the side. Given that zebras have eyes positioned on the sides of their heads, zebra 2 likely has a wide field of view that includes the grass it's grazing on and the surrounding area to its left and right.
# Next, I will consider the environment around zebra 2. The image shows an open grassland with distant hills. Since zebra 2 is focused on grazing, the main portion of the scene visible from its perspective would be the grassy ground directly in front of it and the immediate surrounding area.
# Taking all these observations into account, zebra 2's perspective would mainly include the grass it is eating and the nearby ground area. It might also have a peripheral view of the other zebras and the expansive grassland.
# </think>
# <answer>
# From zebra 2's perspective, the visible portion of the scene primarily includes the grassy ground directly in front of it as it grazes, along with the immediate surrounding area. It may also have a peripheral view of the other zebras and the open grassland.
# </answer>
followup_question = "Check your previous answer, if it has logical error, or misuse of Crop tool, or wrong format. After you fix those, give me final answer"
response2, history2 = model.chat(tokenizer, pixel_values, followup_question, generation_config, history=history, return_history=True)
print(f"Assistant: {response2}")
# <think>
# To determine what portion of the scene is visible from zebra 2's perspective, I need to focus on the position and orientation of zebra 2 in the image. Zebra 2 is on the left side of the image, and its head is facing downward as it grazes.
# First, I will crop the image to zoom in on zebra 2 to better understand its orientation and what it might see.
# { "Crop": "zebra 2" }
# From this closer view, it's evident that zebra 2's head is lowered toward the ground. This suggests that its line of sight would be primarily towards the ground in front of it and slightly to the side. Given that zebras have eyes positioned on the sides of their heads, zebra 2 likely has a wide field of view that includes the grass it's grazing on and the surrounding area to its left and right.
# Next, I will consider the environment around zebra 2. The image shows an open grassland with distant hills. Since zebra 2 is focused on grazing, the main portion of the scene visible from its perspective would be the grassy ground directly in front of it and the immediate surrounding area.
# Taking all these observations into account, zebra 2's perspective would mainly include the grass it is eating and the nearby ground area. It might also have a peripheral view of the other zebras and the expansive grassland.
# </think>
# <answer>
# From zebra 2's perspective, the visible portion of the scene primarily includes the grassy ground directly in front of it as it grazes, along with the immediate surrounding area. It may also have a peripheral view of the other zebras and the open grassland.
# </answer>

# =====

pixel_values = (
    load_image("data_tmp/sample_test/COCO_train2014_000000549347_7.jpg", max_num=12)
    .to(torch.bfloat16)
    .cuda()
)
anno_img1 = """<image>
##Object names with descriptions
- person 1, who is "man turned around", "looking this way", "bottom left: guy turning toward us"
- chair 1, which is "chair in front right being sat in by bald guy looking away", "chair on right, man with white shirt sitting in it", "closest chair on right"
- chair 2, which is "chair of guy looking at you in front", "front L chair", "chair bottom left"
- person 2, who is "person sitting behind green pole on left", "left person not in view"
- person 3, who is "white shirt not facing us", "man on right with back to us", "man in white shirt on right"
##"""

instruction = """
You are performing "Multimodal Interleaved Reasoning". During the thinking process, you need to keep an eye on the visual cues in the original image, find regions of the image that help answer the question, and use the "Crop" tool to crop and zoom in for detailed analysis.
When using the tool, you must output a JSON object in the following format:
{Crop (object name)}
Ensure that you "Crop" at least once. If you crop the region including multiple objects, list them with commas. For example, {Crop person 1, desk 3}, {Crop person 2}.
Continue thinking after each operation until you reach the final answer. Output the thinking process within a pair of <think> </think> tags and then output the final answer within a pair of <answer> </answer> tags. Do not use numberred object name outside the crop tool, but use noun phrase with description instead."""
question_img1_1 = """Question: What is the individual near the bottom left doing in the image?"""
question = anno_img1 + instruction + question_img1_1
response, history = model.chat(tokenizer, pixel_values, question, generation_config, return_history=True)
print(f"User: {question}\nAssistant: {response}")
followup_question = "Check your previous answer, if it has logical error, or misuse of Crop tool, or wrong format. After you fix those, give me final answer"
response2, history2 = model.chat(tokenizer, pixel_values, followup_question, generation_config, history=history, return_history=True)
print(f"Assistant: {response2}")
# <think>
# To determine what the individual near the bottom left is doing, let's identify and zoom in on that area.
# {Crop person 1}
# From the cropped image, we can see the man turned around, looking towards the camera. He is seated at the dining table on the balcony.
# Based on his posture and direction of gaze, it appears he is engaging with the camera or someone near it, possibly acknowledging or interacting with them.
# </think>
# <answer>
# The individual near the bottom left is turned around, looking towards the camera, likely engaging or interacting with someone near it.
# </answer>

question_img1_2 = """Question: What is the individual near the bottom left doing in the image?"""
question = anno_img1 + instruction + question_img1_2
response, history = model.chat(tokenizer, pixel_values, question, generation_config, return_history=True)
print(f"User: {question}\nAssistant: {response}")
followup_question = "Check your previous answer, if it has logical error, or misuse of Crop tool, or wrong format. After you fix those, give me final answer"
response2, history2 = model.chat(tokenizer, pixel_values, followup_question, generation_config, history=history, return_history=True)
print(f"Assistant: {response2}")




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


# =====

pixel_values = (
    load_image("data_tmp/sample_test/COCO_train2014_000000549366_1.jpg", max_num=12)
    .to(torch.bfloat16)
    .cuda()
)
question = """<image>
Image has these objects with bboxes:
- zebra 1 [169, 174, 401, 264], which is "zbra front", "front zeb", "front zebra"
- zebra 2 [0, 202, 114, 218], which is "zebra back side cut off", "zebra body on left of screen", "leftmost zeb"
"""

question_1 = (
    question
    + """Create a few questions from the visuals of the zebra 2 [0, 202, 114, 218]. Generate a detailed reasoning from the visual clue or visual relationships, focusing on the region, and based on reasoning, give me final answer. Do not explicitly write the descriptions in your response.
Output format: {Question: ...\nReasoning: ...\nAnswer: ...}"""
)
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f"User: {question}\nAssistant: {response}")
# {Question: What can be said about the location of zebra 2 compared to the rest of the scene?
# Reasoning: Zebra 2 is positioned on the far left of the image and appears to be cut off, indicating it is at the edge of the frame. Given the layout and setting, it seems zebra 2 is likely outside the main focal point of the group and table.
# Answer: Zebra 2 is located at the left edge of the image, partially cut off, suggesting it is outside the primary focus of the group setting.}

# {Question: How does zebra 2 contribute to the composition of the image?
# Reasoning: The placement of zebra 2 at the edge of the image creates a sense of curiosity or incompleteness. It suggests there might be more to the scene beyond what's visible and adds a dynamic element to the composition.
# Answer: Zebra 2 adds an intriguing edge to the image’s composition by suggesting there is more content beyond the visible frame.}

# {Question: Could zebra 2 be related to any activity or object in the scene?
# Reasoning: Given the primary activity of the people seated around the table, zebra 2 might be incidental or decorative, possibly part of a continuation of the pattern seen in zebra 1.
# Answer: Zebra 2 likely serves a decorative purpose, continuing the pattern observed in zebra 1 rather than being directly related to the activity of the group.}

question_1_2 = (
    question
    + """Create a few questions from the visuals of the zebra 2 [0, 202, 114, 218].
Fisrt step is to write your plan. Suppose that you have crop tool and you can use it by refering the object above like {Crop object}. For example, you can say "to give an answer to the given question, I need to focus on .... There seems ... and I will crop and zoom in the person 1 to investigate more. {Crop person 1}."
Then, generate a detailed reasoning from the visual clue or visual relationships, focusing on the region, and based on reasoning, give me final answer. Do not explicitly write the descriptions in your response.
Output format: {Question: ...\nReasoning: ...\nAnswer: ...}"""
)
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f"User: {question}\nAssistant: {response}")


# =====

pixel_values = (
    load_image("data_tmp/sample_test/COCO_train2014_000000580905_1.jpg", max_num=12)
    .to(torch.bfloat16)
    .cuda()
)
question = """<image>
Image has these objects with bboxes:
- person 1 [118, 300, 175, 181], who is "little girl", "pink", "Your monitor's color's off, friend. ;p Pink shirt girl..."
- person 2 [290, 110, 291, 365], who is "green woman", "person holding umbrella", "woman"
"""
question_1_2 = (
    question
    + """Create a few questions from the visuals of the person 2 [290, 110, 291, 365].
Fisrt step is to write your plan. Suppose that you have crop tool and you can use it by refering the object above like {Crop object}. For example, you can say "to give an answer to the given question, I need to focus on .... There seems ... and I will crop and zoom in the person 1 to investigate more. {Crop person 1}."
Then, generate a detailed reasoning from the visual clue or visual relationships, focusing on the region, and based on reasoning, give me final answer. Do not explicitly write the descriptions in your response.
Output format: {Question: ...\nReasoning: ...\nAnswer: ...}"""
)
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f"User: {question}\nAssistant: {response}")
# {Question: What is the person 2 doing with the umbrella?
# Reasoning: By examining the posture and context around person 2, we can infer their action with the umbrella. Person 2 appears to be holding an opened umbrella. The setting involves outdoor stacked hay bales, and person 2 is extending their arm toward the little girl. To better understand their actions, we should focus on the grip and direction of the umbrella. {Crop person 2}.
# Answer: The person 2 is holding the umbrella open in an outward direction, possibly displaying it or using it as a playful gesture.}

# {Question: What is the attire of person 2?
# Reasoning: Observing the clothing and accessories visible on person 2 will help determine their attire. Person 2 is wearing a specific combination of items that can be identified by focusing on their upper body and accessories. {Crop person 2}.
# Answer: Person 2 is wearing a green long-sleeve shirt, a black vest, blue jeans, and has a black shoulder bag.}

# {Question: What is the probable weather condition based on person 2's actions and clothing?
# Reasoning: To infer the weather condition, we must consider the actions and clothing of person 2. Holding an umbrella typically indicates rain or anticipated rain, while the clothing suggests cooler weather. {Crop person 2}.
# Answer: The probable weather condition is overcast and possibly rainy, as person 2 is holding an umbrella and wearing layers.}


# ===== This is augmented VQA question 2 which might needs localization and CoT reasoning ===== #
question = """<image>
Image has these objects with bboxes and descriptions:
- person 1 [152, 188, 179, 208], who is "man turned around", "looking this way", "bottom left: guy turning toward us"
- chair 1 [376, 347, 219, 133], which is "chair in front right being sat in by bald guy looking away", "chair on right, man with white shirt sitting in it", "closest chair on right"
- chair 2 [134, 389, 201, 91], which is "chair of guy looking at you in front", "front L chair", "chair bottom left"
- person 2 [0, 187, 142, 261], who is "per  son sitting behind green pole on left", "left person not in view"
- person 3 [400, 198, 190, 194], who is "white shirt not facing us", "man on right with back to us", "man in white shirt on right"

Create a few questions from the person 1 [152, 188, 179, 208]. Generate a detailed reasoning that focuses on the region from the visual clue or visual relationships, and based on reasoning, give me final answer. Do not explicitly write the descriptions in your response.
Output format: {Question: ...\nReasoning: ...\nAnswer: ...}"""
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f"User: {question}\nAssistant: {response}")  #
# {Question: What might be the person turned around looking this way at the bottom left doing?
# Reasoning: The individual is seated at a table with several others, suggesting a social setting. Given that they are in a seated position but turned around, they might be engaging in conversation or gesturing towards someone or something outside of the camera's view.
# Answer: They are likely talking to someone or pointing out something outside the frame.}

# {Question: What could be the attire of the "man turned around looking this way"?
# Reasoning: The setting is a relaxed, social gathering outdoors with others dressed in casual attire. The individual in question appears to be dressed in a similar fashion to participate comfortably in this setting.
# Answer: He is likely wearing casual clothing, such as a t-shirt and shorts.}

# {Question: What might be the mood at the gathering based on the actions and posture of the "man turned around looking this way"?
# Reasoning: The individual's posture of turning around to look suggests engagement and interest in the surroundings or the people nearby. This, combined with the casual setting and relaxed postures of others at the table, implies a light-hearted and sociable atmosphere.
# Answer: The mood at the gathering is likely light-hearted and convivial.}

# ==================================================


pixel_values = (
    load_image("data_tmp/sample_test/COCO_train2014_000000549366_1.jpg", max_num=12)
    .to(torch.bfloat16)
    .cuda()
)

# ===== This is augmented VQA question 1 which might needs localization and CoT reasoning ===== #
question = """<image>
Image has these objects with bboxes:
- zebra 1 [169, 174, 401, 264], which is "zbra front", "front zeb", "front zebra"
- zebra 2 [0, 202, 114, 218], which is "zebra back side cut off", "zebra body on left of screen", "leftmost zeb"

Create a few questions from the visuals of the zebra 2 [0, 202, 114, 218]. Generate reasoning from the visual clue or visual relationships, focusing on the region, and based on reasoning, give me final answer. Do not explicitly write the descriptions in your response.
Output format: {Question: ...\nReasoning: ...\nAnswer: ...}"""
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f"User: {question}\nAssistant: {response}")
# {Question: What is the position of the zebra on the left in the field?
# Reasoning: The zebra on the left, only partially visible, appears to be bending down and grazing close to the ground, similar to the other zebras.
# Answer: The zebra is grazing.}

# {Question: Is the zebra on the left fully or partially visible in the image?
# Reasoning: The bounding box for zebra 2 indicates that it is partially cut off on the left side of the screen.
# Answer: The zebra is partially visible.}

# {Question: What activity are the zebras in the foreground likely engaged in?
# Reasoning: The orientation and posture of the zebra on the left suggest it is engaged in a feeding behavior common among zebras.
# Answer: The zebras are likely grazing.}

# ===== This is augmented VQA question 2 which might needs localization and CoT reasoning ===== #
question = """<image>
Image has these objects with bboxes:
- zebra 1 [169, 174, 401, 264], which is "zbra front", "front zeb", "front zebra"
- zebra 2 [0, 202, 114, 218], which is "zebra back side cut off", "zebra body on left of screen", "leftmost zeb"

Create a few questions from the zebra 1 [169, 174, 401, 264]. Generate reasoning that focuses on the region from the visual clue or visual relationships, and based on reasoning, give me final answer. Do not explicitly write the descriptions in your response.
Output format: {Question: ...\nReasoning: ...\nAnswer: ...}"""
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f"User: {question}\nAssistant: {response}")
# {Question: Which zebra is grazing closest to the foreground?
# Reasoning: The zebra labeled as zebra 1 is positioned prominently in the foreground, making it the closest to the viewer.
# Answer: The zebra labeled as zebra 1 is grazing closest to the foreground.}

# {Question: How many zebras are in view within the specified bounding boxes?
# Reasoning: The bounding boxes for zebra 1 and zebra 2 indicate the presence of two distinct zebras within these regions.
# Answer: Two zebras are in view within the specified bounding boxes.}

# {Question: Which zebra's entire body can be most clearly seen?
# Reasoning: The zebra labeled as zebra 1 has most of its body visible in the image, while zebra 2 is partially cut off.
# Answer: The zebra labeled as zebra 1 has its entire body most clearly seen.}

question = """<image>
Image has these objects with bboxes:
- zebra 1 [169, 174, 401, 264], which is "zbra front", "front zeb", "front zebra"
- zebra 2 [0, 202, 114, 218], which is "zebra back side cut off", "zebra body on left of screen", "leftmost zeb"

Create a few questions from the zebra 1 [169, 174, 401, 264]. Generate reasoning that focuses on the region from the visual clue or visual relationships, and based on reasoning, give me final answer. Do not explicitly write the descriptions in your response.
Output format: {Question: ...\nReasoning: ...\nAnswer: ...}"""
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(
    f"User: {question}\nAssistant: {response}"
)  # zebra back side cut off & ZEBRA BODY ON LEFT OF SCREEN

###

pixel_values = (
    load_image("data_tmp/sample_test/COCO_train2014_000000580905_1.jpg", max_num=12)
    .to(torch.bfloat16)
    .cuda()
)
question = """<image>
Image has these objects with bboxes:
- little girl: [118, 300, 175, 181]
- green woman : [ 290, 110, 291, 365]

Question: "What is green woman holding?"
Generate reasoning that focuses on relevant regions. When focusing on the region, use the format "Region: <region_name> <bbox>."""
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f"User: {question}\nAssistant: {response}")  # person holding umbrella

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
