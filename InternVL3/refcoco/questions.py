data_refcoco = [
    {
        "image_path": "data_tmp/sample_test/COCO_train2014_000000549347_7.jpg",
        "anno": """
- person 1 [152, 188, 179, 208], who is "man turned around", "looking this way", "bottom left: guy turning toward us"
- chair 1 [376, 347, 219, 133], which is "chair in front right being sat in by bald guy looking away", "chair on right, man with white shirt sitting in it", "closest chair on right"
- chair 2 [134, 389, 201, 91], which is "chair of guy looking at you in front", "front L chair", "chair bottom left"
- person 2 [0, 187, 142, 261], who is "person sitting behind green pole on left", "left person not in view"
- person 3 [400, 198, 190, 194], who is "white shirt not facing us", "man on right with back to us", "man in white shirt on right"
""",
        "questions": ["What is the individual near the bottom left doing in the image?", ""],
    },
    {
        "image_path": "data_tmp/sample_test/COCO_train2014_000000549366_1.jpg",
        "anno": """
- zebra 1 [169, 174, 401, 264], which is "zbra front", "front zeb", "front zebra"
- zebra 2 [0, 202, 114, 218], which is "zebra back side cut off", "zebra body on left of screen", "leftmost zeb"
""",
        "questions": ["What portion of the scene is visible from zebra 2's perspective?", ""],
    },
    {
        "image_path": "data_tmp/sample_test/COCO_train2014_000000580905_1.jpg",
        "anno": """

""",
        "questions": ["", ""],
    },
]

question_1 = """Create a couple of questions from the visuals of the {object}."""

# question_1_2 = """Create a few questions from the visuals of the {object}. 
# Fisrt step is to write your plan. Suppose that you have crop tool and you can use it by refering the object above like {Crop object}. For example, you can say "to give an answer to the given question, I need to focus on .... There seems ... and I will crop and zoom in the person 1 to investigate more. {Crop person 1}."
# Then, generate a detailed reasoning from the visual clue or visual relationships, focusing on the region, and based on reasoning, give me final answer. Do not explicitly write the descriptions in your response.
# Output format: {Question: ...\nPlanning: ...\nReasoning: ...\nAnswer: ...}"""

question_2_comm = """<image>
##Object names with descriptions
{anno}
##
You are performing "Multimodal Interleaved Reasoning". During the thinking process, you need to keep an eye on the visual cues in the original image, find regions of the image that help answer the question, and use the "Crop" tool to crop and zoom in for detailed analysis.
When using the tool, you must output a JSON object in the following format:
{{Crop (object name)}}
Ensure that you "Crop" at least once. If you crop the region including multiple objects, list them with commas. For example, {{Crop person 1, desk 3}}, {{Crop person 2}}.
Continue thinking after each operation until you reach the final answer. Output the thinking process within a pair of <think> </think> tags and then output the final answer within a pair of <answer> </answer> tags. Do not use numberred object name outside the crop tool, but use noun phrase with description instead.
Question: {question}""" # 여기의 anno 에는 [bbox] 빼야함.
# question_set2_comm.format(anno="test", question="test")

question_2_followup = "Check your previous answer, if it has logical error, or misuse of Crop tool, or wrong format. After you fix those, give me final answer"


sample_responses_2 = {
    "item2_question1": """
<think>
To determine what portion of the scene is visible from zebra 2's perspective, I need to focus on the position and orientation of zebra 2 in the image. Zebra 2 is on the left side of the image, and its head is facing downward as it grazes.
First, I will crop the image to zoom in on zebra 2 to better understand its orientation and what it might see.
{ "Crop": "zebra 2" }
From this closer view, it's evident that zebra 2's head is lowered toward the ground. This suggests that its line of sight would be primarily towards the ground in front of it and slightly to the side. Given that zebras have eyes positioned on the sides of their heads, zebra 2 likely has a wide field of view that includes the grass it's grazing on and the surrounding area to its left and right.
Next, I will consider the environment around zebra 2. The image shows an open grassland with distant hills. Since zebra 2 is focused on grazing, the main portion of the scene visible from its perspective would be the grassy ground directly in front of it and the immediate surrounding area.
Taking all these observations into account, zebra 2's perspective would mainly include the grass it is eating and the nearby ground area. It might also have a peripheral view of the other zebras and the expansive grassland.
</think>
<answer>
From zebra 2's perspective, the visible portion of the scene primarily includes the grassy ground directly in front of it as it grazes, along with the immediate surrounding area. It may also have a peripheral view of the other zebras and the open grassland.
</answer>""",
    "item1_question1": """
<think>
To determine what the individual near the bottom left is doing, let's identify and zoom in on that area.
{Crop person 1}
From the cropped image, we can see the man turned around, looking towards the camera. He is seated at the dining table on the balcony.
Based on his posture and direction of gaze, it appears he is engaging with the camera or someone near it, possibly acknowledging or interacting with them.
</think>
<answer>
The individual near the bottom left is turned around, looking towards the camera, likely engaging or interacting with someone near it.
</answer>""",
}
