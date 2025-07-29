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

question_1 = """<image>
Create a couple of questions from the visuals of the {object}."""

question_1_2 = """<image>
Image has these objects with bboxes and descriptions:
{anno}
Create two or three questions from the visuals of the {target_ann}. Generate a detailed reasoning from the visual clue or visual relationships, focusing on the region, and based on reasoning, give me final answer. Do not explicitly mention numberred object name or the presence of the descriptions in your response.
Output format: {{Question: ...\nReasoning: ...\nAnswer: ...}}"""

# question_1_2 = """Create a few questions from the visuals of the {object}. 
# Fisrt step is to write your plan. Suppose that you have crop tool and you can use it by refering the object above like {Crop object}. For example, you can say "to give an answer to the given question, I need to focus on .... There seems ... and I will crop and zoom in the person 1 to investigate more. {Crop person 1}."
# Then, generate a detailed reasoning from the visual clue or visual relationships, focusing on the region, and based on reasoning, give me final answer. Do not explicitly write the descriptions in your response.
# Output format: {Question: ...\nPlanning: ...\nReasoning: ...\nAnswer: ...}"""

question_2_comm = """<image>
##Object names with descriptions
{anno}
Since there are many objects, you should use them enough to create a well-aligned response.
##
You are performing "Multimodal Interleaved Reasoning". During the thinking process, keep an eye on the visual cues in the original image, identify regions that help answer the question, and use the "Crop" tool to crop and zoom in for detailed analysis.
When using the tool, you must output a JSON object in the following format:
{{Crop (object name w number)}}
Ensure that you "Crop" at least once. You can simultaneously crop multiple adjacent objects to inspect them sufficiently. If you crop the region including multiple objects, list them with commas. For example, {{Crop person 1, desk 3}}, {{Crop person 2}}.
Continue thinking after each operation until you reach the final answer. Output the thinking process within a pair of <think> </think> tags and then output the final answer within a pair of <answer> </answer> tags. Do not use an object name with a number outside the crop tool, but use a noun phrase with a description instead.
Question: {question}""" # 여기의 anno 에는 [bbox] 빼야함.
# question_set2_comm.format(anno="test", question="test")

question_2_followup = "Check your previous answer, if it has logical error, or misuse of Crop tool, or wrong format. After you fix those, give me final answer" 
# candidate :

question_3_llm = """Refine the response below:

I will provide the original question and model's response. Original question includes object annotations, commands, and question at the end. The response includes thought process to reach to the answer to the question at the end of the original question. Model is meant to generate response only from visuals, not descriptions. Thus, existence of object annotations should not be mentioned in the response, and they should be rephrased. Response can include {crop (object with number)}, but be aware that crop (object with description) is invalid. Response should not mention object with number except for crop tool. Therefore, response should be fixed. Give me the fixed version of response.

#Original Question:
{prompt_1}

#Response:
{response_1}
"""


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

    "item3_question1": """
<think>
Okay, let's tackle this question step by step. The user is asking, "What is the person wearing?" and they've provided several descriptions for three different people in the image. My goal is to identify which person they are referring to and then describe their attire accurately.
First, I need to understand the descriptions given for each person. Person 1 is described as an older woman with her hair in a barrette, wearing a blue jacket. Person 2 is a woman in a gray shirt with a watch on her wrist. Person 3 is a woman in glasses wearing a gray coat and scarf.
Looking at the image, there are three women standing in front of a fruit stand. The woman in the foreground, with her back to the camera, has her hair tied back and is wearing a blue jacket. This matches the description of person 1. The woman on the right, facing the camera, is wearing a gray shirt and has a watch on her left wrist, which aligns with person 2. The woman in the middle, also facing the camera, is wearing glasses, a gray coat, and a scarf, fitting the description of person 3.
Since the user is asking about "the person," I need to determine which one they are referring to. However, the question is general, so I should consider all three. But based on the descriptions, each person has distinct clothing. To ensure accuracy, I'll use the crop tool to zoom in on each person.
First, I'll crop person 1 to confirm the blue jacket and hair accessory. Then, I'll crop person 2 to check the gray shirt and watch. Finally, I'll crop person 3 to verify the gray coat and scarf. This will help me provide a precise answer.
</think>
{Crop person 1}
<think>
Looking at the cropped image of person 1, she is indeed wearing a blue jacket and has her hair tied back with a barrette. This confirms the description. Now, I'll check person 2.
</think>
{Crop person 2}
<think>
In the cropped image of person 2, she is wearing a gray shirt and has a watch on her left wrist. This matches the description provided. Now, I'll verify person 3.
</think>
{Crop person 3}
<think>
The cropped image of person 3 shows she is wearing a gray coat, a scarf, and glasses. This aligns with the given description. 
Since the question is general, I should answer based on the descriptions provided for each person. However, if the user is referring to a specific person, the answer would vary. Given the descriptions, here's the summary:
- The woman with her back to the camera is wearing a blue jacket.
- The woman on the right is wearing a gray shirt and a watch.
- The woman in the middle is wearing a gray coat and a scarf.
Assuming the question refers to one of these individuals, I'll provide the answer based on the most detailed description.
</think>
<answer>
The person is wearing a blue jacket.
</answer>
    """
}
