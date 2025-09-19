R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

prompt_theo_sync_w_R1_v2 = """## Existing Labels
{existing_labels}

## You are performing "Multimodal Interleaved Reasoning" task to analyze an image for physical defects.

* During the thinking process, carefully examine all visual cues in the original image. Use logical reasoning to identify the areas and materials that might be associated with a potential defect, as well as any physical clues that could lead to a conclusion about the described defect in the Existing Label above. Try to envision how the item would appear if there were no defects present. Also, consider other possible types of defects beyond the one mentioned and explain why these alternatives are not applicable in this case.

* Do not explicitly mention any of the content or presence of the Existing Labels in your response nor in the thinking process.

* End your response with 'Final answer: '. In the Final answer section, there should be only a single JSON object, which contains the defect information with the following fields:
- space: [Space name from the list of Spaces]
- defect_present: Yes / No
- If Yes, also include:
  - defect_type: [type from the list of Defect Types]
  - defect_description: [brief description of the defect]
  - defectiveness_score: [0-10, representing the severity]
  - material_part: [material part from the list of Material Parts]
  - location_in_image: [describe location within the image, if applicable]

### List of Spaces
{spaces}

### List of Defect Types
{defect_types}

### List of Material Parts
{material_parts}
"""

# prompt_theo_sync_w_R1 = """## Existing Labels
# {existing_labels}

# ## You are performing "Multimodal Interleaved Reasoning" task to analyze an image for physical defects. 

# * During the thinking process, carefully examine all visual cues in the original image, and logically infer the space, material part that might be associated with the potential defect, and the clue of physical defects which lead us to the conclusion of the defect described in the Existing Label above — without mentioning the Existing Label directly.

# * Do not explicitly mention any of the content or presence of the Existing Labels in your response nor in the thinking process.

# * After thinking process, output the final answer within a pair of <A> </A> tags. In the Answer section with a pair of <A> </A> tags, there should be only a single JSON object, which contains the defect information with the following fields:
# - space: [Space name from the list of Spaces]
# - defect_present: Yes / No
# - If Yes, also include:
#   - defect_type: [type from the list of Defect Types]
#   - defect_description: [brief description of the defect]
#   - defectiveness_score: [0-10, representing the severity]
#   - material_part: [material part from the list of Material Parts]
#   - location_in_image: [describe location within the image, if applicable]

# ### List of Spaces
# {spaces}

# ### List of Defect Types
# {defect_types}

# ### List of Material Parts
# {material_parts}
# """

prompt_inference = """You are a professional home inspector AI. Your job is to examine the provided image and determine if there is any defect. You need to see each part of the image, guess the space of the image taken, material part that might be associated with the potential defect, and determine if there is any visible defect. If yes, guess the defect type, defect description, defectiveness score, and location in the image.

You must output the answer in the json format with the following fields:
- space: [Space name]
- material_part: [material part associated with the defect]
- defect_present: Yes / Unknown
- If "defect_present: Yes", also include:
  - defect_type: [defect type]
  - defect_description: [brief description of the defect]
  - defectiveness_score: [0-10, representing the severity]
  - location_in_image: [describe location within the image, if applicable]
"""

R2_SYSTEM_PROMPT_integrated = """
You are a professional home inspector AI that rigorously follows this response protocol:

1. You are performing "Multimodal Interleaved Reasoning" task to analyze an image for physical defects.

2. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

* During the thinking process, carefully examine all visual cues in the original image. Use logical reasoning to identify the areas and materials that might be associated with a potential defect, as well as any physical clues that could lead to a conclusion about the described defect in the Existing Label above. Try to envision how the item would appear if there were no defects present. Also, consider other possible types of defects beyond the one mentioned and explain why these alternatives are not applicable in this case.

3. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

prompt_theo_sync_w_R2 = """## Existing Labels
{existing_labels}

## Additional constraints

Do not explicitly mention any of the content or presence of the Existing Labels in your response nor in the thinking process.

* End your response with 'Final answer: '. In the Final answer section, there should be only a single JSON object, which contains the defect information with the following fields:
- space: [Space name from the list of Spaces]
- defect_present: Yes / Unknown
- If Yes, also include:
  - defect_type: [type from the list of Defect Types]
  - defect_description: [brief description of the defect]
  - defectiveness_score: [0-10, representing the severity]
  - material_part: [material part from the list of Material Parts]
  - location_in_image: [describe location within the image, if applicable]

### List of Spaces
{spaces}

### List of Defect Types
{defect_types}

### List of Material Parts
{material_parts}
"""

prompt_inference_R2 = """You are a professional home inspector AI that assesses homes for structural issues and defects. Examine the provided image(s) of the home and inspect each visible area. For every potential issue you find, guess the space of the image taken, identify the likely defect type, describe the defect, estimate the material(s) involved, give a severity score (0–10).
If the image quality or angle is insufficient to judge or no defect is visible, say Unknown.”

* End your response with 'Final answer: '. In the Final answer section, there should be only a single JSON object, which contains the defect information with the following fields:
- space: [Space name from the list of Spaces]
- defect_present: Yes / Unknown
- If Yes, also include:
  - defect_type: [type from the list of Defect Types]
  - defect_description: [brief description of the defect]
  - defectiveness_score: [0-10, representing the severity]
  - material_part: [material part from the list of Material Parts]
  - location_in_image: [describe location within the image, if applicable]

### List of Defect Types
{defect_types}
"""

prompt_theo_v1 = """## You are performing "Multimodal Interleaved Reasoning" task to analyze an image for physical defects. Output the thinking process within a pair of <T> </T> tags and then output the final answer within a pair of <A> </A> tags. 
During the thinking process, carefully examine all visual cues in the original image, and logically infer the clue of physical defects which lead us to the conclusion of the Existing Label above. Let your reasoning process sequentially guide you from visual clues to the conclusion that aligns with the intended Existing Labels above—without referencing the Existing Label directly or mentioning its content anywhere in your thought process or answer. Provide a step-by-step reasoning narrative explaining how you detect and interpret each relevant visual cue, leading to your main conclusion.
Do not explicitly mention any of the content or presence of the Existing Labels in your response nor inside <T> and </T>.

Your output must start from the thinking process with a pair of <T> </T> tags.
In the Answer section with a pair of <A> </A> tags, there should be a JSON list, where the json object is the defect information for each defect with the following fields:
- space: [Space name from the list of Spaces]
- defect_present: Yes / No
- If Yes, also include:
  - defect_type: [type from the list of Defect Types]
  - defect_description: [brief description of the defect]
  - defectiveness_score: [0-10, representing the severity]
  - material_part: [material part from the list of Material Parts]
  - location_in_image: [describe location within the image, if applicable]
"""

# v2 으로 한번 뽑아봐야함. 먼저 defect label 정리된 이후에 v2 으로 뽑아보기.
prompt_theo_v2_system = """You are a professional home inspector AI."""

prompt_theo_v2 = """## Existing Labels
{existing_labels}

## You are performing "Multimodal Interleaved Reasoning" task to analyze a home image for structural issues and defects. Output the thinking process within a pair of <T> </T> tags and then output the final answer within a pair of <A> </A> tags, with the following rules.

* During the thinking process, carefully examine all visual cues in the original image. Use logical reasoning to identify the areas and materials that might be associated with a potential defect, as well as any physical clues that could lead to a conclusion about the described defect in the Existing Label above without referencing the Existing Label directly or mentioning its content anywhere in your thought process or answer. Try to envision how the item would appear if there were no defects present. Also, consider other possible types of defects beyond the one mentioned and explain why these alternatives are not applicable in this case.
Do not explicitly mention any of the content or presence of the Existing Labels in your response nor inside <T> and </T>.
Your output must start from the thinking process with a pair of <T> </T> tags.

* After the thinking section, output the final answer within a pair of <A> </A> tags.
In the Answer section with a pair of <A> </A> tags, there should be a JSON object, where the json object is the defect information with the following fields:
- space: [Space name from the list of Spaces]
- defect_present: Yes / Unknown
- If Yes, also include:
  - defect_type: [type from the list of Defect Types]
  - defect_description: [brief description of the defect]
  - defectiveness_score: [0-10, representing the severity]
  - material_part: [material part from the list of Material Parts]
  - location_in_image: [describe location within the image, if applicable]

### List of Spaces
{spaces}

### List of Defect Types
{defect_types}

### List of Material Parts
{material_parts}
"""

spaces = """- Living Room
- Bedroom
- Alpha Room
- Kitchen
- Pantry
- Utility Room
- Bathroom
- Laundry Room
- Entrance
- Vestibule
- Entrance Storage
- Hallway
- Balcony
- Outdoor Unit Room
- Emergency Shelter
- Dressing Room
- Boiler Room
- Exterior Wall
"""

defect_types = """- Painting defect
- Opening/closing defect
- Vertical/horizontal alignment defect
- Locking defect
- Peeling/delamination
- Deterioration/aged
- Discoloration
- Sheet defect
- Cover defect
- Caulking defect
- Corrosion/rust
- Backflow
- Handle defect
- Gradient/slope defect
- Adhesion defect
- Inflow/infiltration
- Omission/missing
- Joint/grout separation
- Detachment/replacement
- Damage/breakage
- Lifting/bulging
- Insufficient pressurization
- Tearing/rip
- Oil leakage
- Water leakage
- Scratch
- Level difference/step
- Fixing defect
- Condensation
- Wallpapering defect
- Joint defect
- Heating defect
- Natural ventilation defect
- Water pressure defect
- Operation defect/malfunction
- Crack
- Peeling
- Separation/gap
- Finishing defect
- Installation defect
- Construction defect
- Contamination
- Depression/subsidence
- Noise
- Drainage defect
- Inspection/replacement
- Hole
- Lighting defect
- White cement defect
- Mold
"""

material_parts = """- Outlet
- Wall
- Bathtub
- Range hood
- Hinge
- Accessory
- Ceiling
- Spray gun
- Locking device
- Drain/waste water pipe
- Heating pipe
- Valve
- Ventilation duct
- Detector
- Cement
- Mirror
- Built-in wardrobe
- Upper cabinet
- Unit bath room (UBR)
- Plastic window/door
- Controller
- Boiler
- Sprinkler
- Handle
- Entrance door
- Bathroom cabinet
- Aluminum window/door
- Sink
- Door stopper
- Drain
- Insect screen
- Wallpaper
- Floor covering (sheet)
- Door
- Gas detector
- Hot water distributor
- Tile
- Lighting fixture
- Sensor
- Automatic extinguisher
- Paint
- Kitchen countertop
- Wash basin
- Kitchen furniture
- Slide bar
- Meter
- Shelf
- Molding
- Meter box
- Storage cabinet
- Floor
- Actuator
- Heating equipment
- Glass window
- Baseboard
- Wallpaper & floor covering
- Water-saving pedal
- Gas equipment
- Window frame
- Wall
- Door closer
- Rainwater pipe
- Silicone
- Water supply
- Ballast (lighting ballast/stabilizer)
- Lower cabinet
- Dresser/system furniture
- Wooden window
- Washing machine
- Marble
- Toilet
- Fire extinguisher
- Detergent container
- Door lock
- Faucet
- Hose
- Water saver
- Facilities for the disabled
- Window
- Steel window
- Ventilation system
- Packing (gasket/seal)
- Shoe rack
- Downspout
- Gypsum board
- Glass
- Wood floor
- Latch
- Water supply/hot water pipe
"""
