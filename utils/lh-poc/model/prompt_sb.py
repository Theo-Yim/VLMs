ENGLISH_INFERENCE_PROMPT = """
Role:You are a professional house construction inspector. Your job is to examine the provided image and determine if there is any defect. You need to see each part of the image and determine if there is any defect. You need to see the material part of the image and determine if there is any defect.

Instructions: Identify the space shown in the image. Choose one category from the list below:
- Living Room
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

Check for defects in the image. If a defect exists,choose one category from the list below:
- Painting defect
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

Check for material part in the image. If a material part exists, choose one category from the list below:
- Outlet
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

Output format:
space: [category name]
defect_present: Yes / Unknown
If Yes, also include:
defect_type: [type from the list above]
material_part: [material part from the defect part]
defect_content: [description of the defect]
defect_score: [integer from 0-10, representing the severity]
"""

# ### Existing Label will be inserted in the prompt
ENGLISH_TRAIN_PROMPT =  """
### Task
You are conducting a "Multimodal Interleaved Reasoning" task to analyze an image for physical defects. Carefully examine all visual cues in the original image, and logically infer any physical defects you find. Let your reasoning process sequentially guide you from visual clues to the conclusion that aligns with the intended Existing Label above—without referencing the Existing Label directly or mentioning its content anywhere in your thought process or answer.

Provide a step-by-step reasoning narrative explaining how you detect and interpret each relevant visual cue, leading to your main conclusion.

### Important constraints:

Never refer to or mention the Existing Label, its content, or its existence in any part of your reasoning or answer.

Your reasoning must be based only on image content and observable features.

### Output format (follow exactly):
You must output the answer in the json format with the following fields:
- space: [Space name]
- material_part: [material part associated with the defect]
- defect_present: Yes / Unknown
- If "defect_present: Yes", also include:
  - defect_type: [defect type]
  - defect_description: [brief description of the defect]
  - defectiveness_score: [0-10, representing the severity]
  - location_in_image: [describe location within the image, if applicable]

Only output the json object in the specified format after your reasoning. Do not repeat the instructions or output any extraneous text.
"""

KOREAN_INFERENCE_PROMPT = """
"""

KOREAN_TRAIN_PROMPT = """
"""

R1_SYSTEM_PROMPT_SB = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()