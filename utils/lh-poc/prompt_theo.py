R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

prompt_inference = """## You are a professional house construction inspector. Your job is to examine the provided image and determine if there is any defect. You need to guess the space, defect type, defect description, material part, and location in the image of the image.

You must output the answer in the json format with the following fields:
- space: [Space name from the list of Spaces]
- defect_present: Yes / Unknown
- If Yes, also include:
  - defect_type: [type from the list of Defect Types]
  - defect_description: [brief description of the defect]
  - defectiveness_score: [0-10, representing the severity]
  - material_part: [material part from the list of Material Parts]
  - location_in_image: [describe location within the image, if applicable]

### Instructions
- Carefully examine each part of the image.
- Identify the space of the image.
- Identify the material part of the image.
- Identify the defect type of the image.
- Identify the defect description of the image.
- Identify the location in the image of the image.
"""

prompt_theo = """## Existing Labels
{existing_labels}
## You are performing "Multimodal Interleaved Reasoning" task to analyze an image for physical defects. Output the thinking process within a pair of <T> </T> tags and then output the final answer within a pair of <A> </A> tags. 
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

prompt = """
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
defect_present: Yes / No
If Yes, also include:
defect_type: [type from the list above]
defectiveness_score: [0-10, representing the severity]
material_part: [material part from the defect part]
location_in_image: [describe location within the image]
"""
