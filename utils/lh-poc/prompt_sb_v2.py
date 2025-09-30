ENGLISH_INFERENCE_PROMPT = """You are a professional home inspector AI that assesses homes for structural issues and defects. 
Examine the provided image(s) of the home and inspect each visible area. 
For every potential issue you find, guess the space of the image taken, identify the likely defect type, describe the defect, estimate the material(s) involved, give a severity score (0–10).
If the image quality or angle is insufficient to judge or no defect is visible, say "Unknown".

defect message: {defect_message}

* End your response with 'Final answer: '. In the Final answer section, there should be only a single JSON object, which contains the defect information with the following fields:
- space: [Select from list of spaces]
- defect_present: Yes / Unknown
- If Yes, also include:
  - material_part: [Select from list of material parts]
  - defect_type: [Select from list of defect types]
  - defect_description: [brief description of the defect]
  - defectiveness_score: [0-10, representing the severity]
  - construction_type: [Select from list of construction types]
  - detail_construction_type: [Select from list of detail construction types]
  - location_in_image: [describe location within the image, if applicable]

### List of spaces
{spaces}

### List of material parts
{material_parts}

### List of defect types
{defect_types}

### List of construction types
{construction_types}

### List of detail construction types
{detail_construction_types}
"""

# ### Existing Label will be inserted in the prompt
ENGLISH_TRAIN_PROMPT =  """
## Ground truth labels
{ground_truth_labels}

## Additional constraints

Do not explicitly mention any of the content or presence of the Ground truth labels in your response nor in the thinking process.

* End your response with 'Final answer: '. In the Final answer section, there should be only a single JSON object, which contains the defect information with the following fields:
- space: [Select from list of spaces]
- defect_present: Yes / Unknown
- If Yes, also include:
  - defect_type: [Select from list of defect types]
  - defect_description: [brief description of the defect]
  - defectiveness_score: [0-10, representing the severity]
  - material_part: [Select from list of material parts]
  - construction_type: [Select from list of construction types]
  - detail_construction_type: [Select from list of detail construction types]
  - location_in_image: [describe location within the image, if applicable]

### List of Spaces
{spaces}

### List of Defect Types
{defect_types}

### List of Material Parts
{material_parts}

### List of Construction Types
{construction_types}

### List of Detail Construction Types
{detail_construction_types}
"""

KOREAN_INFERENCE_PROMPT = """
"""

KOREAN_TRAIN_PROMPT = """
"""

R1_SYSTEM_PROMPT = """
You are a professional home inspector AI specializing in defect message interpretation. 
Given a user-supplied defect message describing a suspected issue in a home, process the message as follows:

1. Begin with a step-by-step analysis enclosed within <think> and </think> tags.

2. Carefully interpret what the defect message suggests:
-Infer the likely space (from the provided list).
-Guess the relevant material part(s).
-Identify the probable defect type(s).
-Estimate severity and describe the nature of the defect.
-Determine possible construction and detail construction types.

3. Consider ambiguities in the defect message and reason about alternative interpretations if details are vague.

4. If the defect message is unclear or lacks enough detail, note which assumptions are made.
""".strip()

MAPPING_SYSTEM_PROMPT = """
For given ground truth mapping and input, you need to map the input to the ground truth mapping. You can return "Invalid" if the input is not related to the ground truth mapping.

ground truth mapping:
{ground_truth_mapping}

input:
{input}

output format:
[input] -> [ground truth]
"""

TRANSLATION_SYSTEM_PROMPT = """
You are a professional translator specializing in construction and housing terminology.
Given a Korean term related to construction, housing, or building defects, provide:
1. English translation
2. Brief one-line description in English

Please respond in the following JSON format:
{
    "english": "English translation",
    "description": "Brief one-line description"
}

Korean term: {korean_term}
Category: {category}

Consider the context of construction, housing, and building inspection when translating.
"""

CATEGORY_DESCRIPTIONS = {
    "공간": "Spaces/Rooms in a building",
    "부위자재": "Building materials and components", 
    "하자유형": "Types of defects/issues",
    "보수공사명": "Construction types",
    "공종": "Construction types"
}

space_list = """MDF Room\nPublic Development\nUnit Bath Room\nControl Room\nLiving Room\nBulletin\nSenior Citizens' Center\nGuardroom\nStaircase\nHigh water tank\nShared Space\nCommon entrance\nCommon Facilities\nPublic Bathroom\nCommon Entrance\nPublic Restroom\nManagement Office\nManagement Facilities\nMechanical Room\nGreen Space\nPlayground facilities\nMultipurpose Room\nComplex Internal Road\nDoorpost of a complex\nShelter\nLibrary\nBronze Statue Hall\nDressing room\nBalcony\nLiving Room Balcony\nKitchen Balcony\nShelter Room\nValve Room\nTransformer Room\nBoiler Room\nPedestrian Space/Plaza\nCorridor\nMaster Bathroom\nRecycling Center\nLaundry Room\nCommercial Space\nElevator\nElevator Machine Room\nElevator lobby\nOutdoor Unit Room\nAlfa Room\nKindergarten\nHeat Exchanger Room\nSewage Treatment Facility\nIndoor\nRooftop\nOutdoor/Exterior\nExterior wall\nExterior\nBathroom\nStormwater Management Facility\nExercise Facility\nRainwater Inspection Room\nBicycle Storage Room\nHandicap Ramp\nElectrical Room\nWhole\nResidential Community Facilities\nKitchen\nParking Facility\nRoof\nAbove-ground parking lot\nBasement\nUnderground water tank\nUnderground Parking Lot\nEntrance\nBedroom (Master Bedroom)\nBedroom\nBedrooms\nCommunication Room\nPartial evacuation\nEviction (total)\nPantry\nPump Room\nPiloti\nEntrance / Foyer\nEntrance storage\nEnvironmental (Sculptural) Facility\nHumidity Room\nRest facility"""

material_parts = """Aluminum window / Aluminum frame window\nCCTV\nCRT (Cathode Ray Tube)\nDVR (Digital Video Recorder)\nDVR / NVR\nPlastic (PVC) window / PVC-framed window\nTelevision / TV set\nCommunal TV reception system\nTV antenna / Television antenna\nUPS (Uninterruptible Power Supply)\nStreetlight / Street lamp\nGas meter reading / Gas meter inspection\nGas detector / Gas alarm\nGas equipment / Gas installation\nRemote gas metering / Remote gas reading\nGas cooktop / Gas stove top\nTemporary canopy / Shade structure\nSignboard / Signage\nMonitoring panel / Surveillance control panel\nDetector / Sensor\nSteel window / Steel-framed window\nStrainer / Filter screen\nMirror\nBaseboard / Skirting board\nLatch / Bolt\nSecurity office equipment / Guardhouse equipment\nMeter (utility meter)\nMeter box / Meter enclosure\nRelay (protective relay)\nElevated water tank / Overhead tank\nAir blower / Inflator\nManagement office equipment / Facility management equipment\nCold & hot water supply piping\nWater supply / Hot water pump\nHydrant / Water hydrant\nWater supply pump\nHot water (domestic) pump\nOther mechanical equipment / Misc mechanical parts\nOther parts / Miscellaneous parts\nRailing / Handrail / Guardrail\nHeating piping / Heating circulation pipes\nHeating system / Heating equipment\nHeating circulation pump\nGreen area (lawn surface)\nNon-slip stair nosing / Anti-slip (stair treads)\nPlayground equipment / Playground apparatus\nPlay wall / Partition wall\nWall / Fence wall\nMarble (stone)\nWallpaper / Wall covering\nDoor lock (keyless/electronic door lock)\nDoor lock (electronic door lock)\nDoor stopper / Door stop\nDoor closer (hydraulic door closer)\nIntrusion/body detector / Motion detector\nRadiator\nRange hood / Kitchen hood\nFinish material / Finishing material\nWood flooring / Hardwood floor\nPile / Stake\nMat / Floor mat\nManhole / Manhole cover\nSand / Wood chips\nTimber / Wood material\nWooden deck / Timber decking\nWooden window / Wood-framed window\nMolding / Trim\nUnmanned parcel locker / Package delivery locker\nDoor\nWater play facility / Splash pool facility\nWater tank / Reservoir tank\nAnti-slip treatment / Non-slip treatment\nFloor / Flooring\nFlooring material / Floor covering\nPavement / Floor paving\nTransmitter (alarm transmitter)\nHeating cable / Heat trace cable\nGenerator\nPublic address (PA) system / Broadcasting equipment\nBroadcast amplifier / PA amplifier\nHeating panel / Heat sink\nSpeed bump / Curb / Wheel stop\nFire door\nPipe / Piping\nExhaust fan\nWiring / Electrical wiring\nDrain / Floor drain\nDrainage facility / Drainage system\nDrainage pump / Sewage pump\nBattery\nValve\nSlope protection / Stone revetment\nBench (outdoor)\nWall\nWall surface / Wall\nTransformer\nStorage room / Storage facility\nPaving block / Concrete paver\nBoiler\nDistribution panel / Electrical distribution board\nBuilt-in closet / Built-in wardrobe\nVideo phone / Video intercom\nEmergency call station / Emergency call button\nCountertop / Worktop\nShower booth / Shower cubicle\nStone retaining wall / Stone embankment\nShelf / Shelving\nWashbasin / Bathroom sink\nWashing machine\nSensor\nConsumables / Spare consumable parts\nFire-fighting equipment / Fire protection system\nFire pump\nFire extinguisher\nFire suppression piping / Fire-fighting piping\nFire-fighting supplies / Fire-fighting accessories\nFire hydrant / Hose reel\nFire-fighting pump\nHandle / Door handle / Knob\nWater hammer arrester / Surge arrester (plumbing)\nWater garden / Aquatic planting facility\nCabinet / Storage cabinet\nWater supply / Tap water system\nTrees / Woody plants\nLow-voltage distribution board / Switchboard\nReceiver (signal receiver)\nFaucet / Tap\nWater fixtures / Sanitary fixtures\nSubmersible pump / Underwater pump\nInstantaneous water heater / Tankless water heater\nShingle (roof shingle)\nSwitch (electrical switch)\nScrew / Fastener\nSprinkler (fire sprinkler)\nSpeaker\nCement\nShoe cabinet / Shoe rack\nSilicone sealant / Silicone\nAutomated External Defibrillator (AED)\nAED battery / Defibrillator battery\nAED pads / Defibrillator pads\nKitchen sink\nSignage / Information facilities\nSeating wall / Low sitting wall\nPressure tank\nAccessories / Fittings\nToilet (water closet)\nToilet fittings / Toilet accessories\nAir conditioner (A/C)\nTerminal connector / Connection terminal\nFuel tank\nHeat exchanger\nKey ring / Key fob\nStandby pump / Backup pump\nWater supply & drainage pipes\nRetaining wall\nExterior wall / Outer wall\nBathroom exhaust fan\nBathroom cabinet / Vanity cabinet\nBathtub\nExercise equipment / Fitness equipment\nFence\nFencing / Types of fences\nRemote metering / Remote metering system\nWall pad / Home control wall pad\nEmergency/exit light / Guiding light\nGlass / Glazing\nGlazed window / Window glazing\nFood waste dehydrator / Food waste dewatering machine\nIntercom / Video intercom\nEntrance / Entryway\nAutomatic control system / Automation control\nBicycle parking / Bike rack shelter\nGrass / Lawn\nGrass protection mat / Turf protection mat\nSheet vinyl flooring / Vinyl flooring\nElectrical room equipment / Electrical room devices\nTelephone\nTransfer switch / Changeover switch\nWater-saving pedal (pedal-operated faucet)\nInspection opening / Access panel\nRectifier\nGazebo / Pavilion\nControl panel / Control cabinet\nSmoke-control damper / Smoke damper\nOrnamental trees / Landscaping trees\nLighting fixture / Light fixture\nController / Regulator\nKitchen TV\nKitchen cabinets / Kitchen furniture\nRepeater / Signal repeater\nAmplifier (signal amplifier)\nRoof\nGauge / Indicator\nUnderground reservoir / Underground storage tank\nCircuit breaker\nRoadway / Driveway\nVehicle access control system\nDifferential pressure / Flow control valve\nWindow frame / Sash\nCeiling\nSports facilities / Exercise equipment\nDoorbell\nFlowering plants / Herbaceous plants\nBattery bank / Storage battery\nPressure boosting pump / Pressure pump\nCover / Protective cover\nComputer / PC\nCapacitor / Condenser\nElectrical outlet / Power outlet\nConcrete\nTimer\nTile\nElastic (resilient) surfacing / Rubberized surfacing\nSolar power system / Photovoltaic system\nTable\nOther communications equipment / Comms miscellaneous\nCommunication junction box / Telecom junction box\nCommunication outlet / Data/phone outlet\nTraffic barrier / Access control barrier\nTrench / Trench drain\nPergola\nPacking / Gasket / Seal\nExpansion tank\nPump\nPaint / Coating\nFloor hinge / Floor-mounted door hinge\nLightning protection system / Lightning arrestor\nBanner frame / Banner stand\nHome network server / HomeNet server\nFlower bed / Planting bed\nDisplay / Screen\nFire detector / Smoke detector\nFire alarm / Fire buzzer\nFire alarm control panel\nVentilation duct / Ventilation piping\nVentilation system / Ventilation equipment\nVentilator / Exhaust fan\nExtractor / Exhaust extractor\nHeater\nActuator\nWallpaper and Floor Covering\nInsect Screen\nShower Slide Bar\nHot Water Distributor\nRainwater Pipe\nLocking Device\nAccessibility Facility\nKitchen Countertop\nWindow\nLower Cabinet\nEntrance Door\nHose\nGutter / Downspout"""

defect_type_list = """Poor Caulking : The sealant (caulking/silicone) applied around window frames, bathtubs, or sinks to block moisture and air is peeling, cracked, or missing.\nWater Leak : Water seeps into the building's interior (walls, ceiling, floor) due to a damaged waterproof layer, broken pipes, etc.\nDelamination / Flaking : A paint or coating layer flakes or peels off like a film. It often implies a more severe condition than 'Lifting' (들뜸).\nCorrosion / Rust : Metal materials deteriorate and rust due to chemical reactions with oxygen and moisture in the air.\nScratch / Nick / Dent : A general term for marks or damage on a surface, including nicks, dents, and scratches.\nDamage / Breakage : A material is broken or shattered due to external impact, losing its original function.\nPoor Finishing : The final touches, such as wallpapering, painting, or caulking, are messy or incomplete.\nLifting / Peeling : Wallpaper, vinyl sheets, tiles, or flooring material is detaching from the surface due to poor adhesion.\nCrack : A line-shaped break that appears on the surface of concrete or finishing materials like walls, floors, or ceilings. It can be caused by structural issues or material shrinkage/expansion.\nStain / Contamination : A surface has a mark, spot, or dirt that cannot be easily removed.\nHole / Puncture : An unintentional hole in a wall, floor, or door.\nGrout Loss : The grout material between tiles is cracking, crumbling, or falling out.\nDiscoloration : The original color of a material has changed due to factors like sunlight, chemicals, or moisture.\nGap / Separation : An unintended space has opened up between two components that should be joined, such as a door frame and a wall.\nPoor paintwork : A defect in the quality or application of paint on a building's surface\nPeeling : A defect where a layer or coating comes off, typically referring to paint or plaster.\nPoor Joint / Seam : The connection point between two materials (e.g., wallpaper sheets, floorboards) is not smooth or has a gap.\nSinking / Indentation : A part of a surface, such as a floor or wall, is dented or has sunk inward.\nCondensation : Water droplets form on surfaces like walls and windows due to temperature differences, as moisture from the air condenses. It is a primary cause of mold.\nPoor wallpapering : A defect or issue related to improper or poor quality wallpaper installation\nTear : A type of defect characterized by a long, narrow opening or breakage in a material or structure"""

work_type_list = """Architecture\nMachinery\nElectricity\nLandscaping\nCivil Engineering\nCommunication"""

repair_work_list = """UBR Construction\nIndividual Boiler Installation\nMinor Repair\nMetering device construction\nReplacement Work\nWater Supply/Hot Water Piping Work\nMechanical Miscellaneous Works\nMiscellaneous\nHeating Pipe Installation\nLeakage repair work\nWallpapering work\nUrban Gas Pipeline Construction\nPainting Work\nPower Facility Construction\nFinishing work\nCarpentry\nFlooring Work\nWaterproofing work\nPlumbing/Electrical work\nRepair work\nPart Replacement\nFire Protection Facilities Construction\nFirefighting Piping Work\nPlumbing work\nAir Conditioning Miscellaneous Construction\nWaste/Drainage Pipe Work\nDrainage Pipe Work\nSanitary ware installation\nGlasswork\nAutomatic Control Work\nMiscellaneous construction\nMiscellaneous Iron Works\nVinyl Flooring Work\nFull Frontal Replacement\nInformation Control, Security Facilities Construction\nInformation and Communication Facilities Construction\nLandscape Facilities Construction\nLighting Installation Work\nCentral Boiler Construction\nWindow Installation Work\nReinforced Concrete\nTile Work\nPump Installation Work\nVentilation work\nVentilation Pipe Work"""
