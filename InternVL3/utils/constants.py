
generation_config = dict(
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
)

user_question = """Given the media, output a detailed analysis of how you analyzed the scene in JSON format. 
Conduct an analysis of what you see and how each component interacts with each other.

Follow this JSON structure exactly:

{
 "planning": {
  "ROI": [
   {"category": "noun_category (e.g., person, car, dog)", "description": "noun phrase with short participial phrase and appearance"},
  ]
 },
 "reasoning": {
  "regional_analysis": [
   {"region": "region name from planning stage, sequentially",
    "observations or analysis": ["detailed analysis point 1", "detailed analysis point 2"]}
  ],
  "overall_scene": {
   "observations or analysis": ["overall scene analysis points"]
  }
 },
 "conclusion": {
 "comprehensive_analysis": "comprehensive and detailed summary assembling all reasoning"
 }
}

IMPORTANT: 
- Include only few major regions of interest (ROI).
- During reasoning, follow natural reasoning flow - if you discover something new or need to reconsider, express it naturally (e.g., "Aha,", "Wait,", "Actually...", "Looking more carefully..."). Only use such expressions when there's a genuine reasoning transition, not artificially.
- Each observation or analysis should combine specific visual details with interpretive reasoning
- For video content, note any significant temporal changes or developments naturally within your observations"""