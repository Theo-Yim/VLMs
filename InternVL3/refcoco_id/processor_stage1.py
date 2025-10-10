"""
Stage 1: LLM-based Identity Dataset Generation
- prompts with few-shot examples
- diverse question templates

Generates:
- 1 mock name per person (diverse, realistic)
- 1 question per person (natural, varied with visual details)
- 1 detailed answer with RICH <think> section (observations → tool → integration)
"""

import json
import os
import pickle
import random
import re
from typing import Dict, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class IdentityStage1Processor:
    """Stage 1: Generate high-quality raw identity Q&A data"""

    def __init__(
        self,
        model_path: str = "openai/gpt-oss-20b",
        output_path: str = "/mnt/nas3/Data/coco",
        output_folder: str = None,
    ):
        self.model_path = model_path
        self.dataset_p_root = output_path

        # Output directory
        if output_folder:
            self.output_folder = output_folder
        else:
            self.output_folder = os.path.join(output_path, "refcoco_identity_stage1")
        os.makedirs(self.output_folder, exist_ok=True)

        # Initialize LLM
        print(f"Loading LLM: {model_path}")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.8,
            max_num_seqs=1,
        )

        self.sampling_params = SamplingParams(
            temperature=0.99,  # Higher temperature for diversity
            top_p=0.98,
            max_tokens=2048,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # System prompt
        self.system_prompt = "You are a helpful assistant that generates diverse, natural training data for multimodal AI models."

    def load_datasets(self, merged_data_path: str) -> List[Dict]:
        """Load merged RefCOCO data and filter for person annotations"""
        print(f"Loading merged data from: {merged_data_path}")
        with open(merged_data_path, "rb") as f:
            data = pickle.load(f)

        # Handle both dict and list formats
        if isinstance(data, dict) and "data" in data:
            data_list = data["data"]
        else:
            data_list = data

        # Filter: only images with person annotations
        person_data = []
        for entry in data_list:
            person_annos = [ann for ann in entry["annotations"] if ann.get("category") == "person"]
            if len(person_annos) > 0:
                entry["annotations"] = person_annos
                person_data.append(entry)

        print(f"Found {len(person_data)} images with person annotations")
        return person_data

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with system prompt and extract final answer"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Format as chat template using cached tokenizer
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.llm.generate(formatted, self.sampling_params)
        response = outputs[0].outputs[0].text.strip()

        # GPT-OSS-20B shows reasoning - extract only final answer after "assistantfinal"
        if "assistantfinal" in response:
            parts = response.split("assistantfinal")
            response = parts[-1].strip()

        # Also try to remove analysis/reasoning prefixes
        if response.startswith("analysis"):
            lines = response.split("\n")
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith("analysis"):
                    response = "\n".join(lines[i:]).strip()
                    break

        return response

    def generate_mock_name(self) -> str:
        """Generate ONE diverse, realistic name using LLM with few-shot"""
        prompt = """Generate exactly ONE realistic full name (first and last name) from any culture.

Examples:
James Chen
Maria Rodriguez
Aisha Patel
David Kim
Elena Volkov
Marcus Williams
Yuki Tanaka
Omar Hassan
Sofia Martinez
Priya Singh
Abdelrahman Ahmed
Oliver Evans
Mina Lee
Aleksei Ivanov

Output ONLY the name (2 words), nothing else:"""

        response = self._call_llm(prompt)

        # Clean the response
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        for line in lines:
            # Remove leading numbers/bullets
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            # Check if it looks like a name (2 words, capitalized)
            words = line.split()
            if len(words) == 2 and all(w[0].isupper() for w in words if w):
                return line

        # Fallback to random default name
        defaults = [
            "James Chen",
            "Maria Rodriguez",
            "Aisha Patel",
            "David Kim",
            "Sarah Johnson",
            "Elena Volkov",
            "Marcus Williams",
            "Yuki Tanaka",
            "Omar Hassan",
            "Sofia Martinez",
            "Priya Singh",
            "Carlos Mendez",
            "Oliver Gouldbourne",
            "Junghyun Park",
            "Fatima Noor",
            "Scarlett Johansson",
            "Katie Fischer",
            "Lucas Bader",
            "Riley Redmond",
            "Mohamed Salah",
            "Megan Shirley",
            "Ondrej Kolar",
            "Hunter Richardson",
            "Hector Garcia",
            "Naveen Neppalli",
            "Erfan Darvish",
            "Lina Zhang",
            "Russel Pomeroy",
            "Jayakarthick S",
            "Ken Murakami",
            "Abdelrahman El-Sayed",
            "Aleksander Petrov",
        ]
        return random.choice(defaults)

    def generate_question_for_person(
        self, person_desc: str, position_hint: str, all_descriptions: str
    ) -> str:
        """
        Generate ONE natural, diverse question about a person's identity

        Uses few-shot examples for better quality and variety
        """
        prompt = f"""You are generating identity questions for a training dataset.

Person description: {person_desc}
Position: {position_hint}
Context: Image with multiple people

Generate ONE natural question asking about this person's identity. Use diverse question styles from these examples:

Style 1 - Direct: "Who is the person wearing the blue jacket in the image?"
Style 2 - Polite: "Can you identify the individual standing on the right side?"
Style 3 - Specific: "What is the name of the person holding the tennis racket?"
Style 4 - Simple: "Who is the woman in the grey shirt?"
Style 5 - Location: "Who is the person standing in front of the building entrance?"
Style 6 - Multiple attributes: "Who is the older woman with glasses wearing a blue jacket on the left side?"
Style 7 - Action-based: "Who is the person shopping at the open-air fruit market?"
Style 8 - Casual: "Can you tell me who the person in the red shirt is?"

Requirements:
- Be natural and conversational
- Include specific visual details from the description
- Vary the question structure (don't always start with "Who is")
- 10-20 words
- End with question mark

Output ONLY the question:"""

        response = self._call_llm(prompt)

        # Clean response
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        for line in lines:
            # Remove leading numbers/bullets/quotes
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            line = re.sub(r'^["\']|["\']$', "", line)

            if line.endswith("?") and len(line.split()) >= 8:
                return line

        # Fallback
        return f"Who is the {person_desc} in the image?"

    def generate_answer_with_rich_reasoning(
        self,
        question: str,
        person_name: str,
        person_num: int,
        person_desc: str,
        all_people_context: str,
    ) -> str:
        """
        Generate answer with RICH reasoning pattern:
        1. Visual observation (2-3 sentences about what's in the image)
        2. Spatial reasoning (where the person is, which one matches the question)
        3. Tool call: {{Identify person #X}}
        4. Integration (connect tool response back to question context)
        """
        prompt = f"""You are generating training examples for an AI that uses an "Identify" tool to recognize people.

Context: Image with multiple people
Question: {question}
Person's actual name: {person_name}
Person description: {person_desc}
All people in image:
{all_people_context}

Generate a natural response showing how the AI would:
1. OBSERVE: First describe what you see in the image (be specific about visual details)
2. LOCATE: Identify which person matches the question's description (position, clothing, action)
3. TOOL USE: Call the tool: {{{{Identify person #{person_num}}}}}
4. INTEGRATE: Connect the tool's response back to the question's context

Example 1 (good visual observation + spatial reasoning):
"Looking at the image, I can see several people. The question asks about the person wearing a blue jacket. I can see someone on the left side wearing what appears to be a blue jacket. Let me identify this person.
{{{{Identify person #1}}}}
Based on the identification, the person wearing the blue jacket is James Chen."

Example 2 (action-based observation):
"The question asks me to identify the person standing on the right side. Looking at the scene, I can see a person positioned on the right side of the image. Let me use the Identify tool to determine who this is.
{{{{Identify person #1}}}}
The identification confirms this is Maria Rodriguez."

Example 3 (detailed observation with object context):
"I need to find who is holding the tennis racket in this image. Looking at the scene, I can see a person holding a tennis racket. They appear to be on a tennis court. Let me identify this individual.
{{{{Identify person #1}}}}
The tool confirms that the person holding the tennis racket is Aisha Patel."

Example 4 (multiple people - shows richer reasoning):
"The question asks about two people sitting on the bench. I can see a bench with two individuals seated on it. Let me identify both of them using the Identify tool.

First, let me identify the person on the left side of the bench.
{{{{Identify person #1}}}}
The first person is David Kim.

Now let me identify the person on the right side of the bench.
{{{{Identify person #2}}}}
The second person is Sarah Johnson.

So the two people sitting on the bench are David Kim and Sarah Johnson."

Now generate a response for the given question. Requirements:
- Start with visual observation (what you see)
- Include spatial reasoning (where/which person)
- Use tool call: {{{{Identify person #{person_num}}}}}
- Connect result back to question context
- Be natural and conversational (2-4 sentences before tool, 1-2 after)

Output the response directly:"""

        response = self._call_llm(prompt)

        # Ensure the placeholder is present
        if f"{{{{Identify person #{person_num}}}}}" not in response:
            # Insert placeholder if missing
            lines = response.split(". ")
            if len(lines) >= 2:
                # Insert after first 1-2 sentences
                insert_pos = min(2, len(lines) - 1)
                response = (
                    ". ".join(lines[:insert_pos])
                    + f". {{{{Identify person #{person_num}}}}} "
                    + ". ".join(lines[insert_pos:])
                )
            else:
                response += f"\n{{{{Identify person #{person_num}}}}}\nThe person is {person_name}."

        return response.strip()

    def assign_names_to_people(self, data_entry: Dict) -> List[str]:
        """Generate ONE unique mock name per person"""
        num_people = len(data_entry["annotations"])
        names = []

        # Generate unique names
        for i in range(num_people):
            attempts = 0
            while attempts < 5:  # Try up to 5 times for unique name
                name = self.generate_mock_name()
                if name not in names:
                    names.append(name)
                    break
                attempts += 1
            else:
                # Fallback: append number if can't get unique name
                base_name = self.generate_mock_name()
                names.append(f"{base_name}")

        data_entry["person_names"] = names
        return names

    def generate_questions_and_answers(self, data_entry: Dict) -> int:
        """
        Generate ONE question and answer per person
        Returns: number of LLM calls made
        """
        annotations = data_entry["annotations"]
        person_names = data_entry["person_names"]

        # Create context string with all people
        all_people_desc = []
        for idx, ann in enumerate(annotations, 1):
            descriptions = ann.get("descriptions", [])
            if descriptions:
                all_people_desc.append(f"Person {idx}: {descriptions[0]}")
        all_people_context = "\n".join(all_people_desc)

        qna_list = []
        num_calls = 0

        for idx, (ann, name) in enumerate(zip(annotations, person_names), 1):
            # Get person description
            descriptions = ann.get("descriptions", ann.get("merged_referring_expressions", []))
            if descriptions:
                person_desc = descriptions[0]  # Use first description
            else:
                person_desc = f"person {idx}"

            # Determine position hint from bbox
            bbox = ann["bbox"]
            x_center = (bbox[0] + bbox[2]) / 2
            image_width = 640  # Approximate COCO image width
            if x_center < image_width * 0.33:
                position = "on the left side"
            elif x_center > image_width * 0.67:
                position = "on the right side"
            else:
                position = "in the center"

            # Generate question
            question = self.generate_question_for_person(person_desc, position, all_people_context)
            num_calls += 1

            # Generate answer with RICH reasoning
            answer_raw = self.generate_answer_with_rich_reasoning(
                question, name, idx, person_desc, all_people_context
            )
            num_calls += 1

            qna_list.append(
                {"Q": question, "A_raw": answer_raw, "person_num": idx, "person_name": name}
            )

        data_entry["QnA"] = qna_list
        return num_calls

    def save_results(self, data_entry: Dict, output_file: str):
        """Save results to JSON file"""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_entry, f, indent=2, ensure_ascii=False)
