"""
Stage 1 Multi: LLM-based Multi-Person Identity Dataset Generation

Generates multi-person Q&A pairs where:
- 1 question requires identifying 2+ people
- Multiple tool calls in sequence
- Rich reasoning showing systematic identification

Question Categories:
1. Complete Group (all people)
2. Selective Multi-Person (subset based on criteria)
3. Comparative/Relational (two people for comparison)
4. Sequential/Ordered (specific order)
5. Relationship/Interaction (people interacting)
6. Exclusion/Differentiation (who meets/doesn't meet criteria)
7. Context-Specific (sports, shopping, etc.)
8. Complex Reasoning (multi-step)
"""

import json
import os
import pickle
import random
import re
from typing import Dict, List, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class IdentityStage1MultiProcessor:
    """Stage 1 Multi: Generate high-quality multi-person identity Q&A data"""

    def __init__(
        self,
        model_path: str = "openai/gpt-oss-20b",
        output_path: str = "/mnt/nas3/Data/coco",
        output_folder: str = None,
        existing_stage1_folder: str = None,
    ):
        self.model_path = model_path
        self.dataset_p_root = output_path
        self.existing_stage1_folder = existing_stage1_folder

        # Output directory
        if output_folder:
            self.output_folder = output_folder
        else:
            self.output_folder = os.path.join(output_path, "refcoco_identity_stage1_multi")
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
            temperature=0.99,
            top_p=0.98,
            max_tokens=3072,  # Longer for multi-person answers
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # System prompt
        self.system_prompt = "You are a helpful assistant that generates diverse, natural training data for multimodal AI models with multiple tool calls."

    def load_datasets(self, merged_data_path: str) -> List[Dict]:
        """Load merged RefCOCO data and filter for images with 2+ people"""
        print(f"Loading merged data from: {merged_data_path}")
        with open(merged_data_path, "rb") as f:
            data = pickle.load(f)

        # Handle both dict and list formats
        if isinstance(data, dict) and "data" in data:
            data_list = data["data"]
        else:
            data_list = data

        # Filter: only images with 2+ person annotations
        multi_person_data = []
        for entry in data_list:
            person_annos = [ann for ann in entry["annotations"] if ann.get("category") == "person"]
            if len(person_annos) >= 2:  # At least 2 people
                entry["annotations"] = person_annos
                multi_person_data.append(entry)

        print(f"Found {len(multi_person_data)} images with 2+ people")
        return multi_person_data

    def load_existing_names(self, image_id: str) -> List[str]:
        """Load person names from existing Stage 1 output"""
        if not self.existing_stage1_folder:
            return None

        stage1_file = os.path.join(self.existing_stage1_folder, f"{image_id}.json")
        if not os.path.exists(stage1_file):
            return None

        with open(stage1_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("person_names", [])

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with system prompt and extract final answer"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.llm.generate(formatted, self.sampling_params)
        response = outputs[0].outputs[0].text.strip()

        # GPT-OSS-20B shows reasoning - extract only final answer
        if "assistantfinal" in response:
            parts = response.split("assistantfinal")
            response = parts[-1].strip()

        if response.startswith("analysis"):
            lines = response.split("\n")
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith("analysis"):
                    response = "\n".join(lines[i:]).strip()
                    break

        return response

    def generate_multi_person_question(
        self, num_people: int, all_descriptions: List[str], question_type: str = None
    ) -> str:
        """
        Generate ONE multi-person question based on category

        Question types:
        - "group": Identify all people
        - "selective": Identify subset (position, action, appearance)
        - "comparative": Compare two people
        - "sequential": Identify in order
        - "interaction": People interacting
        - "complex": Multi-step reasoning
        """
        # Auto-select type if not specified
        if question_type is None:
            if num_people >= 3:
                question_type = random.choice(["group", "selective", "sequential", "complex"])
            else:  # num_people == 2
                question_type = random.choice(["selective", "comparative", "interaction", "complex"])

        # Create context
        people_context = "\n".join([f"Person {i+1}: {desc}" for i, desc in enumerate(all_descriptions)])

        prompt = f"""You are generating multi-person identity questions for a training dataset.

Context: Image with {num_people} people
People in image:
{people_context}

Question type: {question_type}

Generate ONE natural question that requires identifying {"ALL" if question_type in ["group", "sequential"] else "2 or more"} people. Use diverse question styles:

**Group Identification** (all people):
- "Who are all the people visible in this image?"
- "Can you identify everyone present in the scene?"
- "Please list the names of all individuals shown here."

**Selective Multi-Person** (subset based on criteria):
- "Who are the two people standing on the left side?"
- "Can you identify the people wearing blue jackets?"
- "Who are the individuals sitting on the bench?"

**Comparative** (two people):
- "Between the two people standing, who is taller?"
- "Who is wearing darker clothing, the person on the left or right?"
- "Which of the two appears to be older?"

**Sequential** (ordered identification):
- "Who are the people from left to right in this image?"
- "Starting from the left, who is each person?"

**Interaction** (people interacting):
- "Who are the two people talking to each other?"
- "Who are the people standing next to each other?"

**Complex Reasoning**:
- "How many people are there and who are they?"
- "What are these people doing and who are they?"

Requirements:
- Must require identifying {num_people if question_type in ["group", "sequential"] else "2+"} people
- Be natural and conversational
- Include specific visual details when relevant
- 10-25 words
- End with question mark

Output ONLY the question:"""

        response = self._call_llm(prompt)

        # Clean response
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        for line in lines:
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            line = re.sub(r'^["\']|["\']$', "", line)

            if line.endswith("?") and len(line.split()) >= 8:
                return line

        # Fallback based on type
        if question_type == "group":
            return f"Who are all the people visible in this image?"
        elif question_type == "comparative" and num_people >= 2:
            return f"Between the two people standing, who is taller?"
        else:
            return f"Can you identify the people in this image?"

    def generate_multi_person_answer(
        self,
        question: str,
        person_names: List[str],
        person_descs: List[str],
        num_people_to_identify: int,
    ) -> str:
        """
        Generate answer with RICH multi-person reasoning:
        1. Overall observation (what's in the image)
        2. Strategy statement (how to approach the question)
        3. Sequential identification (one person at a time)
           - Visual description
           - Tool call: {{Identify person #X}}
           - Acknowledgment
        4. Summary integration (connect all results to question)
        """
        # Build context
        all_people_context = "\n".join(
            [f"Person {i+1} ({name}): {desc}" for i, (name, desc) in enumerate(zip(person_names, person_descs))]
        )

        # Determine which people to identify (could be subset)
        if num_people_to_identify > len(person_names):
            num_people_to_identify = len(person_names)

        identified_people = list(range(1, num_people_to_identify + 1))

        prompt = f"""You are generating training examples for an AI that uses an "Identify" tool to recognize multiple people.

Context: Image with {len(person_names)} people
Question: {question}
Number of people to identify: {num_people_to_identify}

People in image:
{all_people_context}

Generate a natural response showing how the AI would:
1. OBSERVE: First describe what you see overall (the scene, how many people)
2. STRATEGY: State how you'll approach the question (e.g., "Let me identify each person systematically" or "I need to identify both individuals")
3. SEQUENTIAL IDENTIFICATION: For each person:
   - Describe their visual appearance and location
   - Call tool: {{{{Identify person #X}}}}
   - Acknowledge the result (e.g., "The first person is [name]")
4. SUMMARY: Connect all results back to the question

Example 1 (3 people, group identification):
"Looking at the image, I can see three people standing together. To answer this question comprehensively, I need to identify each person systematically.

Starting from the left, there's an older woman facing away from the camera, wearing a blue jacket with a barrette in her hair.
{{{{Identify person #1}}}}
The first person is Liam O'Connor.

In the center, I can see a woman wearing a grey shirt with a watch on her wrist, standing upright.
{{{{Identify person #2}}}}
The second person is Leila Hassan.

On the right, there's another woman with glasses who appears to be shopping.
{{{{Identify person #3}}}}
The third person is Nina Kaur.

So the three people visible are Liam O'Connor, Leila Hassan, and Nina Kaur."

Example 2 (2 people, bench sitting):
"The question asks about two people sitting on a bench. Looking at the image, I can see a bench with two individuals seated on it. Let me identify both of them.

First, the person on the left side of the bench appears to be relaxed and leaning back.
{{{{Identify person #1}}}}
The first person is David Kim.

Next, the person on the right side of the bench is sitting more upright.
{{{{Identify person #2}}}}
The second person is Sarah Johnson.

Both individuals are David Kim and Sarah Johnson, seated on the same bench."

Example 3 (2 people, comparative):
"To answer this comparative question, I first need to identify both people standing in the image, then determine their relative heights.

Looking at the left side, I can see one person standing with a blue jacket.
{{{{Identify person #1}}}}
This is Marcus Williams.

On the right side, there's another person standing in a grey shirt.
{{{{Identify person #2}}}}
This is Elena Petrov.

Comparing their heights in the image, Elena Petrov appears to be taller based on the relative positions of their heads and overall stature."

Now generate a response for the given question with {num_people_to_identify} identifications. Requirements:
- Start with overall observation (scene, number of people)
- State your strategy
- Identify each person sequentially with tool calls: {{{{Identify person #1}}}}, {{{{Identify person #2}}}}, etc.
- For each: visual description → tool call → acknowledgment
- End with summary connecting to question
- Be natural and conversational (3-5 sentences per person)

Output the response directly:"""

        response = self._call_llm(prompt)

        # Ensure all placeholders are present
        for person_num in identified_people:
            if f"{{{{Identify person #{person_num}}}}}" not in response:
                # Add missing placeholder
                response += f"\n{{{{Identify person #{person_num}}}}}\nThe person is {person_names[person_num-1]}."

        return response.strip()

    def generate_multi_person_questions_and_answers(self, data_entry: Dict) -> int:
        """
        Generate 1-2 multi-person Q&A pairs for images with 2+ people

        Strategy:
        - 2 people: 1 question (selective or comparative)
        - 3-4 people: 1-2 questions (group + selective/comparative)
        - 5+ people: 2 questions (group + selective)

        Returns: number of LLM calls made
        """
        annotations = data_entry["annotations"]
        num_people = len(annotations)

        # Load existing names or use provided names
        if "person_names" in data_entry:
            person_names = data_entry["person_names"]
        else:
            image_id = data_entry["image_id"]
            person_names = self.load_existing_names(image_id)
            if not person_names:
                raise ValueError(f"No person names found for image {image_id}. Run Stage 1 first.")
            data_entry["person_names"] = person_names

        # Get descriptions
        person_descs = []
        for ann in annotations:
            descriptions = ann.get("descriptions", ann.get("merged_referring_expressions", []))
            if descriptions:
                person_descs.append(descriptions[0])
            else:
                person_descs.append(f"person")

        # Determine number of questions
        if num_people == 2:
            num_questions = 1
            question_types = ["selective"]
        elif num_people in [3, 4]:
            num_questions = 2
            question_types = ["group", "selective"]
        else:  # 5+
            num_questions = 2
            question_types = ["group", "selective"]

        qna_list = []
        num_calls = 0

        for q_idx in range(num_questions):
            q_type = question_types[q_idx] if q_idx < len(question_types) else random.choice(["group", "selective"])

            # Generate question
            question = self.generate_multi_person_question(num_people, person_descs, q_type)
            num_calls += 1

            # Determine how many people to identify
            if q_type == "group":
                num_to_identify = num_people
            elif q_type == "selective" and num_people >= 3:
                num_to_identify = random.randint(2, min(3, num_people))
            else:
                num_to_identify = 2

            # Generate answer
            answer_raw = self.generate_multi_person_answer(
                question, person_names, person_descs, num_to_identify
            )
            num_calls += 1

            # Track which people are identified
            identified_nums = list(range(1, num_to_identify + 1))

            qna_list.append({
                "Q": question,
                "A_raw": answer_raw,
                "person_nums": identified_nums,
                "person_names": [person_names[i-1] for i in identified_nums],
                "question_type": q_type,
            })

        data_entry["QnA_multi"] = qna_list
        return num_calls

    def save_results(self, data_entry: Dict, output_file: str):
        """Save results to JSON file"""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_entry, f, indent=2, ensure_ascii=False)
