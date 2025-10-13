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
        person_names = data_entry.get("person_names", [])

        if not person_names:
            raise ValueError(f"No person names found in data_entry. Run single-person generation first.")

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
