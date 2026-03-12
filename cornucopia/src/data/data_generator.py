"""Data generation for tool use training."""

import json
import logging
import random
from typing import Dict, Any, List, Tuple
from datasets import Dataset, load_dataset
from ..tools.executor import ToolExecutor
import os
from transformers import AutoTokenizer



logger = logging.getLogger(__name__)


def build_tool_system_prompt(tools_config):
    """Build a system prompt describing available tools.

    Usable from both DataGenerator and ToolTrainer.
    """
    if not tools_config:
        return "You are a helpful assistant."

    tool_parts = []
    for tool in tools_config:
        part = f"- {tool['name']}: {tool['description']}"
        params = tool.get("parameters", {}).get("properties", {})
        if params:
            required = set(tool.get("parameters", {}).get("required", []))
            param_lines = []
            for pname, pinfo in params.items():
                req_tag = "required" if pname in required else "optional"
                param_lines.append(
                    f"    - {pname} ({pinfo.get('type', 'string')}, {req_tag}): "
                    f"{pinfo.get('description', '')}"
                )
            part += "\n  Parameters:\n" + "\n".join(param_lines)
        tool_parts.append(part)

    return (
        "You are a helpful assistant with access to the following tools. "
        "Use them when appropriate to help answer the user's questions.\n\n"
        "Available tools:\n"
        + "\n\n".join(tool_parts)
        + "\n\nWhen you need to use a tool, respond with a JSON object in this format:\n"
        '{"name": "tool_name", "parameters": {"param1": "value1"}}'
    )


class DataGenerator:
    """Generates training data for tool use from various sources."""
    
    def __init__(self, data_config: Dict[str, Any], tools_config: List[Dict[str, Any]], tokenizer_config: List[Dict[str, Any]]):
        self.data_config = data_config
        self.tools_config = tools_config
        self.strategy = data_config["strategy"]
        self.generation_type = data_config.get("generation_type", "online")

        if self.strategy == "position_qa":
            self.tokenizer = None
            self.tool_executor = None
            return

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["name"], trust_remote_code=tokenizer_config['trust_remote_code'])
        self.tool_executor = ToolExecutor(tools_config)
        self.system_prompt = build_tool_system_prompt(tools_config)

        # Cache special-token ids for masking
        self._im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._nl_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]

    # ------------------------------------------------------------------ #
    # Chat-template tokenisation with assistant-only masking
    # ------------------------------------------------------------------ #

    def _tokenize_chat_with_masking(self, messages):
        """Tokenize a chat-template conversation and mask non-assistant tokens.

        Returns a dict with ``input_ids`` and ``labels`` (Python lists).
        Labels for every token outside assistant content spans are set to -100.
        """
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        ids = self.tokenizer(chat_text, return_tensors="pt").input_ids[0]
        labels = ids.clone()
        labels[:] = -100

        n = len(ids)
        i = 0
        while i < n:
            if ids[i].item() == self._im_start_id:
                # Find the matching <|im_end|>
                end_pos = i + 1
                while end_pos < n and ids[end_pos].item() != self._im_end_id:
                    end_pos += 1

                # Decode role name (tokens between <|im_start|> and first \n)
                j = i + 1
                while j < end_pos and ids[j].item() != self._nl_id:
                    j += 1
                role_text = self.tokenizer.decode(ids[i + 1 : j].tolist()).strip()

                if role_text == "assistant":
                    content_start = j + 1
                    labels[content_start : end_pos + 1] = ids[content_start : end_pos + 1]

                i = end_pos + 1
            else:
                i += 1

        return {"input_ids": ids.tolist(), "labels": labels.tolist()}

    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """Prepare training and evaluation datasets."""
        if self.strategy == "toolbench" and self.generation_type.lower()=='real':
            return self._prepare_real_toolbench_data()
        elif self.strategy == "toolbench" and self.generation_type.lower()=='synthetic':
            return self._prepare_synthetic_toolbench_data()
        elif self.strategy == "stub_teacher_mode" and self.generation_type.lower()=='synthetic':
            return self._prepare_stub_teacher_mode_data()
        elif self.strategy == "manual_templates" and self.generation_type.lower()=='synthetic':
            return self._prepare_manual_template_data()
        elif self.strategy == "position_qa":
            return self._prepare_position_qa_placeholder()
        else:
            raise ValueError(f"Unknown data strategy: {self.strategy}. Data generation strategy {self.generation_type} is not implemented for {self.strategy} ")

    def _prepare_position_qa_placeholder(self) -> Tuple[Dataset, Dataset]:
        """Return tiny placeholder datasets for position_qa.

        Actual data generation happens online inside the VLM DPO trainer,
        so these are just empty shells to satisfy the train.py pipeline.
        """
        placeholder = Dataset.from_list([{"text": "placeholder"}])
        return placeholder, placeholder
    

    def _download_from_google_drive(self, folder_url, destination_dir):

        import gdown, pathlib, zipfile

        destination_dir = pathlib.Path(destination_dir)

        files = gdown.download_folder(
            url = folder_url,
            quiet = False,
            use_cookies = False,
            output = destination_dir.as_posix()

        )

        zip_path = next(p for p in files if p.endswith('data.zip'))

        print("✔ downloaded", zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(destination_dir.as_posix()+"/data")
        return 



        
        
    def _prepare_synthetic_toolbench_data(self)->Tuple[Dataset, Dataset]:

        "Get the synthetic tool bench data"

        logger.info("Generating synthetic tool bench data...")

        synthetic_data = self._generate_synthetic_toolbench_data()

        return self._split_dataset(synthetic_data)





    
    def _prepare_real_toolbench_data(self) -> Tuple[Dataset, Dataset]:
        """Get toolbench data."""
        logger.info("Obtaining toolbench data...")

        #download the data from google drive link 
        destination_dir = './data/toolbench/'
        if not os.path.exists(destination_dir):
            folder_url = 'https://drive.google.com/drive/folders/1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL'
            destination_dir = './data/toolbench/'
            self._download_from_google_drive(folder_url, destination_dir)
        
        #loading the toolbench data
        data = load_dataset("json", data_files="./data/toolbench/data/data/toolllama_G123_dfs_train.json")["train"]
        
        data = data.shuffle(seed=42).select(range(self.data_config["max_samples"]))
     
        def to_messages(conv):
             # Map any role names that can appear in ToolBench/Qwen
            role_map = {
                "system": "system",
                "user": "user",
                "assistant": "assistant",
                "tool": "tool",           # tool_response in some repos
                "function": "tool",       # treat function output the same as tool
                "tool_response": "tool",  # safety net for other dumps
                "tool_call": "assistant", # if your dump keeps the call separate
            }

            unknown = {m["from"] for m in conv} - role_map.keys()
            if unknown:                       # fail fast if you meet something new
                raise ValueError(f"Unknown role(s): {unknown}")

            return [
                {"role": role_map[m["from"]], "content": m["value"]}
                for m in conv
            ]
        
        def tokenize(sample):
            msgs = to_messages(sample["conversations"])
            tokenized = self._tokenize_chat_with_masking(msgs)
            sample["input_ids"] = tokenized["input_ids"]
            sample["labels"] = tokenized["labels"]
            return sample

        tokenised = data.map(tokenize, remove_columns=data.column_names)
        tokenised = tokenised.shuffle(seed=42).train_test_split(test_size=1-self.data_config["train_split"])

        dataset_train = tokenised["train"]
        dataset_eval = tokenised['test']

        return dataset_train, dataset_eval


    
    
    def _prepare_stub_teacher_mode_data(self) -> Tuple[Dataset, Dataset]:
        """Generate data using teacher mode (Toolformer-style).

        Each example is pre-tokenized with assistant-only label masking.
        """
        logger.info("Generating teacher mode data...")

        data = []
        for _ in range(self.data_config.get("max_samples", 100)):
            tool = random.choice(self.tools_config)
            user_query, assistant_response = self._generate_tool_qa(tool)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": assistant_response},
            ]
            data.append(self._tokenize_chat_with_masking(messages))

        logger.info(f"Generated {len(data)} teacher mode examples")
        return self._split_dataset(data)
    
    def _prepare_manual_template_data(self) -> Tuple[Dataset, Dataset]:
        """Generate data from manual templates with paraphrasing.

        Each example keeps ``user_query`` and ``assistant_response`` fields so
        that the DPO preference-dataset builder can construct prompt / chosen /
        rejected splits.  A chat-template-formatted ``text`` field is also
        included for any code path that needs the full string.
        """
        logger.info("Generating data from manual templates...")

        canonical_examples = self._create_canonical_examples()

        bootstrapped_data = []
        for example in canonical_examples:
            bootstrapped_data.append(example)
            paraphrases = self._simple_paraphrase(example)
            bootstrapped_data.extend(paraphrases)

        max_samples = self.data_config.get("max_samples", 100)
        if len(bootstrapped_data) > max_samples:
            random.shuffle(bootstrapped_data)
            bootstrapped_data = bootstrapped_data[:max_samples]

        # Format with chat template, keeping structured fields for DPO
        formatted = []
        for ex in bootstrapped_data:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": ex["user_query"]},
                {"role": "assistant", "content": ex["assistant_response"]},
            ]
            chat_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted.append({
                "text": chat_text,
                "tool_name": ex["tool_name"],
                "user_query": ex["user_query"],
                "assistant_response": ex["assistant_response"],
            })

        logger.info(f"Generated {len(formatted)} template-based examples")
        return self._split_dataset(formatted)
    
    def _generate_tool_qa(self, tool: Dict[str, Any]) -> Tuple[str, str]:
        """Generate a (user_query, assistant_response) pair for a tool."""
        tool_name = tool["name"]

        user_queries = {
            "calculator": ["What's 15 * 24?", "Can you calculate 45 + 67 - 12?"],
            "weather": ["What's the weather like in New York?", "Check London weather"],
            "search": ["Search for Python tutorials", "Find ML information"],
        }

        queries = user_queries.get(tool_name, [f"Use {tool_name}"])
        user_query = random.choice(queries)

        if tool_name == "calculator":
            expression = random.choice(["15 * 24", "45 + 67 - 12", "(100 / 5) * 3"])
            params = {"expression": expression}
        elif tool_name == "weather":
            location = random.choice(["New York", "London", "Tokyo"])
            params = {"location": location}
        elif tool_name == "search":
            query = random.choice(["Python tutorials", "machine learning"])
            params = {"query": query}
        else:
            params = {}

        self.tool_executor.execute_tool(tool_name, params)
        tool_call = json.dumps({"name": tool_name, "parameters": params})

        return user_query, tool_call

    def _generate_synthetic_toolbench_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic ToolBench-style data."""
        data = []

        for tool in self.tools_config:
            for i in range(20):
                user_query, assistant_response = self._generate_tool_qa(tool)
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": assistant_response},
                ]
                data.append(self._tokenize_chat_with_masking(messages))

        return data
    
    def _create_canonical_examples(self) -> List[Dict[str, Any]]:
        """Create canonical examples for each tool.

        Each example contains ``user_query``, ``assistant_response``, and
        ``tool_name`` so callers can build chat-template text or DPO splits.
        """
        examples = []

        templates = {
            "calculator": [
                "Calculate {expression}",
                "What is {expression}?",
                "Compute {expression}",
                "Solve {expression}",
                "Find the result of {expression}",
            ],
            "weather": [
                "What's the weather in {location}?",
                "Check weather for {location}",
                "Weather forecast for {location}",
                "How's the weather in {location}?",
                "Tell me about {location} weather",
            ],
            "search": [
                "Search for {query}",
                "Find information about {query}",
                "Look up {query}",
                "Research {query}",
                "Get results for {query}",
            ],
        }

        max_samples = self.data_config.get("max_samples", 100)
        num_tools = max(len(self.tools_config), 1)
        per_tool = max(max_samples // (num_tools * 4), 1)

        for tool in self.tools_config:
            tool_name = tool["name"]
            tool_templates = templates.get(tool_name, [f"Use {tool_name}"])

            for i in range(per_tool):
                template = random.choice(tool_templates)

                if tool_name == "calculator":
                    expressions = ["2 + 3", "10 * 5", "100 / 4", "15 - 7", "2 ** 3"]
                    expression = random.choice(expressions)
                    user_query = template.format(expression=expression)
                    params = {"expression": expression}
                elif tool_name == "weather":
                    locations = ["Paris", "Tokyo", "Sydney", "Berlin", "Cairo"]
                    location = random.choice(locations)
                    user_query = template.format(location=location)
                    params = {"location": location}
                elif tool_name == "search":
                    queries = ["Python", "AI", "cooking", "travel", "science"]
                    query = random.choice(queries)
                    user_query = template.format(query=query)
                    params = {"query": query}
                else:
                    user_query = template
                    params = {}

                result = self.tool_executor.execute_tool(tool_name, params)
                tool_call = json.dumps({"name": tool_name, "parameters": params})
                result_str = json.dumps(result)

                assistant_response = (
                    f"I'll help you with that. Let me use the {tool_name} function.\n\n"
                    f"[TOOL_CALL]{tool_call}[/TOOL_CALL]\n\n"
                    f"{result_str}\n\n"
                    f"Based on the result, the answer is "
                    f"{result.get('result', 'processed successfully')}."
                )

                examples.append({
                    "user_query": user_query,
                    "assistant_response": assistant_response,
                    "tool_name": tool_name,
                })

        return examples
    
    def _simple_paraphrase(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate simple paraphrases of an example's user query."""
        paraphrases = []
        original_query = example["user_query"]

        replacements = {
            "Calculate": "Compute",
            "What is": "What's",
            "Can you": "Could you",
            "Find": "Get",
            "Search for": "Look up",
            "weather": "forecast",
            "information": "info",
        }

        for _ in range(3):
            paraphrased = original_query
            for original, replacement in replacements.items():
                if original in paraphrased and random.random() < 0.5:
                    paraphrased = paraphrased.replace(original, replacement)

            if paraphrased != original_query:
                paraphrases.append({
                    "user_query": paraphrased,
                    "assistant_response": example["assistant_response"],
                    "tool_name": example["tool_name"],
                })

        return paraphrases
    
    def _split_dataset(self, data: List[Dict[str, Any]]) -> Tuple[Dataset, Dataset]:
        """Split data into train and eval datasets."""
        random.shuffle(data)
        
        train_split = self.data_config.get("train_split", 0.8)
        split_idx = int(len(data) * train_split)
        
        train_data = data[:split_idx]
        eval_data = data[split_idx:]
        
        # Ensure we have at least some eval data
        if len(eval_data) == 0 and len(train_data) > 1:
            eval_data = [train_data.pop()]
        
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
        
        return train_dataset, eval_dataset
    
    def _log_dataset_sample(self, dataset: Dataset, num_samples: int = 3):
        """Log a few samples from the dataset."""
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            logger.info(f"Sample {i+1}: {example}")
