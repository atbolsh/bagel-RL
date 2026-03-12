"""Data preparation for prompt_swallowing training.

Produces task examples (system+user+assistant) and control texts.
The dataset interleaves three batch types:
  - swallowing  (50%): teacher=[prompt][task], student=[task]
  - unswallowed (25%): teacher=student=[prompt][task]
  - control     (25%): teacher=student=[control text]

Indexing is deterministic: given the same idx you get the same sample.
The batch-type assignment and the underlying example are both derived
from idx, so results are reproducible.
"""

import logging
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset

from .data_generator import DataGenerator, build_tool_system_prompt
from .control_dataset import load_control_texts

logger = logging.getLogger(__name__)


def prompt_swallowing_collate_fn(batch: List[Dict], pad_token_id: int, max_length: int):
    """Collate batch: pad to max_length and stack into tensors."""
    teacher_ids = []
    student_ids = []
    prompt_lens = []
    task_lens = []

    for b in batch:
        t_ids = b["teacher_input_ids"][:max_length]
        s_ids = b["student_input_ids"][:max_length]
        t_ids = t_ids + [pad_token_id] * (max_length - len(t_ids))
        s_ids = s_ids + [pad_token_id] * (max_length - len(s_ids))
        teacher_ids.append(t_ids)
        student_ids.append(s_ids)
        prompt_lens.append(b["prompt_len"])
        task_lens.append(b["task_len"])

    return {
        "teacher_input_ids": torch.tensor(teacher_ids, dtype=torch.long),
        "student_input_ids": torch.tensor(student_ids, dtype=torch.long),
        "prompt_len": prompt_lens,
        "task_len": task_lens,
        "batch_types": [b["batch_type"] for b in batch],
    }


class PromptSwallowingDataset(Dataset):
    """Deterministic dataset for prompt_swallowing.

    Layout (contiguous index ranges, so DataLoader shuffling works):
      [0, n_swallow)                          -> swallowing
      [n_swallow, n_swallow + n_unswallowed)  -> unswallowed
      [n_swallow + n_unswallowed, total)      -> control
    """

    BATCH_TYPE_SWALLOWING = "swallowing"
    BATCH_TYPE_UNSWALLOWED = "unswallowed"
    BATCH_TYPE_CONTROL = "control"

    def __init__(
        self,
        task_examples: List[Dict[str, Any]],
        control_texts: List[str],
        tokenizer,
        system_prompt: str,
        max_length: int = 512,
        swallow_frac: float = 0.5,
        unswallowed_frac: float = 0.25,
    ):
        if not task_examples:
            raise ValueError("task_examples cannot be empty")
        if not control_texts:
            raise ValueError("control_texts cannot be empty")

        self.task_examples = task_examples
        self.control_texts = control_texts
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length

        n_task = len(task_examples)
        n_ctrl = len(control_texts)
        total = 2 * n_task + n_ctrl

        self.n_swallow = max(1, round(total * swallow_frac))
        self.n_unswallowed = max(1, round(total * unswallowed_frac))
        self.n_control = max(1, total - self.n_swallow - self.n_unswallowed)

    def __len__(self) -> int:
        return self.n_swallow + self.n_unswallowed + self.n_control

    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
            add_special_tokens=True,
        )["input_ids"]

    def _get_task_full_text(self, ex: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": ex["user_query"]},
            {"role": "assistant", "content": ex["assistant_response"]},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def _get_task_only_text(self, ex: Dict[str, Any]) -> str:
        messages = [
            {"role": "user", "content": ex["user_query"]},
            {"role": "assistant", "content": ex["assistant_response"]},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < self.n_swallow:
            ex = self.task_examples[idx % len(self.task_examples)]
            teacher_ids = self._tokenize(self._get_task_full_text(ex))
            student_ids = self._tokenize(self._get_task_only_text(ex))
            prompt_len = len(teacher_ids) - len(student_ids)
            if prompt_len < 0:
                prompt_len = 0
            return {
                "batch_type": self.BATCH_TYPE_SWALLOWING,
                "teacher_input_ids": teacher_ids,
                "student_input_ids": student_ids,
                "prompt_len": prompt_len,
                "task_len": len(student_ids),
            }

        elif idx < self.n_swallow + self.n_unswallowed:
            local = idx - self.n_swallow
            ex = self.task_examples[local % len(self.task_examples)]
            ids = self._tokenize(self._get_task_full_text(ex))
            return {
                "batch_type": self.BATCH_TYPE_UNSWALLOWED,
                "teacher_input_ids": ids,
                "student_input_ids": list(ids),
                "prompt_len": 0,
                "task_len": len(ids),
            }

        else:
            local = idx - self.n_swallow - self.n_unswallowed
            text = self.control_texts[local % len(self.control_texts)]
            ids = self._tokenize(text)
            return {
                "batch_type": self.BATCH_TYPE_CONTROL,
                "teacher_input_ids": ids,
                "student_input_ids": list(ids),
                "prompt_len": 0,
                "task_len": len(ids),
            }


def prepare_prompt_swallowing_datasets(
    data_config: Dict[str, Any],
    tools_config: List[Dict[str, Any]],
    tokenizer_config: Dict[str, Any],
    max_length: int = 512,
) -> Tuple[PromptSwallowingDataset, PromptSwallowingDataset]:
    """Prepare train and eval datasets for prompt_swallowing.

    Uses ``generate_diverse_tool_examples`` for high-variety task data
    at scale (supports 50k+ unique examples).
    """
    # We need a DataGenerator for the tokenizer and tool executor
    data_config_stub = {**data_config, "strategy": "manual_templates"}
    data_gen = DataGenerator(data_config_stub, tools_config, tokenizer_config)

    max_samples = data_config.get("max_samples", 500)
    train_split = data_config.get("train_split", 0.85)
    n_train = int(max_samples * train_split)
    n_eval = max_samples - n_train

    logger.info(f"Generating {n_train} train + {n_eval} eval task examples...")
    task_examples_train = data_gen.generate_diverse_tool_examples(n_train)
    task_examples_eval = data_gen.generate_diverse_tool_examples(n_eval)

    max_control = data_config.get("control_max_samples", 5000)
    control_texts = load_control_texts(max_samples=max_control)
    if len(control_texts) < 100:
        logger.warning(f"Only {len(control_texts)} control texts loaded")

    system_prompt = build_tool_system_prompt(tools_config)

    train_dataset = PromptSwallowingDataset(
        task_examples=task_examples_train,
        control_texts=control_texts,
        tokenizer=data_gen.tokenizer,
        system_prompt=system_prompt,
        max_length=max_length,
    )
    eval_dataset = PromptSwallowingDataset(
        task_examples=task_examples_eval,
        control_texts=control_texts,
        tokenizer=data_gen.tokenizer,
        system_prompt=system_prompt,
        max_length=max_length,
    )

    return train_dataset, eval_dataset
