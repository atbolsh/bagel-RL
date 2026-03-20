"""Online DPO data generation for position QA using the game engine.

Generates batches of (image, prompt, chosen_response, rejected_response)
on-the-fly for DPO training of a VLM on spatial reasoning tasks.
"""

import os
import sys
import random
import logging
from typing import List, Dict, Any

import torch
from PIL import Image

# Add project root so game_frameworks is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from game_frameworks.general_framework_lightweight import get_settings_batch, get_images

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Position QA data (mirrored from game_frameworks/position_qa.py)
# Kept here to avoid triggering the heavy import chain in general_framework.
# --------------------------------------------------------------------------- #

# Pygame coordinate note: agent_y > gold_y means gold is LEFT, agent_x > gold_x means gold is UP.

_LR_GOLD = {
    'axis': 'lr',
    'prompts': [
        "Is the gold to the left or to the right of you?",
        "Which side is it on?",
        "Is it to the left or right of the agent?",
        "Do you need to go left or right to get the gold?",
        "Please tell me whether the gold is left or right.",
        "Please tell me which side is the gold on.",
        "Which side to you need to go to get it?",
        "Which side has gold?",
        "On which side is the gold?",
    ],
    'yes_responses': ["Left", "It's to the left.", "It's on the left.", "Go left."],
    'no_responses':  ["Right", "It's to the right.", "It's on the right.", "Go right."],
    'decision_func': lambda s: s.agent_y > s.gold[0][1],
}

_UD_GOLD = {
    'axis': 'ud',
    'prompts': [
        "Is the gold above or below you?",
        "Is it up or down from the agent?",
        "Do you need to go up or down to get the gold?",
        "Please tell me whether the gold is above or below you.",
        "Please tell me whether the gold is up or down.",
        "Do you need to go up or down to get it?",
        "Which side has gold?",
        "On which side is the gold?",
    ],
    'yes_responses': ["Up", "Above", "It's up.", "It's above me.", "Go up."],
    'no_responses':  ["Down", "Below", "It's down.", "It's below me.", "Go down."],
    'decision_func': lambda s: s.agent_x > s.gold[0][0],
}

_LR_AGENT = {
    'axis': 'lr',
    'prompts': [
        "Are you to the left or right of the gold?",
        "Which side is the gold on?",
        "Is the agent to the left or right of the gold?",
        "Please tell me whether you are right or left of the gold.",
        "Please tell me which side you are relative to the gold.",
        "On which side of the gold are you?",
    ],
    'yes_responses': ["Left", "I'm to the left.", "The agent is on the left."],
    'no_responses':  ["Right", "I'm to the right.", "The agent is on the right."],
    'decision_func': lambda s: s.agent_y <= s.gold[0][1],   # is_agent_left = NOT is_gold_left
}

_UD_AGENT = {
    'axis': 'ud',
    'prompts': [
        "Are you below or above the gold?",
        "Is the agent above or below the gold?",
        "Please tell me whether you are up or down from the gold.",
        "Please tell me whether you are above or below the gold.",
    ],
    'yes_responses': ["Up", "I'm above it.", "The agent is above the gold."],
    'no_responses':  ["Down", "I'm below it.", "The agent is below the gold."],
    'decision_func': lambda s: s.agent_x <= s.gold[0][0],   # is_agent_up = NOT is_gold_up
}

QA_TYPES = [_LR_GOLD, _UD_GOLD, _LR_AGENT, _UD_AGENT]

# --------------------------------------------------------------------------- #


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (3, H, W) float [0-1] tensor to a PIL RGB image."""
    arr = (tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode='RGB')


class PositionQAGenerator:
    """Generates DPO sample batches for position-QA from the game engine."""

    def __init__(self, cross_axis_negative_prob: float = 0.3):
        self.cross_axis_prob = cross_axis_negative_prob
        self.qa_types = QA_TYPES

    def generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Generate *batch_size* independent DPO samples.

        Each sample is a dict with keys:
            image   : PIL.Image.Image   – the 224x224 game screenshot
            prompt  : str               – the user question
            chosen  : str               – correct answer
            rejected: str               – wrong answer
        """
        self.last_batch_task = "position_qa"
        settings_batch = get_settings_batch(batch_size)
        imgs_tensor = get_images(settings_batch, device='cpu')  # (B, 3, 224, 224)

        samples: List[Dict[str, Any]] = []
        for i in range(batch_size):
            settings = settings_batch[i]
            pil_img = _tensor_to_pil(imgs_tensor[i])

            qa = random.choice(self.qa_types)
            prompt = random.choice(qa['prompts'])

            if qa['decision_func'](settings):
                chosen = random.choice(qa['yes_responses'])
                rejected = random.choice(qa['no_responses'])
            else:
                chosen = random.choice(qa['no_responses'])
                rejected = random.choice(qa['yes_responses'])

            if random.random() < self.cross_axis_prob:
                cross_types = [t for t in self.qa_types if t['axis'] != qa['axis']]
                cross_qa = random.choice(cross_types)
                rejected = random.choice(
                    cross_qa['yes_responses'] + cross_qa['no_responses']
                )

            samples.append({
                'image': pil_img,
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
            })

        return samples
