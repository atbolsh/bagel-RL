"""Online DPO data generation for game-state comparison (v1).

The agent is shown two game states side-by-side (left / right in a
single composite image) and must decide which state is closer to the
gold, measured by optimal-path length (``trace_forward``).
"""

import os
import sys
import random
import logging
from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from game_frameworks.general_framework_lightweight import get_settings_batch, get_images
from game_frameworks.game_logic_solver import trace_forward

logger = logging.getLogger(__name__)

_PROMPTS = [
    "Two game states are shown side by side. In which one will the agent reach the gold faster - left or right?",
    "Looking at these two game states, which is better positioned - left or right?",
    "Compare these two states: is the left or right one closer to reaching the gold?",
    "Pick the better game state: left or right?",
    "Which of these two positions do you prefer - left or right?",
]

_LEFT  = ["Left", "The left one.", "Left for sure.", "I think left.", "Left of course"]
_RIGHT = ["Right", "The right one.", "Right for sure.", "I think right.", "Right of course"]


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode='RGB')


def _make_composite(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """Place two images side by side with a thin gray divider."""
    w, h = img1.size
    gap = 4
    composite = Image.new('RGB', (w * 2 + gap, h), (128, 128, 128))
    composite.paste(img1, (0, 0))
    composite.paste(img2, (w + gap, 0))
    return composite


class ComparisonV1Generator:
    """Generates DPO sample batches for game-state comparison."""

    def generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        s1_batch = get_settings_batch(batch_size)
        s2_batch = get_settings_batch(batch_size)
        imgs1 = get_images(s1_batch, device='cpu')
        imgs2 = get_images(s2_batch, device='cpu')

        samples: List[Dict[str, Any]] = []
        for i in range(batch_size):
            pil1 = _tensor_to_pil(imgs1[i])
            pil2 = _tensor_to_pil(imgs2[i])
            composite = _make_composite(pil1, pil2)

            wait1 = len(trace_forward(s1_batch[i]))
            wait2 = len(trace_forward(s2_batch[i]))
            left_is_better = (wait1 <= wait2)

            prompt = random.choice(_PROMPTS)
            if left_is_better:
                chosen  = random.choice(_LEFT)
                rejected = random.choice(_RIGHT)
            else:
                chosen  = random.choice(_RIGHT)
                rejected = random.choice(_LEFT)

            samples.append({
                'image': composite,
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
            })

        return samples
