"""Online DPO data generation for blue-line QA.

Binary task: is the agent facing (roughly) the same direction as a
drawn blue arrow?  The image includes the arrow, so the model must
perceive both the agent heading and the arrow direction.

Custom image generation: each screenshot has a blue arrow drawn with
``discreteGame.draw_arrow``.
"""

import os
import sys
import math
import random
import logging
from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from game_frameworks.general_framework_lightweight import get_settings_batch
from game import discreteGame
from game_frameworks.game_logic_solver import true_angle_difference_magnitude

logger = logging.getLogger(__name__)

_PROMPTS = [
    "Are you facing the blue line?",
    "Does your direction line up with the blue line?",
    "Are you facing where it's pointing?",
    "Are you facing in the right direction?",
]

_YES = ["Yep", "Absolutely.", "Certainly", "I think so.", "Uh-huh.", "Sure"]
_NO  = ["Nuh-uh", "No", "I don't think so.", "Certainly not", "Absolutely not", "Nah"]

_ALIGNMENT_THRESHOLD = math.pi / 6


def _mod2pi(theta: float) -> float:
    rot = math.floor(theta / (2 * math.pi)) * 2 * math.pi
    return theta - rot


def _arrow_near_agent_dir(agent_direction: float) -> float:
    return _mod2pi(
        (np.random.random() * math.pi / 3)
        + agent_direction
        - (math.pi / 6)
    )


def _arrow_far_agent_dir(agent_direction: float) -> float:
    return _mod2pi(
        agent_direction
        + math.pi / 6
        + (5 * math.pi / 3) * np.random.random()
    )


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode='RGB')


class BlueLineQAGenerator:
    """Generates DPO sample batches for blue-line direction QA.

    Each sample has a custom game screenshot that includes a drawn
    blue arrow; the question is whether the agent faces the arrow.
    """

    def generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        settings_batch = get_settings_batch(batch_size)

        samples: List[Dict[str, Any]] = []
        for i in range(batch_size):
            s = settings_batch[i]

            # Decide arrow direction (50 % near, 50 % far)
            if np.random.random() < 0.5:
                arrow_dir = _arrow_near_agent_dir(s.direction)
            else:
                arrow_dir = _arrow_far_agent_dir(s.direction)

            same_direction = (
                true_angle_difference_magnitude(arrow_dir, s.direction)
                < _ALIGNMENT_THRESHOLD
            )

            # Render game + arrow
            game = discreteGame(s)
            game.draw_arrow(
                extension=1.0 + 3.0 * np.random.random(),
                direction=arrow_dir,
            )
            raw = game.getData()  # (224, 224, 3) float array in [0, 1]
            pil_img = Image.fromarray(
                (np.asarray(raw) * 255).astype(np.uint8), mode='RGB'
            )

            prompt = random.choice(_PROMPTS)
            if same_direction:
                chosen  = random.choice(_YES)
                rejected = random.choice(_NO)
            else:
                chosen  = random.choice(_NO)
                rejected = random.choice(_YES)

            samples.append({
                'image': pil_img,
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
            })

        return samples
