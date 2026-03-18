"""Online DPO data generation for gold-direction QA.

Binary task: is the agent's current heading aligned with the gold
(i.e. would moving forward eventually intersect the gold)?
"""

import os
import sys
import random
import logging
from typing import List, Dict, Any
from copy import deepcopy

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from game_frameworks.general_framework_lightweight import get_settings_batch, get_images
from game import discreteGame
from game_frameworks.game_logic_solver import will_intersect_forward

logger = logging.getLogger(__name__)

_PROMPTS = [
    "Are you facing the gold?",
    "Does your direction line up with the gold?",
    "Are you facing where the gold is?",
    "Are you facing in the right direction?",
]

_YES = ["Yep", "Absolutely.", "Certainly", "I think so.", "Uh-huh.", "Sure"]
_NO  = ["Nuh-uh", "No", "I don't think so.", "Certainly not", "Absolutely not", "Nah"]


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode='RGB')


class GoldDirectionQAGenerator:
    """Generates DPO sample batches for gold-direction QA."""

    def generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        settings_batch = get_settings_batch(batch_size)
        imgs_tensor = get_images(settings_batch, device='cpu')

        samples: List[Dict[str, Any]] = []
        for i in range(batch_size):
            s = settings_batch[i]
            pil_img = _tensor_to_pil(imgs_tensor[i])
            prompt = random.choice(_PROMPTS)

            game = discreteGame(deepcopy(s))
            if will_intersect_forward(game):
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
