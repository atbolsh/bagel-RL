"""Online DPO data generation for near-gold QA.

Binary task: is the agent close to the gold (Euclidean distance < 0.15)?
"""

import os
import sys
import random
import logging
from typing import List, Dict, Any

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from game_frameworks.general_framework_lightweight import get_settings_batch, get_images

logger = logging.getLogger(__name__)

_PROXIMITY_THRESHOLD = 0.15

_PROMPTS = [
    "Are you near the gold?",
    "Does your direction line up with the gold?",
    "Are so close to the gold you're salivating?",
    "Is the meal right in front of you?",
    "Are you almost at the reward?",
    "The coin's right there, yes?",
]

_YES = ["Yep", "Absolutely.", "Certainly", "I think so.", "Uh-huh.", "Sure"]
_NO  = ["Nuh-uh", "No", "I don't think so.", "Certainly not", "Absolutely not", "Nah"]


def _gold_is_near(s) -> bool:
    dx = s.agent_x - s.gold[0][0]
    dy = s.agent_y - s.gold[0][1]
    return (dx * dx + dy * dy) < _PROXIMITY_THRESHOLD ** 2


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode='RGB')


class NearGoldQAGenerator:
    """Generates DPO sample batches for gold-proximity QA."""

    def generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        settings_batch = get_settings_batch(batch_size)
        imgs_tensor = get_images(settings_batch, device='cpu')

        samples: List[Dict[str, Any]] = []
        for i in range(batch_size):
            s = settings_batch[i]
            pil_img = _tensor_to_pil(imgs_tensor[i])
            prompt = random.choice(_PROMPTS)

            if _gold_is_near(s):
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
