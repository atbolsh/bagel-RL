"""Online DPO data generation for relative-position QA.

Three subtasks:
  1. Will the agent intersect the gold going forward? (yes/no)
  2. Which way should it turn? (CW/CCW)
  3. What is the best next move? (forward/CW/CCW)
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
from game_frameworks.game_logic_solver import (
    will_intersect_forward,
    should_turn_anticlockwise_forward,
    best_move_forward,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# QA definitions
# ------------------------------------------------------------------ #

_WILL_INTERSECT = {
    'prompts': [
        "If you go forward, will you eat?",
        "Is the gold in your path?",
        "How do you figure, will you intersect it just by going forward?",
        "Is going forward enough?",
        "Can you get the gold without turning?",
        "You don't need to turn, right?",
    ],
    'yes_responses': ["Yep", "Absolutely.", "Certainly", "I think so.", "Uh-huh.", "Sure"],
    'no_responses':  ["Nuh-uh", "No", "I don't think so.", "Certainly not", "Absolutely not", "Nah"],
}

_WHICH_WAY_TURN = {
    'prompts': [
        "Which way should you turn, do you figure?",
        "Damn, how can I twist in the right direction?",
        "Which way to fix our direction?",
        "How should you turn?",
    ],
    'cw_responses':  ["Clockwise", "I should turn clockwise", "CW", "Clockwise, sir!"],
    'ccw_responses': ["Counter-clockwise", "I should turn counter-clockwise", "CCW", "Counter-clockwise, sir!"],
}

_WHAT_NEXT_MOVE = {
    'prompts': [
        "Damn it, what's the move here, partner?",
        "What should you do here?",
        "In this position, what should you do?",
        "How do you figure, what's the move for us?",
        "What's the move?",
    ],
    'forward_responses': ["Just go straight.", "We just go straight", "Full speed ahead!"],
    'cw_responses':      ["Clockwise", "I should turn clockwise", "CW", "Clockwise, sir!"],
    'ccw_responses':     ["Counter-clockwise", "I should turn counter-clockwise", "CCW", "Counter-clockwise, sir!"],
}

_MOVE_INDEX = {1: 0, 3: 1, 4: 2}


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode='RGB')


class RelpositionQAGenerator:
    """Generates DPO sample batches for relative-position QA."""

    def generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        settings_batch = get_settings_batch(batch_size)
        imgs_tensor = get_images(settings_batch, device='cpu')

        samples: List[Dict[str, Any]] = []
        for i in range(batch_size):
            s = settings_batch[i]
            pil_img = _tensor_to_pil(imgs_tensor[i])
            game = discreteGame(deepcopy(s))

            subtask = random.randint(0, 2)

            if subtask == 0:
                prompt = random.choice(_WILL_INTERSECT['prompts'])
                if will_intersect_forward(game):
                    chosen  = random.choice(_WILL_INTERSECT['yes_responses'])
                    rejected = random.choice(_WILL_INTERSECT['no_responses'])
                else:
                    chosen  = random.choice(_WILL_INTERSECT['no_responses'])
                    rejected = random.choice(_WILL_INTERSECT['yes_responses'])

            elif subtask == 1:
                prompt = random.choice(_WHICH_WAY_TURN['prompts'])
                if not should_turn_anticlockwise_forward(game):
                    chosen  = random.choice(_WHICH_WAY_TURN['cw_responses'])
                    rejected = random.choice(_WHICH_WAY_TURN['ccw_responses'])
                else:
                    chosen  = random.choice(_WHICH_WAY_TURN['ccw_responses'])
                    rejected = random.choice(_WHICH_WAY_TURN['cw_responses'])

            else:
                prompt = random.choice(_WHAT_NEXT_MOVE['prompts'])
                correct_idx = _MOVE_INDEX[best_move_forward(game)]
                pools = [
                    _WHAT_NEXT_MOVE['forward_responses'],
                    _WHAT_NEXT_MOVE['cw_responses'],
                    _WHAT_NEXT_MOVE['ccw_responses'],
                ]
                chosen = random.choice(pools[correct_idx])
                wrong_idx = random.choice([j for j in range(3) if j != correct_idx])
                rejected = random.choice(pools[wrong_idx])

            samples.append({
                'image': pil_img,
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
            })

        return samples
