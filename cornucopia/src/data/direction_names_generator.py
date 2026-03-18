"""Online DPO data generation for direction-names (action-token naming).

Teaches the model to associate the special action tokens ``<forward>``,
``<clock>``, ``<anticlock>`` with their natural-language meanings.

Three subtask families:
  A. Command  -> correct action token
  B. Token    -> description
  C. Token(s) -> name of action just taken

For rejected responses we substitute the wrong action token or wrong
description text, and occasionally a pre-existing special symbol like
``<|im_end|>`` (as per user guidance).
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

_FWD  = "<forward>"
_CW   = "<clock>"
_CCW  = "<anticlock>"
_CONFUSERS = ["<|im_end|>"]

# ------------------------------------------------------------------ #
# Group A – command prompts whose correct answer is the action token
# ------------------------------------------------------------------ #
_COMMAND_ENTRIES = [
    {
        'prompts': [
            "Please go forward.",
            "Go forward:",
            "Please make the forward move",
            "Please progress",
        ],
        'chosen': [_FWD],
        'rejected_pool': [_CW, _CCW] + _CONFUSERS,
    },
    {
        'prompts': [
            "Please turn clockwise",
            "Could you turn clockwise?",
            "Just take the CW move.",
            "Take the CW move.",
            "Please take the CW move.",
        ],
        'chosen': [_CW],
        'rejected_pool': [_FWD, _CCW] + _CONFUSERS,
    },
    {
        'prompts': [
            "Please turn counter-clockwise",
            "Could you turn counter-clockwise?",
            "Just take the CCW move.",
            "Take the CCW move.",
            "Please take the CCW move.",
        ],
        'chosen': [_CCW],
        'rejected_pool': [_FWD, _CW] + _CONFUSERS,
    },
]

# ------------------------------------------------------------------ #
# Group B – "What action is <token>?" -> description
# ------------------------------------------------------------------ #
_DESCRIPTION_ENTRIES = [
    {
        'prompts': ["What action is <forward>?"],
        'chosen': ["That's a move forward"],
        'rejected_pool': [
            "That's a CW turn", "That's a clockwise turn",
            "That's a CCW turn", "That's a counter-clockwise turn",
        ],
    },
    {
        'prompts': ["What action is <clock>?"],
        'chosen': ["That's a CW turn", "That's a clockwise turn"],
        'rejected_pool': [
            "That's a move forward",
            "That's a CCW turn", "That's a counter-clockwise turn",
        ],
    },
    {
        'prompts': ["What action is <anticlock>?"],
        'chosen': ["That's a CCW turn", "That's a counter-clockwise turn"],
        'rejected_pool': [
            "That's a move forward",
            "That's a CW turn", "That's a clockwise turn",
        ],
    },
]

# ------------------------------------------------------------------ #
# Group C – token appears first, model names the action taken
# ------------------------------------------------------------------ #
_ACTION_NAME_ENTRIES = [
    {
        'prompts': [
            "<forward> What action did you just take?",
            "<forward> What was that??",
        ],
        'chosen': ["Forward!", "Forward move", "Forward move."],
        'rejected_pool': [
            "Clockwise turn!", "Clockwise turn",
            "Counter-clockwise turn", "Counterclockwise turn!",
            "I turned clockwise, sir", "I turned counter-clockwise, sir",
        ],
    },
    {
        'prompts': [
            "<clock> What action did you just take?",
            "<clock> What was that??",
        ],
        'chosen': ["Clockwise turn!", "Clockwise turn", "I turned clockwise, sir"],
        'rejected_pool': [
            "Forward!", "Forward move",
            "Counter-clockwise turn", "Counterclockwise turn!",
            "I turned counter-clockwise, sir",
        ],
    },
    {
        'prompts': [
            "<anticlock> What action did you just take?",
            "<anticlock> What was that??",
        ],
        'chosen': [
            "Counterclockwise turn!", "Counter-clockwise turn",
            "Counter-clockwise turn.", "I turned counter-clockwise, sir",
        ],
        'rejected_pool': [
            "Forward!", "Forward move",
            "Clockwise turn!", "Clockwise turn",
            "I turned clockwise, sir",
        ],
    },
]

_ALL_ENTRIES = _COMMAND_ENTRIES + _DESCRIPTION_ENTRIES + _ACTION_NAME_ENTRIES


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode='RGB')


class DirectionNamesGenerator:
    """Generates DPO sample batches for action-token naming."""

    def generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        settings_batch = get_settings_batch(batch_size)
        imgs_tensor = get_images(settings_batch, device='cpu')

        samples: List[Dict[str, Any]] = []
        for i in range(batch_size):
            pil_img = _tensor_to_pil(imgs_tensor[i])

            entry = random.choice(_ALL_ENTRIES)
            prompt   = random.choice(entry['prompts'])
            chosen   = random.choice(entry['chosen'])
            rejected = random.choice(entry['rejected_pool'])

            samples.append({
                'image': pil_img,
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
            })

        return samples
