# This file should have all the code shared between all (or most) of the tasks
# should have all the torch libraries I need

import json
import random
import os
import sys
from pathlib import Path

# Load environment variables from .env file (for HF_TOKEN, etc.)
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# Add parent directory to path so `game` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import *

from .general_framework_lightweight import (
    device, G, game_settings, get_settings_batch, get_images, img_criterion,
    tokenizer, vocab_size, MAX_SEQ_LENGTH, QWEN_MODEL_NAME,
    encode_text, encode_batch, decode_text, decode_batch,
    ProcessBenchDataset, FineWebEduDataset, load_text_datasets, get_text_batch,
    SPECIAL_TOKENS,
)

# Re-export model_name for backward compatibility
model_name = QWEN_MODEL_NAME

########
# Load default datasets
########

sdt, sdv = load_text_datasets()
num_controls = len(sdt)
