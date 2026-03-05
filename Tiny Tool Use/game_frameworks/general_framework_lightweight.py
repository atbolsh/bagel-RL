# Lightweight utilities - no full QwenAgentPlayer model loading
# Import this for standalone training without the heavy model overhead

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from torch.utils.data import Dataset

# Add parent directory to path for game imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import discreteGame, BIG_tool_use_advanced_2_5

########
# Device
########

device = torch.device('cuda:0')  # CHANGE THIS EVERY TIME
# device = torch.device('cuda:1')  # CHANGE THIS EVERY TIME

########
# Game setup
########

game_settings = BIG_tool_use_advanced_2_5
game_settings.gameSize = 224  # for compatibility with brain's expected size
G = discreteGame(game_settings)

########
# Loss functions
########

img_criterion = nn.MSELoss()

########
# Qwen Tokenizer setup (lightweight - no model loading)
########

from transformers import AutoTokenizer
from datasets import load_dataset

QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B"
vocab_size = 151936  # Qwen vocab size

# Load the Qwen tokenizer
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)

# Special tokens for game controls
SPECIAL_TOKENS = ['<forward>', '<clock>', '<anticlock>']
tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})

# Max sequence length for text
MAX_SEQ_LENGTH = 32

def encode_text(text, max_length=MAX_SEQ_LENGTH):
    """Encode text using Qwen tokenizer, returns tensor of token ids."""
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoded['input_ids'].squeeze(0)

def encode_batch(text_list, max_length=MAX_SEQ_LENGTH):
    """Encode a batch of texts using Qwen tokenizer."""
    encoded = tokenizer(
        text_list,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoded['input_ids']

def decode_text(token_ids, skip_special_tokens=False):
    """Decode token ids back to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

def decode_batch(token_ids_batch, skip_special_tokens=False):
    """Decode a batch of token ids back to text."""
    return tokenizer.batch_decode(token_ids_batch, skip_special_tokens=skip_special_tokens)

########
# Dataset classes
########

class ProcessBenchDataset(Dataset):
    """Dataset based on Qwen's ProcessBench for training."""
    def __init__(self, split='gsm8k', seq_length=MAX_SEQ_LENGTH, device=None):
        if device is None:
            device = 'cpu'
        self.device = device
        self.seq_length = seq_length
        
        # Load ProcessBench dataset
        self.dataset = load_dataset('Qwen/ProcessBench', split=split)
        
        # Pre-tokenize all examples
        self.examples = []
        for item in self.dataset:
            # ProcessBench has various fields - we'll use the main text content
            if 'problem' in item:
                text = item['problem']
            elif 'question' in item:
                text = item['question']
            elif 'text' in item:
                text = item['text']
            else:
                text = str(item)
            
            encoded = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.seq_length,
                return_tensors='pt'
            )
            self.examples.append(encoded['input_ids'].squeeze(0))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i].to(self.device)


class SampleDataset(Dataset):
    """Legacy dataset for local text files, updated for Qwen tokenizer."""
    def __init__(self, seq_length=MAX_SEQ_LENGTH, evaluate=False, device=None):
        if device is None:
            device = 'cpu'
        self.device = device
        self.seq_length = seq_length
        
        self.examples = []
        
        # Look for data in parent directory
        data_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "text_pretraining_data"
        src_files = data_path.glob("*-eval.txt") if evaluate else data_path.glob("*-train.txt")
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if line.strip():
                    encoded = tokenizer(
                        line,
                        padding='max_length',
                        truncation=True,
                        max_length=self.seq_length,
                        return_tensors='pt'
                    )
                    self.examples.append(encoded['input_ids'].squeeze(0))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i].to(self.device)


def load_text_datasets():
    """Load the default text datasets. Returns (train_dataset, val_dataset)."""
    try:
        sdt = ProcessBenchDataset(split='gsm8k', device='cpu')
        sdv = ProcessBenchDataset(split='gsm8k', device='cpu')
        print(f"Loaded ProcessBench dataset with {len(sdt)} examples")
        return sdt, sdv
    except Exception as e:
        print(f"Warning: Could not load ProcessBench dataset: {e}")
        print("Falling back to local SampleDataset")
        sdt = SampleDataset()
        sdv = SampleDataset(evaluate=True)
        return sdt, sdv


def get_text_batch(dataset, ind, batch_size, target_device=None):
    """
    Get a batch of text tensors from a dataset.
    
    Dataset slicing (dataset[a:b]) returns a list, not a tensor.
    This function stacks the items into a proper tensor.
    """
    if target_device is None:
        target_device = device
    return torch.stack([dataset[i] for i in range(ind, ind + batch_size)]).to(target_device)

########
# Game utilities
########

def get_settings_batch(batch_size, bare=True, restrict_angles=True, max_agent_offset=2.0):
    """
    Generate a batch of game settings.
    
    Args:
        batch_size: Number of settings to generate
        bare: If True, use bare settings (no extra walls, 1 gold)
        restrict_angles: Whether to restrict wall angles
        max_agent_offset: Max distance from agent to gold (in game-widths, 0-1 normalized).
                         2.0 ensures gold can appear anywhere on the map (box always covers full game area).
    """
    if bare:
        return [G.random_bare_settings(gameSize=224, max_agent_offset=max_agent_offset) for i in range(batch_size)]
    else:
        return [G.random_settings(gameSize=224, restrict_angles=restrict_angles) for i in range(batch_size)]


def get_images(settings_batch=None, device=device, ignore_agent=False, ignore_gold=False, ignore_walls=False, batch_size=None, bare=True, restrict_angles=True, dtype=torch.float32):
    """
    Get game images as tensors.
    
    Args:
        settings_batch: List of game settings (or None to generate)
        device: Target device
        ignore_agent, ignore_gold, ignore_walls: Drawing options
        batch_size: Number of images to generate (if settings_batch is None)
        bare, restrict_angles: Generation options (if settings_batch is None)
        dtype: Output dtype (default: torch.float32 for consistency with model)
        
    Returns:
        Tensor of shape (batch_size, 3, 224, 224) in specified dtype
    """
    # If no settings provided, generate them using bare/restrict_angles flags
    if settings_batch is None:
        if batch_size is None:
            raise ValueError("Must provide either settings_batch or batch_size")
        settings_batch = get_settings_batch(batch_size, bare=bare, restrict_angles=restrict_angles)
    
    batch_size = len(settings_batch)
    img = torch.zeros(batch_size, 224, 224, 3, dtype=dtype)
    should_draw = (ignore_agent or ignore_gold or ignore_walls)
    for i in range(batch_size):
        G2 = discreteGame(settings_batch[i])
        if should_draw:
            G2.draw(ignore_agent=ignore_agent, ignore_gold=ignore_gold, ignore_wals=ignore_walls)
        img[i] = torch.tensor(G2.getData(), dtype=dtype)
    img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(device)
    return img
