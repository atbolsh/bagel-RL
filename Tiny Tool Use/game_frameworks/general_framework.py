# This file should have all the code shared between all (or most) of the tasks
# should have all the torch libraries I need
# Updated for QwenAgentPlayer with Qwen tokenizer

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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_transformer import QwenAgentPlayer, QwenAgentPipe, QwenExtension
from visual_transformer.model import ImageTransformerEncoder, ImageTransformerDecoder

from game import *

# Import from lightweight module at root level (tokenizer, datasets, game utilities)
# This allows the lightweight module to be imported without loading the full frameworks package
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from general_framework_lightweight import (
    device, G, game_settings, get_settings_batch, get_images, img_criterion,
    tokenizer, vocab_size, MAX_SEQ_LENGTH, QWEN_MODEL_NAME,
    encode_text, encode_batch, decode_text, decode_batch,
    ProcessBenchDataset, SampleDataset, load_text_datasets, get_text_batch,
    SPECIAL_TOKENS,
)

# Re-export model_name for backward compatibility
model_name = QWEN_MODEL_NAME

########
# Model initialization for QwenAgentPlayer
########

def create_model(model_name: str = "Qwen/Qwen3-0.6B", device=None, use_lora: bool = False):
    """
    Create a QwenAgentPlayer model.
    
    Args:
        model_name: Qwen model name/path
        device: Device to load model on
        use_lora: If True, wrap model with LoRA adapters for efficient training
        
    Returns:
        QwenAgentPlayer instance (optionally with LoRA)
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Creating QwenAgentPlayer with {model_name}...")
    model = QwenAgentPlayer(
        model_name=model_name,
        embed_dim=1024,
        num_heads=8,
        device=device,
    )
    print("Model ready!")
    
    if use_lora:
        model = apply_lora_to_text(model)
    
    return model


def apply_lora_to_text(model, r=32, lora_alpha=64, lora_dropout=0.1):
    """
    Apply LoRA adapters to the text model (Qwen) for efficient training.
    
    Note: This only applies LoRA to the Qwen language model. The image
    encoder (img_enc) and decoder (img_dec) remain fully trainable.
    
    Args:
        model: QwenAgentPlayer instance
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout rate for LoRA layers
        
    Returns:
        Model with LoRA adapters applied to text model
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("Please install peft: pip install peft")
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    
    # Apply LoRA to the underlying Qwen model
    model.pipe.model.qwen_model = get_peft_model(model.pipe.model.qwen_model, lora_config)
    print("LoRA adapters applied!")
    model.pipe.model.qwen_model.print_trainable_parameters()
    
    return model


# Create default model instance
print("Loading QwenAgentPlayer...")

# NEW METHOD: Create model and load frankenstein checkpoint (pretrained vision encoder/decoder)
# Try bf16 checkpoint first, fall back to original
FRANKENSTEIN_CHECKPOINT_BF16 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brain_checkpoints", "qwen_agent_scales_control_only_batch2000_merged.pth")
FRANKENSTEIN_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brain_checkpoints", "first_frankenstein.pt")
model = create_model(device=device)

# Model is created in float32 for training (better gradient precision)
# Load checkpoint if available
checkpoint_to_load = None
if os.path.exists(FRANKENSTEIN_CHECKPOINT_BF16):
    checkpoint_to_load = FRANKENSTEIN_CHECKPOINT_BF16
    print(f"Loading frankenstein checkpoint from {FRANKENSTEIN_CHECKPOINT_BF16}...")
elif os.path.exists(FRANKENSTEIN_CHECKPOINT):
    checkpoint_to_load = FRANKENSTEIN_CHECKPOINT
    print(f"Loading frankenstein checkpoint from {FRANKENSTEIN_CHECKPOINT}...")

if checkpoint_to_load:
    # Use strict=False to allow loading old checkpoints that don't have layer_scale_factors
    # New parameters (like layer_scale_factors) will use their default initialization
    checkpoint_state = torch.load(checkpoint_to_load, map_location=device, weights_only=True)
    # Convert checkpoint to float32 if it was saved in bf16
    for k, v in checkpoint_state.items():
        if v.dtype == torch.bfloat16:
            checkpoint_state[k] = v.float()
    load_result = model.pipe.model.load_state_dict(checkpoint_state, strict=False)
    print("Checkpoint loaded!")
    
    # Diagnostic: show what was loaded
    if load_result.missing_keys:
        print(f"  WARNING: Missing keys (using fresh init): {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"  INFO: Unexpected keys (ignored): {load_result.unexpected_keys}")
else:
    print(f"WARNING: Frankenstein checkpoint not found")
    print("Using fresh model weights. Run frankensteinify.py first to create the checkpoint.")

########
# Load default datasets
########

sdt, sdv = load_text_datasets()
num_controls = len(sdt)

########
# Adapter functions for QwenAgentPlayer
# These provide backward compatibility with old EnhancedAgentBrain/QwenBastardBrain API
########

def model_forward(model, text_batch, img_batch, ret_imgs=True, generate_image=True):
    """
    Forward pass for string text inputs.
    
    Use model_forward_with_tokens() for token tensor inputs (more efficient).
    
    Args:
        model: QwenAgentPlayer instance
        text_batch: List of strings (NOT token tensors - use model_forward_with_tokens for that)
        img_batch: Image tensor (batch_size, 3, 224, 224)
        ret_imgs: Whether to return reconstructed images
        generate_image: Whether to generate output images
        
    Returns:
        If ret_imgs: (text_probs, img_recon)
        Otherwise: text_probs
        
        text_probs is in format (batch, vocab, seq_len) for backward compatibility
    """
    # This function is for string inputs - tokenize them first for efficiency
    if isinstance(text_batch, torch.Tensor):
        # If given tokens, just use model_forward_with_tokens directly
        return model_forward_with_tokens(model, text_batch, img_batch, ret_imgs)
    
    # Tokenize the strings
    texts = text_batch if isinstance(text_batch, list) else [text_batch]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    
    # Handle single image for batch
    if img_batch.dim() == 3:
        img_batch = img_batch.unsqueeze(0)
    img_batch = img_batch.to(device)
    
    # Create attention mask
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    attention_mask = (input_ids != pad_token_id).long()
    
    # Use batch_forward with the tokenized inputs
    result = model.batch_forward(
        input_ids=input_ids,
        image=img_batch,
        attention_mask=attention_mask,
        generate_image=ret_imgs and generate_image,
    )
    
    # Extract logits and convert to old format (batch, vocab, seq_len)
    # With KV-cache approach, logits are already text-only (images are in cache)
    # No slicing needed - image_seq_len is for reference only
    logits = result['outputs'].logits  # (batch, text_seq_len, vocab)
    text_probs = logits.permute(0, 2, 1)  # (batch, vocab, text_seq_len)
    
    if ret_imgs:
        img_recon = result.get('generated_images')
        return text_probs, img_recon
    return text_probs


def model_forward_with_tokens(model, text_batch, img_batch, ret_imgs=True,
                              return_canvas_logits=False):
    """
    Forward pass using tokenized input directly.
    
    This is the primary interface for frameworks that work with token tensors.
    Operates directly on token tensors without decode/encode round-trip.
    
    Args:
        model: QwenAgentPlayer instance
        text_batch: Token tensor (batch_size, seq_len) - must be a tensor
        img_batch: Image tensor (batch_size, 3, 224, 224)
        ret_imgs: Whether to return reconstructed images
        return_canvas_logits: If True, also return VisionWeightedSum raw logits
        
    Returns:
        If ret_imgs and return_canvas_logits: (text_probs, img_recon, canvas_logits)
        If ret_imgs: (text_probs, img_recon)
        If return_canvas_logits: (text_probs, canvas_logits)
        Otherwise: text_probs
        
        text_probs is in format (batch, vocab, seq_len) for backward compatibility
        canvas_logits: raw unnormalized logits (batch, num_images, 1) for cross_entropy
    """
    # Ensure text_batch is a tensor on the right device
    if not isinstance(text_batch, torch.Tensor):
        raise ValueError("model_forward_with_tokens expects token tensors, not strings. Use encode_batch() first.")
    
    input_ids = text_batch.to(device)
    
    # Create attention mask (1 for non-pad tokens)
    # Qwen uses pad_token_id, default to 0 if not set
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    attention_mask = (input_ids != pad_token_id).long()
    
    # Ensure img_batch has correct shape
    if img_batch.dim() == 3:
        img_batch = img_batch.unsqueeze(0)
    img_batch = img_batch.to(device)
    
    # Use batch_forward directly with token tensors - no encode/decode overhead
    result = model.batch_forward(
        input_ids=input_ids,
        image=img_batch,
        attention_mask=attention_mask,
        generate_image=ret_imgs,
        return_canvas_logits=return_canvas_logits,
    )
    
    # Extract logits and convert to old format (batch, vocab, seq_len)
    # With KV-cache approach, logits are already text-only (images are in cache)
    # No slicing needed - image_seq_len is for reference only
    logits = result['outputs'].logits  # (batch, text_seq_len, vocab)
    text_probs = logits.permute(0, 2, 1)  # (batch, vocab, text_seq_len)
    
    if ret_imgs and return_canvas_logits:
        img_recon = result.get('generated_images')
        canvas_logits = result.get('canvas_logits')
        return text_probs, img_recon, canvas_logits
    if ret_imgs:
        img_recon = result.get('generated_images')
        return text_probs, img_recon
    if return_canvas_logits:
        canvas_logits = result.get('canvas_logits')
        return text_probs, canvas_logits
    return text_probs


########
# Loss functions
########

ent_criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    label_smoothing=0.01,
)

def get_text_loss(res, inputs):
    """Compute cross-entropy loss for text prediction."""
    return torch.sum(ent_criterion(res[:, :, :-1], inputs[:, 1:]))
