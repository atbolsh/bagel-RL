# Position QA framework (formerly tutorialQA_framework)
# Task: Answer questions about relative positions (left/right/up/down) of agent and gold

from .general_framework import *
from .general_qa import *

############
# Prompts and responses for position QA tasks

# Left-right gold position
task2_prompts_lrgold = [
    "Is the gold to the left or to the right of you?",
    "Which side is it on?",
    "Is it to the left or right of the agent?",
    "Do you need to go left or right to get the gold?",
    "Please tell me whether the gold is left or right.",
    "Please tell me which side is the gold on.",
    "Which side to you need to go to get it?",
    "Which side has gold?",
    "On which side is the gold?"
]

task2_Lreplies_lrgold = ["Left", "It's to the left.", "It's on the left.", "Go left."]
task2_Rreplies_lrgold = ["Right", "It's to the right.", "It's on the right.", "Go right."]

# Up-down gold position
task2_prompts_udgold = [
    "Is the gold above or below you?",
    "Is it up or down from the agent?",
    "Do you need to go up or down to get the gold?",
    "Please tell me whether the gold is above or below you.",
    "Please tell me whether the gold is up or down.",
    "Do you need to go up or down to get it?",
    "Which side has gold?",
    "On which side is the gold?"
]

task2_Ureplies_udgold = ["Up", "Above", "It's up.", "It's above me.", "Go up."]
task2_Dreplies_udgold = ["Down", "Below", "It's down.", "It's below me.", "Go down."]

# Left-right agent position
task2_prompts_lragent = [
    "Are you to the left or right of the gold?",
    "Which side is the gold on?",
    "Is the agent to the left or right of the gold?",
    "Please tell me whether you are right or left of the gold.",
    "Please tell me which side you are relative to the gold.",
    "On which side of the gold are you?"
]

task2_Lreplies_lragent = ["Left", "I'm to the left.", "The agent is on the left."]
task2_Rreplies_lragent = ["Right", "I'm to the right.", "The agent is on the right."]

# Up-down agent position
task2_prompts_udagent = [
    "Are you below or above the gold?",
    "Is the agent above or below the gold?",
    "Please tell me whether you are up or down from the gold.",
    "Please tell me whether you are above or below the gold."
]

task2_Ureplies_udagent = ["Up", "I'm above it.", "The agent is above the gold."]
task2_Dreplies_udagent = ["Down", "I'm below it.", "The agent is below the gold."]

########
# Tensorify prompts and responses

task2_prompts_lrgold_tensor = tensorify_list(task2_prompts_lrgold)
task2_Lreplies_lrgold_tensor = tensorify_list(task2_Lreplies_lrgold)
task2_Rreplies_lrgold_tensor = tensorify_list(task2_Rreplies_lrgold)

task2_prompts_udgold_tensor = tensorify_list(task2_prompts_udgold)
task2_Ureplies_udgold_tensor = tensorify_list(task2_Ureplies_udgold)
task2_Dreplies_udgold_tensor = tensorify_list(task2_Dreplies_udgold)

task2_prompts_lragent_tensor = tensorify_list(task2_prompts_lragent)
task2_Lreplies_lragent_tensor = tensorify_list(task2_Lreplies_lragent)
task2_Rreplies_lragent_tensor = tensorify_list(task2_Rreplies_lragent)

task2_prompts_udagent_tensor = tensorify_list(task2_prompts_udagent)
task2_Ureplies_udagent_tensor = tensorify_list(task2_Ureplies_udagent)
task2_Dreplies_udagent_tensor = tensorify_list(task2_Dreplies_udagent)

########
# Compute prompt lengths

task2_prompts_lrgold_lens = get_lens(task2_prompts_lrgold_tensor)
task2_prompts_udgold_lens = get_lens(task2_prompts_udgold_tensor)
task2_prompts_lragent_lens = get_lens(task2_prompts_lragent_tensor)
task2_prompts_udagent_lens = get_lens(task2_prompts_udagent_tensor)

########
# Decision functions

# Unintuitive, but pygame flips these
# This is 'left' and 'right' relative to the game setup, not the agent
is_gold_left = (lambda settings: settings.agent_y > settings.gold[0][1])
is_gold_up = (lambda settings: settings.agent_x > settings.gold[0][0])
is_agent_left = (lambda settings: not is_gold_left(settings))
is_agent_up = (lambda settings: not is_gold_up(settings))

########
# Text generators (simple versions for training)

task2_lrgold_generator_simple = lambda settings_batch: text_generator_simple(
    settings_batch, task2_prompts_lrgold_tensor, task2_Lreplies_lrgold_tensor,
    task2_Rreplies_lrgold_tensor, task2_prompts_lrgold_lens, is_gold_left, device)

task2_udgold_generator_simple = lambda settings_batch: text_generator_simple(
    settings_batch, task2_prompts_udgold_tensor, task2_Ureplies_udgold_tensor,
    task2_Dreplies_udgold_tensor, task2_prompts_udgold_lens, is_gold_up, device)

task2_lragent_generator_simple = lambda settings_batch: text_generator_simple(
    settings_batch, task2_prompts_lragent_tensor, task2_Lreplies_lragent_tensor,
    task2_Rreplies_lragent_tensor, task2_prompts_lragent_lens, is_agent_left, device)

task2_udagent_generator_simple = lambda settings_batch: text_generator_simple(
    settings_batch, task2_prompts_udagent_tensor, task2_Ureplies_udagent_tensor,
    task2_Dreplies_udagent_tensor, task2_prompts_udagent_lens, is_agent_up, device)

text_generators_simple = [
    task2_lrgold_generator_simple,
    task2_udgold_generator_simple,
    task2_lragent_generator_simple,
    task2_udagent_generator_simple
]

########

def _qa_task_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")
    
    # Split batch across 5 generators: control + 4 QA tasks
    n_generators = 5
    chunk_size = batch_size // n_generators
    if chunk_size < 1:
        chunk_size = 1
    
    # Get settings for each QA task (4 chunks)
    S_lrg = get_settings_batch(chunk_size)
    S_udg = get_settings_batch(chunk_size)
    S_lra = get_settings_batch(chunk_size)
    S_uda = get_settings_batch(chunk_size)
    
    # Get images for each chunk
    imgs_lrg = get_images(S_lrg)
    imgs_udg = get_images(S_udg)
    imgs_lra = get_images(S_lra)
    imgs_uda = get_images(S_uda)
    
    # Generate texts for each chunk using corresponding settings
    texts_lrg = task2_lrgold_generator_simple(S_lrg)
    texts_udg = task2_udgold_generator_simple(S_udg)
    texts_lra = task2_lragent_generator_simple(S_lra)
    texts_uda = task2_udagent_generator_simple(S_uda)
    
    # Get control texts and images (control uses sdt, but still needs images)
    ind = (batch_num * chunk_size) % num_controls
    if ind + chunk_size > num_controls:
        ind = num_controls - chunk_size
    control_texts = get_text_batch(sdt, ind, chunk_size)
    S_control = get_settings_batch(chunk_size)
    imgs_control = get_images(S_control)
    
    # Pad all texts to the same length before concatenation
    # Order: lrg, udg, lra, uda, control (task first so demo images show task output)
    text_list = [texts_lrg, texts_udg, texts_lra, texts_uda, control_texts]
    max_len = max(t.size(1) for t in text_list)
    padded_texts = []
    for t in text_list:
        if t.size(1) < max_len:
            pad = torch.zeros(t.size(0), max_len - t.size(1), dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        padded_texts.append(t)
    
    # Concatenate all texts and images in consistent order
    # Order: lrg, udg, lra, uda, control (task first so demo images show task output)
    all_texts = torch.cat(padded_texts, dim=0)
    all_imgs = torch.cat([imgs_lrg, imgs_udg, imgs_lra, imgs_uda, imgs_control], dim=0)
    
    # Single forward pass with image reconstruction
    all_probs, all_recon = model_forward_with_tokens(model, all_texts, all_imgs, ret_imgs=True)
    
    # Compute text losses for each chunk
    # all_probs has shape (batch, vocab, seq_len) - slice on batch dimension (dim 0)
    text_losses = []
    for i in range(n_generators):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_probs = all_probs[start_idx:end_idx, :, :]
        chunk_texts = all_texts[start_idx:end_idx]
        text_losses.append(get_text_loss(chunk_probs, chunk_texts))
    
    # Compute image loss
    img_loss = img_criterion(all_recon, all_imgs)
    
    # Total text loss
    text_loss = sum(text_losses)
    
    # Combined loss (same weighting as control framework)
    loss = img_loss + (text_loss / 1000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()} (img: {img_loss.item()}, text: {text_loss.item()}):\n"
              f"  {text_losses[0].item()} lrg,\n"
              f"  {text_losses[1].item()} udg,\n"
              f"  {text_losses[2].item()} lra,\n"
              f"  {text_losses[3].item()} uda,\n"
              f"  {text_losses[4].item()} control\n")

    if reset_model:
        model.reset()

    return (loss.item(), text_losses[0].item(), text_losses[1].item(), text_losses[2].item(), text_losses[3].item(), text_losses[4].item(), img_loss.item())


def qa_task_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _qa_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _qa_task_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
