# Direction Names framework
# Task: Learn to associate action tokens with their names

from .general_framework import *
from .general_qa import *

prompts_for_action_names = [
    "Please go forward.<forward>",
    "Go forward:<forward>",
    "Please make the forward move<forward>",
    "Please progress<forward>",
    "Please turn clockwise <clock>",
    "Could you turn clockwise?<clock>Sure!",
    "Just take the CW move.<clock>",
    "Take the CW move.<clock>",
    "Please take the CW move.<clock>",
    "Please turn counter-clockwise <anticlock>",
    "Could you turn counter-clockwise?<anticlock>Sure!",
    "Just take the CCW move.<anticlock>",
    "Take the CCW move.<anticlock>",
    "Please take the CCW move.<anticlock>",
    "What action is <forward>? That's a move forward",
    "What action is <clock>? That's a CW turn",
    "What action is <clock>? That's a clockwise turn",
    "What action is <anticlock>? That's a CCW turn",
    "What action is <anticlock>? That's a counter-clockwise turn",
    "<forward> What action did you just take? Forward!",
    "<clock> What action did you just take? Clockwise turn!",
    "<anticlock> What action did you just take? Counterclockwise turn!",
    "<forward> What action did you just take? Forward move",
    "<clock> What action did you just take? Clockwise turn",
    "<anticlock> What action did you just take? I turned counter-clockwise, sir",
    "<forward> What action did you just take? Forward move",
    "<clock> What action did you just take? I turned clockwise, sir",
    "<anticlock> What action did you just take? Counter-clockwise turn",
    "<forward> What was that?? Forward move.",
    "<clock> What was that?? Clockwise turn",
    "<anticlock> What was that?? Counter-clockwise turn."
]

prompts_for_action_names_tensor = tensorify_list(prompts_for_action_names)


def _direction_names_batch(batch_size, model, optimizer=None, batch_num=0, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if training and model_eval:
        raise ValueError("Cannot be training and model_eval cannot both be True")
    
    if model_eval:
        model.pipe.model.eval()

    if training:
        model.pipe.model.train()

    if training and (optimizer is None):
        raise ValueError("Must provide an optimizer if training")
    
    # Split batch across 2 generators: task + control; remainder goes to a random chunk
    n_generators = 2
    chunk_size = batch_size // n_generators
    if chunk_size < 1:
        chunk_size = 1
    remainder = batch_size - n_generators * chunk_size
    chunk_sizes = [chunk_size] * n_generators
    if remainder > 0:
        chunk_sizes[random.randint(0, n_generators - 1)] += remainder
    
    # Task chunk
    S_task = get_settings_batch(chunk_sizes[0])
    imgs_task = get_images(S_task)
    texts_direction_names = simple_sample(chunk_sizes[0], prompts_for_action_names_tensor, device=device)
    
    # Control chunk
    ind = (batch_num * chunk_sizes[1]) % num_controls
    if ind + chunk_sizes[1] > num_controls:
        ind = num_controls - chunk_sizes[1]
    control_texts = get_text_batch(sdt, ind, chunk_sizes[1])
    S_control = get_settings_batch(chunk_sizes[1])
    imgs_control = get_images(S_control)
    
    # Pad texts to same length
    text_list = [texts_direction_names, control_texts]
    max_len = max(t.size(1) for t in text_list)
    padded_texts = []
    for t in text_list:
        if t.size(1) < max_len:
            pad = torch.zeros(t.size(0), max_len - t.size(1), dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        padded_texts.append(t)
    
    all_texts = torch.cat(padded_texts, dim=0)
    all_imgs = torch.cat([imgs_task, imgs_control], dim=0)
    
    # Single forward pass
    all_probs, all_recon = model_forward_with_tokens(model, all_texts, all_imgs, ret_imgs=True)
    
    # Text losses per chunk
    text_losses = []
    offset = 0
    for cs in chunk_sizes:
        chunk_probs = all_probs[offset:offset + cs, :, :]
        chunk_texts = all_texts[offset:offset + cs]
        text_losses.append(get_text_loss(chunk_probs, chunk_texts))
        offset += cs
    
    img_loss = img_criterion(all_recon, all_imgs)
    text_loss = sum(text_losses)
    loss = img_loss + (text_loss / 1000)

    if training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.soft_reset()

    if printing:
        print(f"Total loss: {loss.item()} (img: {img_loss.item()}, text: {text_loss.item()}):\n"
              f"  {text_losses[0].item()} direction naming,\n"
              f"  {text_losses[1].item()} control\n")

    if reset_model:
        model.reset()

    return (loss.item(), text_losses[0].item(), text_losses[1].item(), img_loss.item())


def direction_names_batch(batch_size, model, optimizer=None, batch_num=0, compute_grad=False, random_order=True, model_eval=True, reset_model=True, printing=True, training=False, use_lora=False):
    if compute_grad:
        return _direction_names_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
    else:
        if training:
            raise ValueError("If training is True, compute_grad must also be True")
        with torch.no_grad():
            return _direction_names_batch(batch_size, model, optimizer, batch_num, random_order, model_eval, reset_model, printing, training, use_lora)
