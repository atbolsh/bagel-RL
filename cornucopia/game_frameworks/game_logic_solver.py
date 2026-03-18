# Game logic solver - utilities for computing optimal moves
# Mostly used to train the agent when it's facing the gold and when it isn't

import os
import sys
import math
from copy import deepcopy

import numpy as np

# Add parent directory to path for game imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import discreteGame

tau = 2 * math.pi

# Symbol to action mapping
symbol_action_map = {1: 1, 3: 3, 4: 4, 108: 2}


def gold_direction_angle(G):
    gx, gy = G.settings.gold[0]
    ax, ay = G.settings.agent_x, G.settings.agent_y
    return G.direction_angle(ax, ay, gx, gy)


def will_intersect_forward(G):
    """
    Check if you will intersect the gold by moving forward only.
    Does not check walls or anything.
    """
    gx, gy = G.settings.gold[0]
    ax, ay = G.settings.agent_x, G.settings.agent_y
    # turns the gold into the agent's field of reference; 
    # then, check if it's in front of you and within the line you'll sweep while moving forward
    rel_gx, rel_gy = G.backRot(gx-ax, gy-ay, G.settings.direction)
    return (rel_gx > 0) and (abs(rel_gy) < G.settings.agent_r)


def true_angle_difference_magnitude(alpha, beta):
    """Compute the magnitude of the difference between two angles."""
    theta1 = (alpha - beta) % tau
    theta2 = (beta - alpha) % tau
    return min(theta1, theta2)


def should_turn_anticlockwise_forward_ENGINE(current_theta, target_theta):
    """
    Core computation for determining turn direction.
    
    Returns True if turning anticlockwise is shorter path.
    """
    cw_theta = (current_theta - target_theta) % tau
    acw_theta = (target_theta - current_theta) % tau
    return acw_theta < cw_theta


def should_turn_anticlockwise_forward(G):
    """
    True if the shortest path is turning anticlockwise (for a forward trajectory).
    False if you should turn clockwise instead.
    
    Note: turning anticlockwise increases 'direction' value,
          turning clockwise decreases 'direction' value.
    """
    gx, gy = G.settings.gold[0]
    ax, ay = G.settings.agent_x, G.settings.agent_y
    theta = G.settings.direction
    theta_to_gold = G.direction_angle(ax, ay, gx, gy)
    return should_turn_anticlockwise_forward_ENGINE(theta, theta_to_gold)


def best_move_forward(G):
    """
    Best move right now, bare settings.
    Can use this with game to create track to gold (moving only forward).
    """
    if will_intersect_forward(G):
        return 1  # action 1, G.stepForward
    else:
        if should_turn_anticlockwise_forward(G):
            return 4  # action 4, G.swivel_anticlock
        else:
            return 3  # action 3, G.swivel_clock


def trace_forward(settings, maxlen=1024, zeroPad=False, ret_rewards=False):
    """Generate trace of optimal forward-only moves to reach gold."""
    G = discreteGame(deepcopy(settings))
    reward = 0
    steps = 0
    trace = []
    if ret_rewards:
        rewards = [0]  # receive reward in next timestep
    while reward < 1e-4 and steps < maxlen - 1:
        action = best_move_forward(G)
        trace.append(action)
        res = G.actions[action]()
        reward += res
        if ret_rewards:
            rewards.append(res)
        steps += 1
    if zeroPad:
        if len(trace) < maxlen:
            trace.append(2)
            trace = trace + [0 for i in range(maxlen - len(trace))]
            if ret_rewards:
                rewards = rewards + [0 for i in range(maxlen - len(rewards))]
    if ret_rewards:
        return trace, rewards
    else:
        return trace


def _trace_forward(settings, maxlen=1024):
    """Internal version that just returns the trace."""
    return trace_forward(settings, maxlen, zeroPad=False, ret_rewards=False)


def best_move(G):
    """
    Best move right now, bare settings, both forward and backward.
    Can use this with game to create track to gold (either forward or backward).
    """
    gx, gy = G.settings.gold[0]
    ax, ay = G.settings.agent_x, G.settings.agent_y
    # turns the gold into the agent's field of reference; 
    # then, check if it's in front of you and within the line you'll sweep while moving forward
    rel_gx, rel_gy = G.backRot(gx-ax, gy-ay, G.settings.direction)

    if abs(rel_gy) < G.settings.agent_r:
        if rel_gx > 0:
            return 1
        else:
            return 108  # becomes '2' when decoded; this is the symbol used by the agent
    
    # if not, we have to figure out the correct angle
    theta = G.settings.direction
    back_theta = (theta + math.pi) % tau
    theta_to_gold = G.direction_angle(ax, ay, gx, gy)

    cw_theta = (theta - theta_to_gold) % tau
    cwb_theta = (back_theta - theta_to_gold) % tau
    acw_theta = (theta_to_gold - theta) % tau
    acwb_theta = (theta_to_gold - back_theta) % tau

    ind = np.argmin(np.array([cw_theta, cwb_theta, acw_theta, acwb_theta]))
    if ind < 2:  # cw_theta or cwb_theta
        return 3
    else:
        return 4


def trace_any(settings, maxlen=1024, zeroPad=False, ret_rewards=False):
    """Generate trace of optimal moves (forward or backward) to reach gold."""
    G = discreteGame(deepcopy(settings))
    reward = 0
    steps = 0
    trace = []
    if ret_rewards:
        rewards = [0]  # receive reward in next time step
    while reward < 1e-4 and steps < maxlen - 1:
        action = best_move(G)
        trace.append(action)
        res = G.actions[symbol_action_map[action]]()
        reward += res
        if ret_rewards:
            rewards.append(res)
        steps += 1
    if zeroPad:
        if len(trace) < maxlen:
            trace.append(2)
            trace = trace + [0 for i in range(maxlen - len(trace))]
            rewards = rewards + [0 for i in range(maxlen - len(rewards))]
    if ret_rewards:
        return trace, rewards
    else:
        return trace


def get_trace(settings, maxlen=1024, zeroPad=False, ret_rewards=False, forward_only=True):
    """
    Wrapper function to get optimal trace.
    
    Args:
        settings: Game settings
        maxlen: Maximum trace length
        zeroPad: Whether to zero-pad the trace
        ret_rewards: Whether to return rewards
        forward_only: If True, only use forward movement; otherwise allow backward
    """
    if forward_only:
        return trace_forward(settings, maxlen, zeroPad, ret_rewards)
    else:
        return trace_any(settings, maxlen, zeroPad, ret_rewards)
