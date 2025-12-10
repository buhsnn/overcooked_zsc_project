# utils/layout_utils.py

import random
import numpy as np
from env.overcooked_wrapper import OvercookedGym


"""
asymmetric_advantages
bottleneck
centre_objects
centre_pots
coordination_ring
corridor
counter_circuit_o_1order
cramped_room
five_by_five
forced_coordination
large_room
m_shaped_s
random0
random3
scenario1_s
scenario2
scenario2_s
scenario3
scenario4
schelling
schelling_s
simple_o
small_corridor
tutorial_0
unident
"""

####################### Full Set ######################### 

# List of available layouts in Overcooked-AI
# Option 1: easy eval layouts
# AVAILABLE_LAYOUTS = [
#     "asymmetric_advantages",
#     "bottleneck",
#     "centre_objects",
#     "coordination_ring",
#     "corridor",
#     "counter_circuit_o_1order",
#     "five_by_five",
#     "forced_coordination",
#     "large_room",
#     "m_shaped_s",
#     "random0",
#     "random3",
#     "scenario1_s",
#     "scenario2",
#     "scenario2_s",
#     "scenario3",
#     "scenario4",
#     "schelling_s",
#     "simple_o",
#     "small_corridor",
#     "tutorial_0",
#     "unident",
# ]

# EVAL_LAYOUTS = [
#     "centre_pots",
#     "cramped_room",
#     "schelling",
# ]

# Option 2: hard eval layouts
# AVAILABLE_LAYOUTS = [
#     "asymmetric_advantages",
#     "bottleneck",
#     "centre_objects",
#     "centre_pots",
#     "coordination_ring",
#     "corridor",
#     "cramped_room",
#     "five_by_five",
#     "forced_coordination",
#     "large_room",
#     "m_shaped_s",
#     "random0",
#     "random3",
#     "scenario1_s",
#     "scenario2",
#     "scenario2_s",
#     "scenario3",
#     "schelling",
#     "schelling_s",
#     "simple_o",
#     "tutorial_0",
#     "unident",
# ]

# EVAL_LAYOUTS = [
#     "small_corridor",
#     "counter_circuit_o_1order",
#     "scenario4",
# ]

# Option 3: mix
AVAILABLE_LAYOUTS = [
    "asymmetric_advantages",
    "bottleneck",
    "centre_objects",
    "coordination_ring",
    "corridor",
    "five_by_five",
    "forced_coordination",
    "large_room",
    "m_shaped_s",
    "random0",
    "random3",
    "scenario1_s",
    "scenario2",
    "scenario2_s",
    "scenario3",
    "schelling_s",
    "simple_o",
    "tutorial_0",
    "unident",
]

EVAL_LAYOUTS = [
    "centre_pots",
    "cramped_room",
    "schelling",
    "small_corridor",
    "counter_circuit_o_1order",
    "scenario4",
]


# Approximate "optimal" value per layout (for regret).
# You can adjust these values later.
APPROX_OPTIMAL_RETURN = {
    "asymmetric_advantages": 200,
    "bottleneck": 200,
    "centre_objects": 200,
    "centre_pots": 200,
    "coordination_ring": 200,
    "corridor": 200,
    "counter_circuit_o_1order": 200,
    "cramped_room": 200,
    "five_by_five": 200,
    "forced_coordination": 200,
    "large_room": 200,
    "m_shaped_s": 200,
    "random0": 200,
    "random3": 200,
    "scenario1_s": 200,
    "scenario2": 200,
    "scenario2_s": 200,
    "scenario3": 200,
    "scenario4": 200,
    "schelling": 200,
    "schelling_s": 200,
    "simple_o": 200,
    "small_corridor": 200,
    "tutorial_0": 200,
    "unident": 200,
}


####################### Minimal Set ######################### 

# AVAILABLE_LAYOUTS = [
#     "cramped_room",
#     "asymmetric_advantages",
#     "coordination_ring",
#     "forced_coordination",
# ]

# EVAL_LAYOUTS = [
#     "counter_circuit_o_1order",
# ]

# # Approximate "optimal" value per layout (for regret).
# # You can adjust these values later.
# APPROX_OPTIMAL_RETURN = {
#     "cramped_room": 200,
#     "asymmetric_advantages": 200,
#     "coordination_ring": 200,
#     "forced_coordination": 200,
#     "counter_circuit_o_1order": 200,
# }


def one_hot_layout(layout_name: str) -> np.ndarray:
    """Encode a layout as a simple one-hot vector for 'novelty'."""
    vec = np.zeros(len(AVAILABLE_LAYOUTS), dtype=float)
    if layout_name in AVAILABLE_LAYOUTS:
        idx = AVAILABLE_LAYOUTS.index(layout_name)
        vec[idx] = 1.0
    return vec


def featurize_layout(layout_name: str) -> np.ndarray:
    """Encode a layout into a handcrafted feature vector for 'novelty'."""
    vec_env = OvercookedGym(layout_name, horizon=400)
    obs_p0, obs_p1 = vec_env.env.featurize_state_mdp(vec_env.env.state)
    vec = np.concatenate([obs_p0.flatten(), obs_p1.flatten()])
    return vec


def mutate_layout(layout_name: str) -> str:
    """
    Very simple mutation:
    - randomly chooses another layout from AVAILABLE_LAYOUTS.
    - can be improved later if you want real structural mutations.
    """
    candidates = [l for l in AVAILABLE_LAYOUTS if l != layout_name]
    if not candidates:
        return layout_name
    return random.choice(candidates)
