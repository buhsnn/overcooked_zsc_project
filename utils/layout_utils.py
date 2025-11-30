# utils/layout_utils.py

import random
import numpy as np


# List of available layouts in Overcooked-AI
AVAILABLE_LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]


# Approximate "optimal" value per layout (for regret).
# You can adjust these values later.
APPROX_OPTIMAL_RETURN = {
    "cramped_room": 200,
    "asymmetric_advantages": 200,
    "coordination_ring": 200,
    "forced_coordination": 200,
    "counter_circuit_o_1order": 200,
}


def one_hot_layout(layout_name: str) -> np.ndarray:
    """Encode a layout as a simple one-hot vector for 'novelty'."""
    vec = np.zeros(len(AVAILABLE_LAYOUTS), dtype=float)
    if layout_name in AVAILABLE_LAYOUTS:
        idx = AVAILABLE_LAYOUTS.index(layout_name)
        vec[idx] = 1.0
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
