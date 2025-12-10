# utils/layout_utils.py

import random
import numpy as np
from overcooked_ai_py.utils import read_layout_dict, write_layout_dict

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
# AVAILABLE_LAYOUTS = [
#     "asymmetric_advantages_M1",
# ]

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


def one_hot_layout(layout_name: str) -> np.ndarray:  # TODO: rather than one-hot, use embedding
    """Encode a layout as a simple one-hot vector for 'novelty'."""
    vec = np.zeros(len(AVAILABLE_LAYOUTS), dtype=float)
    if layout_name in AVAILABLE_LAYOUTS:
        idx = AVAILABLE_LAYOUTS.index(layout_name)
        vec[idx] = 1.0
    return vec

def swap_1_and_2(grid_str: str) -> str:
    rows = grid_str.split("\n")
    rows = [list(r) for r in rows]  

    pos1 = None
    pos2 = None

    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            if c == '1':
                pos1 = (i, j)
            elif c == '2':
                pos2 = (i, j)

    if pos1 is None or pos2 is None:
        raise ValueError("no two players in the layout")

    (i1, j1), (i2, j2) = pos1, pos2
    rows[i1][j1], rows[i2][j2] = rows[i2][j2], rows[i1][j1]

    return "\n".join("".join(r) for r in rows)

def swap_chars(grid_str: str, c1: str, c2: str) -> str:
    rows = grid_str.splitlines()
    rows = [list(r) for r in rows]

    for i, row in enumerate(rows):
        for j, ch in enumerate(row):
            if ch == c1:
                rows[i][j] = c2
            elif ch == c2:
                rows[i][j] = c1

    return "\n".join("".join(r) for r in rows)

def swap_ODSP(grid_str: str, chars: list) -> str:
    valid_chars = ['O', 'D', 'S', 'P']
    if len(chars) != 2:
        raise ValueError("need exactly 2 chars to swap")
    if chars[0] or chars[1] not in valid_chars:  
        raise ValueError("chars to swap not in grid")
    present = {ch for ch in chars if ch in grid_str}
    if len(present) < 2:
        raise ValueError("no all 4 chars in the grid")
    c1, c2 = valid_chars
    print(f"Swapping {c1} <-> {c2}")
    return swap_chars(grid_str, c1, c2)

def random_swap_ODSP(grid_str: str) -> str:
    valid_chars = ['O', 'D', 'S', 'P']
    present = {ch for ch in valid_chars if ch in grid_str}
    if len(present) < 2:
        raise ValueError("no all 4 chars in the grid")
    c1, c2 = random.sample(list(present), 2)
    print(f"Swapping {c1} <-> {c2}")
    return swap_chars(grid_str, c1, c2)

def mutate_layout(layout_name: str, current_levels: list) -> str:

    base_layout_name = layout_name.split('_M')[0]
    mutated_list = [base_layout_name+f'_M{i}' for i in range(12)]
    candidates = [l for l in mutated_list if l not in current_levels]
    if not candidates:
        return layout_name  # no mutation possible
    new_layout_name = random.choice(candidates)
    
    return new_layout_name
