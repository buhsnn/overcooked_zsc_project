# utils/layout_utils.py

import random
import numpy as np
from env.overcooked_wrapper import OvercookedGym
import json

from overcooked_ai_py.utils import read_layout_dict, write_layout_dict


train_layout_path = "./TRAIN_LAYOUTS.txt"
val_layout_path = "./VAL_LAYOUTS.txt"
approx_optimal_return_path = "./APPROX_OPTIMAL_RETURN.json"

with open(train_layout_path, "r") as f:
    AVAILABLE_LAYOUTS = [line.strip() for line in f.readlines()]
    
with open(val_layout_path, "r") as f:
    EVAL_LAYOUTS = [line.strip() for line in f.readlines()]

with open(approx_optimal_return_path, "r") as f:
    APPROX_OPTIMAL_RETURN = json.load(f)


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
