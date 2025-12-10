from utils import *
from tqdm import tqdm

# LAYOUTS = [
#     "asymmetric_advantages",
#     "bottleneck",
#     "centre_objects",
#     "coordination_ring",
#     "corridor",
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
#     "schelling_s",
#     "simple_o",
#     "tutorial_0",
#     "unident",
#     "centre_pots",
#     "cramped_room",
#     "schelling",
#     "small_corridor",
#     "counter_circuit_o_1order",
#     "scenario4",
# ]


train_layout_path = "/media/yujin/AI611/overcooked_zsc_project/utils/TRAIN_LAYOUTS.txt"

with open(train_layout_path, "r") as f:
    LAYOUTS = [line.strip() for line in f.readlines()]


swap_list = [
            ['O', 'D'],
            ['O', 'S'],
            ['O', 'P'],
            ['D', 'S'],
            ['D', 'P'],
            ['S', 'P']
        ]

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
    if chars[0] not in valid_chars or chars[1] not in valid_chars:  
        raise ValueError("chars to swap not in grid")
    present = {ch for ch in chars if ch in grid_str}
    if len(present) < 2:
        raise ValueError("no all 4 chars in the grid")
    c1, c2 = chars[0], chars[1]
    print(f"Swapping {c1} <-> {c2}")
    return swap_chars(grid_str, c1, c2)

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

for layout_name in tqdm(LAYOUTS) :
    for j in range(2) :
        for i in range(6) :
            
            new_layout_name = layout_name + f'_M{j*6+i}'  
            base_layout_params = read_layout_dict(layout_name)
            grid_str = base_layout_params['grid']
            if j == 1 :
                grid_str = swap_1_and_2(grid_str)
            swap_pair = swap_list[i]
            new_grid_str = swap_ODSP(grid_str, swap_pair)
            new_layout_params = base_layout_params.copy()
            new_layout_params['grid'] = new_grid_str
            write_layout_dict(new_layout_name, new_layout_params)


        