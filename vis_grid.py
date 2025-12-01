
import os
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
layouts_list = sorted(os.listdir("/media/yujin/AI611/overcooked_ai/src/overcooked_ai_py/data/layouts"))
for layout in layouts_list:
    layout_name = layout.split(".")[0]
    mdp = OvercookedGridworld.from_layout_name(
        layout_name,
        # old_dynamics=True,
    )
    save_path = "/media/yujin/AI611/overcooked_zsc_project/layout_vis/" + layout_name + ".png"
    StateVisualizer().render_env(mdp.terrain_mtx, save_path)
