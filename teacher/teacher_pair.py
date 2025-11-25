# teacher/teacher_agent.py

from overcooked_ai_py.agents.agent import GreedyHumanModel, AgentPair
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv


def make_teacher_env_and_pair(layout_name: str = "cramped_room", horizon: int = 400):
    """
    Build a Teacher based on the official Overcooked planners.

    Returns
    -------
    base_env : OvercookedEnv
        Base environment (same as used by AgentEvaluator).
    teacher_pair : AgentPair
        (Teacher0, Teacher1), both GreedyHumanModel agents.
    """

    # 1) Create MDP + base env for this layout
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon)

    # 2) Medium-level action manager (path-planning & high-level goals)
    mlam = base_env.mlam  # computed lazily inside the env

    # 3) Two “teacher” agents using the same mlam
    teacher0 = GreedyHumanModel(mlam)
    teacher1 = GreedyHumanModel(mlam)

    # 4) AgentPair = what AgentEvaluator expects
    teacher_pair = AgentPair(teacher0, teacher1, allow_duplicate_agents=False)

    return base_env, teacher_pair
